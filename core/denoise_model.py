# core/denoise_model.py  (slim：数值主干 + 共享表示 + 类别输出头；可选 AR 类别头)
from __future__ import annotations
import torch
import torch.nn as nn

from core.denoise_utils import (
    timestep_embedding, PeriodicEmbedding, MLPResidualBlock,
    ColumnTypeEmbedding, MultiScaleNumEncoder,
)

AR_EMB_DIM = 16  # AR 头的类别 embedding 维度（固定，不作为超参）


class TabMLPDenoise(nn.Module):
    def __init__(
        self,
        task,
        info,
        num_classes_cond: int = 2,
        dim_t: int = 64,
        dim_hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_frequencies: int = 16,
        nhead_col: int = 4,
        few_num_threshold: int = 3,
        n_expand: int = 3,
        share_decoder: bool = True,
        cat_ar: bool = False,
    ):
        super().__init__()
        self.task          = task
        self.dim_t         = dim_t
        self.dim_hidden    = dim_hidden
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']          # list[int]
        true_marginals     = info.get('true_marginals', None)
        self.share_decoder = share_decoder
        self.cat_ar        = bool(cat_ar) and len(self.num_classes) > 0

        # ── cat_loss.py 兼容接口 ─────────────────────────────────────────
        self.num_numerical_features = self.num_numerical
        _start = 0
        self.slices_for_classes = []
        for n_c in self.num_classes:
            self.slices_for_classes.append(slice(_start, _start + n_c))
            _start += n_c
        self.true_marginals = true_marginals

        # ── 少数值列模式（保持不变）──────────────────────────────────────
        self.few_num_mode = (
            0 < self.num_numerical <= few_num_threshold and n_expand > 1
        )
        self.n_expand = n_expand if self.few_num_mode else 1
        self._eff_num_tokens = self.num_numerical * self.n_expand
        if self.few_num_mode:
            print(
                f"[TabMLPDenoise] few_num_mode 激活：num_numerical={self.num_numerical}，"
                f"每列扩展为 {n_expand} 个 Token，有效 Token 数 {self._eff_num_tokens}"
            )

        # ── 1. 数值列编码 ────────────────────────────────────────────────
        if self.num_numerical > 0:
            if self.few_num_mode:
                self.num_col_encoders = nn.ModuleList([
                    MultiScaleNumEncoder(dim_hidden, n_expand, n_frequencies)
                    for _ in range(self.num_numerical)
                ])
            else:
                self.num_col_encoders = nn.ModuleList([
                    PeriodicEmbedding(dim_hidden, n_frequencies)
                    for _ in range(self.num_numerical)
                ])
        else:
            self.num_col_encoders = None

        # ── 2. 列类型嵌入（仅数值 token，保持不动）──────────────────────
        self.col_type_emb = ColumnTypeEmbedding(dim_hidden)

        # ── 3. 数值列间 Transformer ──────────────────────────────────────
        if self._eff_num_tokens > 1:
            self.num_transformer = nn.TransformerEncoderLayer(
                d_model=dim_hidden, nhead=nhead_col,
                dim_feedforward=dim_hidden * 2, dropout=dropout,
                batch_first=True, norm_first=True,
            )
        else:
            self.num_transformer = None

        # ── 4. CLS 聚合 ──────────────────────────────────────────────────
        self.cls_num = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.pool_num_attn = nn.MultiheadAttention(
            dim_hidden, nhead_col, dropout=dropout, batch_first=True)
        self.pool_norm_num_q  = nn.LayerNorm(dim_hidden)
        self.pool_norm_num_kv = nn.LayerNorm(dim_hidden)

        # ── 5. 条件编码 ──────────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ── 6. 共享表示 h_shared（D → D）─────────────────────────────────
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.SiLU(), nn.LayerNorm(dim_hidden),
        )

        # ── 7. 双专属主干 ────────────────────────────────────────────────
        if self.num_numerical > 0:
            self.num_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout) for _ in range(num_layers)
            ])
        else:
            self.num_backbone = None

        if len(self.num_classes) > 0:
            self.cat_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout) for _ in range(num_layers)
            ])
        else:
            self.cat_backbone = None

        # ── 8. 解码器 + 输出头 ───────────────────────────────────────────
        if self.num_numerical > 0:
            if share_decoder:
                self.num_shared_dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden), nn.SiLU(),
                    nn.Dropout(dropout), nn.LayerNorm(dim_hidden),
                )
                self.out_num = nn.Linear(dim_hidden, self.num_numerical)
            else:
                self.num_shared_dec = None
                self.out_num = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden), nn.SiLU(),
                    nn.Linear(dim_hidden, self.num_numerical),
                )
        else:
            self.num_shared_dec = None
            self.out_num = None

        if len(self.num_classes) > 0:
            if share_decoder:
                self.cat_shared_dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden), nn.SiLU(),
                    nn.Dropout(dropout), nn.LayerNorm(dim_hidden),
                )
            else:
                self.cat_shared_dec = None

            if self.cat_ar:
                # AR 头：各列输出头恒为 Linear，由 GRU 状态产生 logits
                self.out_cat = nn.ModuleList([
                    nn.Linear(dim_hidden, c) for c in self.num_classes
                ])
            elif share_decoder:
                self.out_cat = nn.ModuleList([
                    nn.Linear(dim_hidden, c) for c in self.num_classes
                ])
            else:
                self.out_cat = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden), nn.SiLU(),
                        nn.Linear(dim_hidden, c),
                    ) for c in self.num_classes
                ])
        else:
            self.cat_shared_dec = None
            self.out_cat = None

        # ── 9. AR 类别头（共享 GRUCell + 逐列 embedding）────────────────
        if self.cat_ar:
            self.ar_init  = nn.Sequential(nn.Linear(dim_hidden, dim_hidden), nn.SiLU())
            self.ar_cell  = nn.GRUCell(AR_EMB_DIM, dim_hidden)
            # 最后一列之后不再更新状态，所以只需要前 m-1 列的 embedding
            self.ar_embed = nn.ModuleList([
                nn.Embedding(c, AR_EMB_DIM) for c in self.num_classes[:-1]
            ])

        if true_marginals is not None and self.out_cat is not None:
            self._apply_prior_initialization(true_marginals)

    # ────────────────────────────────────────────────────────────────────
    def _apply_prior_initialization(self, true_marginals):
        """输出头偏置 = log(prior)，权重乘 0.1。"""
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                log_prior = torch.log(true_marginals[i] + 1e-8)
                target = head if isinstance(head, nn.Linear) else head[-1]
                target.bias.copy_(log_prior)
                target.weight.data.mul_(0.1)

    # ────────────────────────────────────────────────────────────────────
    def _encode_num_tokens(self, x_num: torch.Tensor) -> list:
        num_tokens = []
        if self.num_col_encoders is None:
            return num_tokens
        for i, enc in enumerate(self.num_col_encoders):
            if self.few_num_mode:
                expanded = enc(x_num[:, i])                    # [B, n_expand, D]
                for j in range(expanded.shape[1]):
                    num_tokens.append(expanded[:, j])
            else:
                num_tokens.append(enc(x_num[:, i]))
        return num_tokens

    # ────────────────────────────────────────────────────────────────────
    def _trunk(self, x_in, timesteps, y):
        """返回 (out_n [B,N_num], h_cat [B,D] 或 None)。类别输入部分被忽略。"""
        B = x_in.shape[0]
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        emb   = t_emb + self.label_emb(y.view(-1).long())

        x_num      = x_in[:, :self.num_numerical]
        num_tokens = self._encode_num_tokens(x_num)

        if num_tokens:
            num_type_vec = self.col_type_emb.get(0, x_in.device)
            num_tokens = [tok + num_type_vec for tok in num_tokens]
            num_seq = torch.stack(num_tokens, dim=1)
            if self.num_transformer is not None:
                num_seq = self.num_transformer(num_seq)
            cls = self.cls_num.expand(B, -1, -1)
            q   = self.pool_norm_num_q(cls)
            kv  = self.pool_norm_num_kv(num_seq)
            h_pool, _ = self.pool_num_attn(q, kv, kv)
            h_num_pool = (cls + h_pool).squeeze(1)
        else:
            h_num_pool = torch.zeros(B, self.dim_hidden, device=x_in.device)

        h_shared = self.proj_fuse(h_num_pool)

        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h_num = h_shared.clone()
            for layer in self.num_backbone:
                h_num = layer(h_num, emb)
            if self.num_shared_dec is not None:
                h_num = self.num_shared_dec(h_num)
            out_n = self.out_num(h_num)

        h_cat = None
        if self.cat_backbone is not None:
            h_cat = h_shared.clone()
            for layer in self.cat_backbone:
                h_cat = layer(h_cat, emb)
            if self.cat_shared_dec is not None:
                h_cat = self.cat_shared_dec(h_cat)

        return out_n, h_cat

    # ────────────────────────────────────────────────────────────────────
    def _ar_logits(self, h_cat, cat_tf):
        """教师强制：第 i 列以真实的前 i-1 列为条件，返回拼接 logits [B, ΣK]。"""
        g, outs = self.ar_init(h_cat), []
        last = len(self.out_cat) - 1
        for i, head in enumerate(self.out_cat):
            outs.append(head(g))
            if i < last:
                g = self.ar_cell(self.ar_embed[i](cat_tf[:, i]), g)
        return torch.cat(outs, dim=-1)

    def forward(self, x_in, timesteps, y, cat_tf=None, decode_cat=True):
        """
        cat_tf    : [B, n_cat] 真实类别索引（仅 cat_ar 训练时传入）
        decode_cat: cat_ar 时，采样的非末步传 False，类别部分返回 0，省去无用计算
        """
        B = x_in.shape[0]
        out_n, h_cat = self._trunk(x_in, timesteps, y)

        if h_cat is None:
            out_c = torch.empty((B, 0), device=x_in.device)
        elif self.cat_ar:
            if decode_cat:
                out_c = self._ar_logits(h_cat, cat_tf)
            else:
                out_c = torch.zeros(B, sum(self.num_classes), device=x_in.device)
        else:
            out_c = torch.cat([head(h_cat) for head in self.out_cat], dim=-1)

        return torch.cat([out_n, out_c], dim=-1)

    @torch.no_grad()
    def sample_final(self, x_in, timesteps, y, temperature: float = 1.0):
        """cat_ar 的最终一步：返回 (out_n, 类别索引 [B, n_cat])，按数据集列顺序逐列采样。"""
        out_n, h_cat = self._trunk(x_in, timesteps, y)
        g, idx_list = self.ar_init(h_cat), []
        last = len(self.out_cat) - 1
        for i, head in enumerate(self.out_cat):
            probs = torch.softmax(head(g) / temperature, dim=-1)
            idx = torch.multinomial(probs, 1).squeeze(-1)
            idx_list.append(idx)
            if i < last:
                g = self.ar_cell(self.ar_embed[i](idx), g)
        return out_n, torch.stack(idx_list, dim=1)
        