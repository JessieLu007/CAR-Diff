# core/denoise_model_v2.py  (最终版：数值扩散主干 + 共享表示 + 时间条件的类别分类头)
from __future__ import annotations
import torch
import torch.nn as nn

from core.denoise_utils import (
    timestep_embedding, PeriodicEmbedding, MLPResidualBlock,
    ColumnTypeEmbedding, MultiScaleNumEncoder,
)


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
    ):
        super().__init__()
        self.task          = task
        self.dim_t         = dim_t
        self.dim_hidden    = dim_hidden
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']          # list[int]
        true_marginals     = info.get('true_marginals', None)
        self.share_decoder = share_decoder

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

        # ── 2. 列类型嵌入（数值 token，保持不动）────────────────────────
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
                self.out_cat = nn.ModuleList([
                    nn.Linear(dim_hidden, c) for c in self.num_classes
                ])
            else:
                self.cat_shared_dec = None
                self.out_cat = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden), nn.SiLU(),
                        nn.Linear(dim_hidden, c),
                    ) for c in self.num_classes
                ])
        else:
            self.cat_shared_dec = None
            self.out_cat = None

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
    def forward(self, x_num: torch.Tensor, timesteps: torch.Tensor,
                y: torch.Tensor, decode_cat: bool = True):
        """
        x_num     : [B, N_num] 加噪后的数值列（类别列不参与扩散，不作为输入）
        decode_cat: False 时只返回数值输出 [B, N_num]，采样的非末步使用，省去无用计算
        返回       : decode_cat=True 时 [B, N_num + ΣK]（数值噪声预测 + 各类别 logits）
        """
        B = x_num.shape[0]
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        emb   = t_emb + self.label_emb(y.view(-1).long())

        num_tokens = self._encode_num_tokens(x_num)
        if num_tokens:
            num_type_vec = self.col_type_emb.get(0, x_num.device)
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
            h_num_pool = torch.zeros(B, self.dim_hidden, device=x_num.device)

        h_shared = self.proj_fuse(h_num_pool)

        out_n = torch.empty((B, 0), device=x_num.device)
        if self.num_backbone is not None:
            h_num = h_shared.clone()
            for layer in self.num_backbone:
                h_num = layer(h_num, emb)
            if self.num_shared_dec is not None:
                h_num = self.num_shared_dec(h_num)
            out_n = self.out_num(h_num)

        if not decode_cat or self.cat_backbone is None:
            return out_n

        h_cat = h_shared.clone()
        for layer in self.cat_backbone:
            h_cat = layer(h_cat, emb)
        if self.cat_shared_dec is not None:
            h_cat = self.cat_shared_dec(h_cat)
        out_c = torch.cat([head(h_cat) for head in self.out_cat], dim=-1)

        return torch.cat([out_n, out_c], dim=-1)
        