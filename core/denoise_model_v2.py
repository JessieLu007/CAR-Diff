# core/denoise_model_v2.py  (v2 — 双路 Cross-Attention，显式跨模态路由)
from __future__ import annotations
import math
import torch
import torch.nn as nn


from core.denoise_utils import (
    timestep_embedding, PeriodicEmbedding, MLPResidualBlock, ColumnTypeEmbedding, MultiScaleNumEncoder, CrossAttentionBlock,
    NumToCatFiLM
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
        nhead_cross: int = 4,
        few_num_threshold: int = 3,
        # num_numerical <= few_num_threshold 时激活 few_num_mode；设为 0 可禁用
        n_expand: int = 3,
        # 每个数值列扩展为多少个 Token（few_num_mode 生效时）
        share_decoder: bool = True,  # 新增：解码器是否共享展开层，与 v1 对齐
        use_time_gate: bool = True,
    ):
        super().__init__()

        self.task          = task
        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']          # list[int]
        true_marginals     = info.get('true_marginals', None)
        self.share_decoder = share_decoder
        self.use_time_gate = use_time_gate

        # ── cat_loss.py 兼容接口───────────
        self.num_numerical_features = self.num_numerical
        _start = 0
        self.slices_for_classes = []
        for n_c in self.num_classes:
            self.slices_for_classes.append(slice(_start, _start + n_c))
            _start += n_c
        self.true_marginals = true_marginals

        # ── 少数值列模式判断 ─────────────────
        self.few_num_mode = (
            0 < self.num_numerical <= few_num_threshold
            and n_expand > 1
        )
        self.n_expand = n_expand if self.few_num_mode else 1
        self._eff_num_tokens = self.num_numerical * self.n_expand

        if self.few_num_mode:
            print(
                f"[TabMLPDenoise-v2] few_num_mode 激活："
                f"num_numerical={self.num_numerical} <= threshold={few_num_threshold}，"
                f"每列扩展为 {n_expand} 个 Token，"
                f"有效数值 Token 数 {self.num_numerical} → {self._eff_num_tokens}"
            )

        # ── 1. 逐列编码器 ────────────────────────────────────────────────
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

        self.cat_dropout = nn.Dropout(0.2)
        if len(self.num_classes) > 0:
            self.cat_embed_layers = nn.ModuleList([
                nn.Embedding(c, min(16, c * 4)) for c in self.num_classes
            ])
            self.cat_col_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(min(16, c * 4), dim_hidden),
                    nn.SiLU(),
                    nn.LayerNorm(dim_hidden),
                )
                for c in self.num_classes
            ])
        else:
            self.cat_embed_layers = None
            self.cat_col_projs    = None

        # ── 2. 可学习门控尺度 ────────────────────────────────────────────
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.gate_layer = nn.Linear(dim_hidden, dim_hidden)
            nn.init.xavier_uniform_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, -1.0)

            self.cat_to_num = nn.Linear(dim_hidden, dim_hidden * 2)
            nn.init.zeros_(self.cat_to_num.weight)
            nn.init.zeros_(self.cat_to_num.bias)

            if self.task in ('minority_cond', 'minority_uncond'):
                self.gate_scale = nn.Parameter(torch.zeros(1))
                print(f"task = {self.task}, 初始化gate_scale为nn.Parameter(torch.zeros(1))")
            elif self.task == 'full_data':
                self.gate_scale = nn.Parameter(torch.full((1,), 0.1))
                print(f"task = {self.task}, 初始化gate_scale为nn.Parameter(torch.full((1,), 0.1))")

        # ── 3. 列类型嵌入 ────────────────────────────────────────────────
        self.col_type_emb = ColumnTypeEmbedding(dim_hidden)

        # ── 4. Cross-Attention（并行双向）───────────────────────────────
        # cross_attn 前保存 num_seq 快照，reverse_cross_attn 用该快照作为 KV,
        # 确保 cat ← num 方向读到的是真实数值信号，而非 num 已经融合过 cat 之后的折射结果。
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cross_attn         = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
            if not self.few_num_mode:
                self.reverse_cross_attn = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
            else:
                self.reverse_cross_attn = None
        else:
            self.cross_attn         = None
            self.reverse_cross_attn = None

        # ── 4b. few_num_mode 补充：NumToCatFiLM ─────────────────────────
        if self.few_num_mode and len(self.num_classes) > 0:
            self.num_to_cat_film = NumToCatFiLM(dim_hidden)
        else:
            self.num_to_cat_film = None

        # ── 5. 列间 Transformer（模态内自注意力）────────────────────────
        if self._eff_num_tokens > 1:
            self.num_transformer = nn.TransformerEncoderLayer(
                d_model=dim_hidden, nhead=nhead_col,
                dim_feedforward=dim_hidden * 2, dropout=dropout,
                batch_first=True, norm_first=True,
            )
        else:
            self.num_transformer = None

        if len(self.num_classes) > 1:
            self.cat_transformer = nn.TransformerEncoderLayer(
                d_model=dim_hidden, nhead=nhead_col,
                dim_feedforward=dim_hidden * 2, dropout=dropout,
                batch_first=True, norm_first=True,
            )
        else:
            self.cat_transformer = None

        # ── 6. 双路 CLS 聚合 ─────────────────────────────────────────────
        # cls_num 专门聚合 num_seq，cls_cat 专门聚合 cat_seq，拼接后经
        # proj_fuse 得到 h_shared —— 避免 N_cat >> N_num 时数值信息被稀释。
        self.cls_num = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.cls_cat = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.pool_num_attn  = nn.MultiheadAttention(dim_hidden, nhead_col, dropout=dropout, batch_first=True)
        self.pool_cat_attn  = nn.MultiheadAttention(dim_hidden, nhead_col, dropout=dropout, batch_first=True)
        self.pool_norm_num_q  = nn.LayerNorm(dim_hidden)
        self.pool_norm_num_kv = nn.LayerNorm(dim_hidden)
        self.pool_norm_cat_q  = nn.LayerNorm(dim_hidden)
        self.pool_norm_cat_kv = nn.LayerNorm(dim_hidden)

        # ── 7. 条件编码 ──────────────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # Cross-Attention 时间步门控生成器（只在同时有数值和类别列时才需要）
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cross_attn_gate = nn.Linear(dim_t, 1)
            nn.init.constant_(self.cross_attn_gate.bias, -2.0) # 初始sigmoid约为0.12， 温和开启
        else:
            self.cross_attn_gate = None

        # ── 8. 聚合投影（dim_hidden*2 → dim_hidden，适配双路 CLS 拼接）─
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden),
        )

        # ── 9. 双专属主干 ────────────────────────────────────────────────
        if self.num_numerical > 0:
            self.num_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.num_backbone = None

        if len(self.num_classes) > 0:
            self.cat_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.cat_backbone = None

        # ── 10. （可选共享）解码器 + 输出头 ──────────────────────────────
        # share_decoder=True ：与 v1 对齐，num/cat 各一个共享展开层 + 轻量列头
        # share_decoder=False：每列独立 Sequential，原始版本
        if self.num_numerical > 0:
            if share_decoder:
                self.num_shared_dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(dim_hidden),
                )
                self.out_num = nn.Linear(dim_hidden, self.num_numerical)
            else:
                self.num_shared_dec = None
                self.out_num = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Linear(dim_hidden, self.num_numerical),
                )
        else:
            self.num_shared_dec = None
            self.out_num = None

        if len(self.num_classes) > 0:
            if share_decoder:
                self.cat_shared_dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(dim_hidden),
                )
                self.out_cat = nn.ModuleList([
                    nn.Linear(dim_hidden, c) for c in self.num_classes
                ])
            else:
                self.cat_shared_dec = None
                self.out_cat = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(dim_hidden, dim_hidden),
                        nn.SiLU(),
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
        """
        先验初始化：输出头偏置 = log(prior)，权重乘 0.1。

        share_decoder=True  → out_cat[i] 是 nn.Linear，直接访问属性
        share_decoder=False → out_cat[i] 是 nn.Sequential，取最后一层 [-1]
        """
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                prior     = true_marginals[i]
                log_prior = torch.log(prior + 1e-8)
                target = head if self.share_decoder else head[-1]
                target.bias.copy_(log_prior)
                target.weight.data.mul_(0.1)

    # ────────────────────────────────────────────────────────────────────
    def _encode_num_tokens(self, x_num: torch.Tensor) -> list:
        """
        数值列编码 → Token 列表（每项 [B, D]）。
        正常模式  : 每列 → 1 个 Token（PeriodicEmbedding）
        few_num_mode: 每列 → n_expand 个 Token（MultiScaleNumEncoder）
        """
        num_tokens = []
        if self.num_col_encoders is None:
            return num_tokens

        for i, enc in enumerate(self.num_col_encoders):
            if self.few_num_mode:
                expanded = enc(x_num[:, i])                    # [B, n_expand, D]
                for j in range(expanded.shape[1]):
                    num_tokens.append(expanded[:, j])          # [B, D]
            else:
                num_tokens.append(enc(x_num[:, i]))

        return num_tokens

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x_in: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor):
        B = x_in.shape[0]

        # ── 条件向量 ─────────────────────────────────────────────────────
        t_emb     = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_raw_emb = self.label_emb(y.view(-1).long())
        emb       = t_emb + y_raw_emb                         # [B, dim_t]

        # 计算时间步门控(如果启用)
        if self.use_time_gate and self.cross_attn_gate is not None:
            lambda_t = torch.sigmoid(self.cross_attn_gate(t_emb)).unsqueeze(-1)
        else:
            lambda_t = None

        # ── 拆分输入 ─────────────────────────────────────────────────────
        x_num      = x_in[:, :self.num_numerical]             # [B, N_num]
        x_cat_prob = x_in[:, self.num_numerical:].long()

        # ── 逐列 Token 化 ────────────────────────────────────────────────
        num_tokens = self._encode_num_tokens(x_num)

        cat_tokens, start = [], 0
        if self.cat_embed_layers is not None:
            for emb_layer, proj, n_class in zip(
                self.cat_embed_layers, self.cat_col_projs, self.num_classes
            ):
                end  = start + n_class
                prob = x_cat_prob[:, start:end]
                tok  = self.cat_dropout(proj(prob.float() @ emb_layer.weight))
                cat_tokens.append(tok)
                start = end

        # ── CTE：注入列类型标识向量 ──────────────────────────────────────
        if num_tokens:
            num_type_vec = self.col_type_emb.get(0, x_in.device)  # [D]
            num_tokens = [tok + num_type_vec for tok in num_tokens]
        if cat_tokens:
            cat_type_vec = self.col_type_emb.get(1, x_in.device)  # [D]
            cat_tokens = [tok + cat_type_vec for tok in cat_tokens]

        # ── 阶段0：Token 化后 FiLM（Intra-Attn 之前）────────────────────
        if self.cross_attn is not None and num_tokens and cat_tokens:
            h_cat_mean = torch.stack(cat_tokens, dim=1).mean(dim=1)  # [B, D]
            gamma_beta = self.cat_to_num(h_cat_mean)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gate        = torch.sigmoid(self.gate_layer(h_cat_mean))
            scale_coef  = torch.tanh(self.gate_scale)

            num_tokens = [
                tok + scale_coef * gate * (tok * gamma + beta)
                for tok in num_tokens
            ]

        # ── 阶段1：模态内自注意力 ─────────────────────────────────────────
        num_seq = torch.stack(num_tokens, dim=1) if num_tokens else \
                  torch.empty(B, 0, self.cls_num.shape[-1], device=x_in.device)
        cat_seq = torch.stack(cat_tokens, dim=1) if cat_tokens else \
                  torch.empty(B, 0, self.cls_num.shape[-1], device=x_in.device)

        if self.num_transformer is not None and num_seq.shape[1] > 0:
            num_seq = self.num_transformer(num_seq)
        if self.cat_transformer is not None and cat_seq.shape[1] > 0:
            cat_seq = self.cat_transformer(cat_seq)

        # ── 阶段2：FiLM + 并行双向 Cross-Attention ───────────────────────
        # 并行修复：cross_attn 前保存 num_seq 快照（num_seq_pre），
        # reverse_cross_attn 用该快照作为 KV，确保两个方向均读取对方
        # cross-attn 之前的序列，消除 cat 信息折射。
        if self.cross_attn is not None and num_seq.shape[1] > 0 and cat_seq.shape[1] > 0:
            h_cat_mean = cat_seq.mean(dim=1)
            gamma_beta = self.cat_to_num(h_cat_mean)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gate        = torch.sigmoid(self.gate_layer(h_cat_mean))
            scale_coef  = torch.tanh(self.gate_scale)

            num_seq = num_seq + scale_coef * gate.unsqueeze(1) * (
                num_seq * gamma.unsqueeze(1) + beta.unsqueeze(1)
            )

            num_seq_pre = num_seq   # ← 保存快照：cross_attn 之前的 num_seq
            num_seq = self.cross_attn(num_seq, cat_seq, gate=lambda_t)               # num ← cat（原始）
            if not self.few_num_mode:
                cat_seq = self.reverse_cross_attn(cat_seq, num_seq_pre, gate=lambda_t)   # cat ← 原始 num！

        # ── 阶段2b：NumToCatFiLM（仅 few_num_mode）─────────────────────
        if self.num_to_cat_film is not None and num_seq.shape[1] > 0:
            cat_seq = self.num_to_cat_film(cat_seq, num_seq)

        # ── 阶段3：双路 CLS 聚合 ─────────────────────────────────────────
        if num_seq.shape[1] > 0:
            q_num = self.pool_norm_num_q(self.cls_num.expand(B, -1, -1))   # [B,1,D]
            kv_num = self.pool_norm_num_kv(num_seq)                         # [B,L_num,D]
            h_num_pool, _ = self.pool_num_attn(q_num, kv_num, kv_num)
            h_num_pool = (self.cls_num.expand(B, -1, -1) + h_num_pool).squeeze(1)  # [B,D]
        else:
            h_num_pool = torch.zeros(B, self.cls_num.shape[-1], device=x_in.device)

        if cat_seq.shape[1] > 0:
            q_cat = self.pool_norm_cat_q(self.cls_cat.expand(B, -1, -1))   # [B,1,D]
            kv_cat = self.pool_norm_cat_kv(cat_seq)                         # [B,L_cat,D]
            h_cat_pool, _ = self.pool_cat_attn(q_cat, kv_cat, kv_cat)
            h_cat_pool = (self.cls_cat.expand(B, -1, -1) + h_cat_pool).squeeze(1)  # [B,D]
        else:
            h_cat_pool = torch.zeros(B, self.cls_cat.shape[-1], device=x_in.device)

        h_shared = self.proj_fuse(torch.cat([h_num_pool, h_cat_pool], dim=-1))  # [B,D]

        # ── 双专属主干 → （可选共享解码器）→ 输出头 ──────────────────────
        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h_num = h_shared.clone()
            for layer in self.num_backbone:
                h_num = layer(h_num, emb)
            if self.num_shared_dec is not None:
                h_num = self.num_shared_dec(h_num)
            out_n = self.out_num(h_num)

        out_c_list = []
        if self.cat_backbone is not None:
            h_cat = h_shared.clone()
            for layer in self.cat_backbone:
                h_cat = layer(h_cat, emb)
            if self.cat_shared_dec is not None:
                h_cat = self.cat_shared_dec(h_cat)
            out_c_list = [head(h_cat) for head in self.out_cat]

        out_c = (
            torch.cat(out_c_list, dim=-1)
            if out_c_list
            else torch.empty((B, 0), device=x_in.device)
        )

        return torch.cat([out_n, out_c], dim=-1)