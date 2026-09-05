# core/denoise_model_v1.py  (v1 — Unified FT-Transformer, 隐式跨模态路由)
from __future__ import annotations
import math
import torch
import torch.nn as nn

from core.denoise_utils import timestep_embedding, PeriodicEmbedding, MLPResidualBlock, ColumnTypeEmbedding


class TabMLPDenoise(nn.Module):
    def __init__(
        self,
        task,
        info,
        num_classes_cond:    int   = 2,
        dim_t:                int   = 128,
        dim_hidden:           int   = 256,
        num_layers:           int   = 3,    # Unified Transformer 层数
        num_backbone_layers:  int   = 2,    # MLP backbone 层数
        dropout:              float = 0.1,
        n_frequencies:        int   = 16,
        nhead:                int   = 4,
        share_decoder:        bool  = True,  # 新增：解码器是否共享展开层
    ):
        super().__init__()

        self.task          = task
        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']       # list[int]
        true_marginals     = info.get('true_marginals', None)
        self.share_decoder = share_decoder

        N_num = self.num_numerical
        N_cat = len(self.num_classes)

        # ── cat_loss.py 兼容接口 ─────────────
        self.num_numerical_features = N_num

        _start = 0
        self.slices_for_classes = []
        for n_c in self.num_classes:
            self.slices_for_classes.append(slice(_start, _start + n_c))
            _start += n_c

        self.true_marginals = true_marginals   # list[Tensor] | None

        # ── 1. Column Tokenizer ───────────────────────
        if N_num > 0:
            self.num_col_encoders = nn.ModuleList([
                PeriodicEmbedding(dim_hidden, n_frequencies)
                for _ in range(N_num)
            ])
        else:
            self.num_col_encoders = None

        # 类别列：soft embedding（prob @ W_emb）→ 线性投影 → dim_hidden
        # Dropout 作用于 soft embedding（列级 feature dropout，增强鲁棒性）
        self.cat_emb_dropout = nn.Dropout(0.1)
        if N_cat > 0:
            self.cat_embed_layers = nn.ModuleList([
                nn.Embedding(c, min(16, c * 4)) for c in self.num_classes
            ])
            self.cat_col_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(min(16, c * 4), dim_hidden),
                    nn.SiLU(),
                    nn.LayerNorm(dim_hidden),
                ) for c in self.num_classes
            ])
        else:
            self.cat_embed_layers = None
            self.cat_col_projs    = None

        # ── 2. Column Type Embedding ─────────────────────────────────────
        self.col_type_emb = ColumnTypeEmbedding(dim_hidden)

        # ── 3. [CLS] Token ───────────────────────────────────────────────
        # 全局聚合 token，经过 Transformer 后代表整个行的表示。
        # 不偏向任何一侧：N_cat=20 和 N_num=1 时，[CLS] 的注意力由
        # Transformer 自适应学习，不被 token 数量比例机械决定。
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)

        # ── 4. Unified Transformer ────────────────────────────────────────
        # 所有列（数值 + 类别）的 token 一起处理。
        # Pre-norm (norm_first=True)：梯度更稳定，在小数据集上训练更可靠。
        # dim_feedforward = dim_hidden * 4：FT-Transformer 标准配置。
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = dim_hidden,
            nhead           = nhead,
            dim_feedforward = dim_hidden * 4,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.unified_transformer = nn.TransformerEncoder(
            encoder_layer        = encoder_layer,
            num_layers           = num_layers,
            norm                 = nn.LayerNorm(dim_hidden),
            enable_nested_tensor = False,
        )

        # ── 5. 条件编码（时间步 + 标签）─────────────────────────────────
        # MLP backbone 通过 AdaLN 接收条件向量，而不是注入 Transformer。
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ── 6. 双专属 MLP 主干 + （可选）共享解码器 + 输出头 ─────────────
        #
        # share_decoder=True：
        #   shared_dec  : Linear(D,D) → SiLU → Dropout → LayerNorm（num/cat 各一份）
        #   out_num     : Linear(D, N_num)   ← 轻量，单个头
        #   out_cat[i]  : Linear(D, c_i)     ← 轻量，各列独立但仅 D×c 参数
        #   （hepatitis, D=512 实测：3,430K → 279K，减少 12.3 倍，
        #    直接缓解类别列过拟合 / DCR 下降）
        #
        # share_decoder=False：
        #   每列独立 Sequential(Linear(D,D), SiLU, Linear(D,c))，原始版本。
        if N_num > 0:
            self.num_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_backbone_layers)
            ])
            if share_decoder:
                self.num_shared_dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(dim_hidden),
                )
                self.out_num = nn.Linear(dim_hidden, N_num)
            else:
                self.num_shared_dec = None
                self.out_num = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Linear(dim_hidden, N_num),
                )
        else:
            self.num_backbone   = None
            self.num_shared_dec = None
            self.out_num        = None

        if N_cat > 0:
            self.cat_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_backbone_layers)
            ])
            if share_decoder:
                self.cat_shared_dec = nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.LayerNorm(dim_hidden),
                )
                # 各列只保留 Linear(D, c)：真正产生列间差异的部分
                self.out_cat = nn.ModuleList([
                    nn.Linear(dim_hidden, c)
                    for c in self.num_classes
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
            self.cat_backbone   = None
            self.cat_shared_dec = None
            self.out_cat        = None

        if true_marginals is not None and self.out_cat is not None:
            self._apply_prior_initialization(true_marginals)

    # ─────────────────────────────────────────────────────────────────────
    def _apply_prior_initialization(self, true_marginals):
        """
        输出头偏置初始化为 log(prior)，使训练初期类别预测接近先验分布。
        权重乘以 0.1 以减小初始输出幅度，配合 zero-init backbone 使用。

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

    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_in:      torch.Tensor,   # [B, N_num + sum(num_classes)]
        timesteps: torch.Tensor,   # [B]
        y:         torch.Tensor,   # [B] — disease label or task label
    ) -> torch.Tensor:             # [B, N_num + sum(num_classes)]

        B = x_in.shape[0]

        # ── 条件向量 ─────────────────────────────────────────────────────
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        emb   = t_emb + self.label_emb(y.view(-1).long())   # [B, dim_t]

        # ── 拆分输入 ─────────────────────────────────────────────────────
        x_num     = x_in[:, :self.num_numerical]              # [B, N_num]
        x_cat_raw = x_in[:, self.num_numerical:].long()       # [B, sum(n_class)]

        # ── Column Tokenization ──────────────────────────────────────────
        # 两种列均映射到 dim_hidden，之后进入同一 Transformer。
        tokens = []

        # 数值列：PeriodicEmbedding + 数值类型标识
        if self.num_col_encoders is not None:
            num_type = self.col_type_emb.get(0, x_in.device)   # [D]
            for i, enc in enumerate(self.num_col_encoders):
                tokens.append(enc(x_num[:, i]) + num_type)      # [B, D]

        # 类别列：soft embedding + 线性投影 + 类别类型标识
        if self.cat_embed_layers is not None:
            cat_type = self.col_type_emb.get(1, x_in.device)   # [D]
            start = 0
            for emb_layer, proj, n_c in zip(
                self.cat_embed_layers, self.cat_col_projs, self.num_classes
            ):
                prob = x_cat_raw[:, start : start + n_c]        # [B, n_c]
                raw  = self.cat_emb_dropout(
                    prob.float() @ emb_layer.weight              # [B, emb_dim]
                )
                tokens.append(proj(raw) + cat_type)             # [B, D]
                start += n_c

        # ── 构建 Transformer 输入序列 ─────────────────────────────────────
        # [CLS] token 位于序列首位，统一聚合所有列的信息。
        token_seq = torch.stack(tokens, dim=1)                   # [B, N_cols, D]
        cls_exp   = self.cls_token.expand(B, -1, -1)            # [B, 1,      D]
        seq       = torch.cat([cls_exp, token_seq], dim=1)      # [B, 1+N_cols, D]

        # ── Unified Transformer ───────────────────────────────────────────
        # 所有列（数值 + 类别）相互 attend，无类型约束。
        # Transformer 自主学习：哪些列相关、跨模态信号如何流动。
        h_seq    = self.unified_transformer(seq)                 # [B, 1+N_cols, D]
        h_shared = h_seq[:, 0]                                   # [B, D]

        # ── 双专属 MLP 主干 → （可选共享解码器）→ 输出头 ─────────────────
        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h = h_shared.clone()
            for layer in self.num_backbone:
                h = layer(h, emb)
            if self.num_shared_dec is not None:
                h = self.num_shared_dec(h)          # 共享展开 + Dropout
            out_n = self.out_num(h)

        out_c_list = []
        if self.cat_backbone is not None:
            h = h_shared.clone()
            for layer in self.cat_backbone:
                h = layer(h, emb)
            if self.cat_shared_dec is not None:
                h = self.cat_shared_dec(h)          # 共享展开 + Dropout，各列公用
            out_c_list = [head(h) for head in self.out_cat]

        out_c = (torch.cat(out_c_list, dim=-1)
                 if out_c_list
                 else torch.empty((B, 0), device=x_in.device))

        return torch.cat([out_n, out_c], dim=-1)