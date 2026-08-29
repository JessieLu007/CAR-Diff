# core/denoise_model.py  (v4 — Unified FT-Transformer)
"""
TabMLPDenoise v4：统一 Transformer 架构
=========================================

设计动机
--------
v3 对 cross-attention / FiLM / dual-CLS 的精细调制在 Framingham 数据集上
过度特化，迁移到 Thyroid（N_num=1）、PIMA（label in cat）等数据集时
泛化能力不足。

根本原因：手工设计的跨模态路由（cat→num FiLM、双向 cross-attn）是
对特定 N_num/N_cat 比例的硬编码假设。当比例变化时这些假设失效。

借鉴 FT-Transformer（Gorishniy et al. 2021）的核心思想：
    异构性在 Tokenization 层解决，之后统一处理。
    每一列（无论数值/类别）→ 统一维度 Token → 同一个 Transformer。
    Transformer 本身学习哪些列应互相影响，无需手工设计路由。

架构
----
    x_in
     │
     ├─ num_col_encoders (PeriodicEmbedding × N_num)
     │   └─ + col_type_emb(0)          # 数值类型标识
     │
     └─ cat_embed_layers + cat_col_projs (× N_cat)
         └─ + col_type_emb(1)          # 类别类型标识
     │
     ▼
    [CLS] ‖ num_tokens ‖ cat_tokens    → [B, 1 + N_num + N_cat, D]
     │
     ▼
    unified_transformer (pre-norm TransformerEncoder, L 层)
    ── 所有列在同一 Transformer 中相互 attend ──
     │
     ▼
    h_shared = h_seq[:, 0]             # [CLS] 无偏全局聚合 → [B, D]
     │
     ├─ num_backbone (AdaLN MLPResidualBlock × M)  ← emb(t, y)
     │   └─ out_num
     │
     └─ cat_backbone (AdaLN MLPResidualBlock × M)  ← emb(t, y)
         └─ out_cat_heads

外部接口（与 cat_loss.py / num_loss.py / 训练循环 完全兼容）
------------------------------------------------------------
    model.num_numerical_features  —— cat_loss.py 使用
    model.slices_for_classes      —— cat_loss.py 使用
    model.true_marginals          —— cat_loss.py 使用
    model.model_cfg               —— 由训练循环外部赋值，不变
    forward(x_in, timesteps, y)   —— 签名不变

超参数推荐
----------
    小数据集（N < 500，如 Thyroid）：
        dim_hidden=128, num_attn_layers=2, num_backbone_layers=2, dropout=0.2

    中等数据集（500 < N < 5000，如 Framingham、PIMA）：
        dim_hidden=256, num_attn_layers=3, num_backbone_layers=2, dropout=0.1

    大数据集（N > 5000）：
        dim_hidden=256, num_attn_layers=4, num_backbone_layers=3, dropout=0.1
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数（与历史版本完全相同，保持兼容）
# ─────────────────────────────────────────────────────────────────────────────

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicEmbedding(nn.Module):
    """
    单列数值 → [B, embed_dim]。
    可学习频率，sin/cos 展开后线性投影。
    """
    def __init__(self, embed_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_frequencies) * 0.02)
        self.proj  = nn.Linear(n_frequencies * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f   = x.float().unsqueeze(-1) * self.freqs * 2 * math.pi
        x_enc = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
        return self.proj(x_enc)


class MLPResidualBlock(nn.Module):
    """
    AdaLN 风格残差块。
    接收时间步 + 标签条件向量 emb，通过 scale/shift 调制特征。
    cond_proj zero-init：训练初期条件影响为零，保证收敛稳定。
    内部升维 dim → dim*2 → dim，增加表达能力（对比 v3 的 dim → dim）。
    """
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.ff        = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.cond_proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h            = self.norm(x)
        scale, shift = self.cond_proj(emb).chunk(2, dim=-1)
        h            = h * (1 + scale) + shift
        return x + self.ff(h)


class ColumnTypeEmbedding(nn.Module):
    """
    列类型标识向量（0=数值列, 1=类别列）。
    small init (std=0.02) 保证训练初期不干扰其他嵌入。
    让 Transformer attention head 可以学习类型感知的路由策略。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def get(self, type_id: int, device: torch.device) -> torch.Tensor:
        """返回 [D] 类型向量，广播叠加到任意形状 Token 上。"""
        return self.emb(torch.tensor(type_id, dtype=torch.long, device=device))


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class TabMLPDenoise(nn.Module):

    def __init__(
        self,
        task,
        info,
        num_classes_cond:     int   = 2,
        dim_t:                int   = 128,
        dim_hidden:           int   = 256,
        num_layers:      int   = 3,    # Unified Transformer 层数
        num_backbone_layers:  int   = 2,    # MLP backbone 层数
        dropout:              float = 0.1,
        n_frequencies:        int   = 16,
        nhead:                int   = 4,
    ):
        super().__init__()

        self.task          = task
        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']       # list[int]
        true_marginals     = info.get('true_marginals', None)

        N_num = self.num_numerical
        N_cat = len(self.num_classes)

        # ── cat_loss.py 兼容接口 ─────────────────────────────────────────
        # cat_loss.py 通过这三个属性读取模型信息，必须保持。
        self.num_numerical_features = N_num

        _start = 0
        self.slices_for_classes = []
        for n_c in self.num_classes:
            self.slices_for_classes.append(slice(_start, _start + n_c))
            _start += n_c

        self.true_marginals = true_marginals   # list[Tensor] | None
        # model_cfg 由外部训练循环赋值（model.model_cfg = cfg），此处不初始化

        # ── 1. Column Tokenizer ──────────────────────────────────────────
        # 数值列：每列独立 PeriodicEmbedding → dim_hidden
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
        # 为 Token 化完成后的每个 Token 注入类型标识（在 Transformer 之前）。
        # 对 Thyroid / PIMA 等极端 N_num/N_cat 比例的数据集尤其重要：
        # attention head 可以学习"我是 age token，应重点关注哪些 cat token"。
        self.col_type_emb = ColumnTypeEmbedding(dim_hidden)

        # ── 3. [CLS] Token ───────────────────────────────────────────────
        # 全局聚合 token，经过 Transformer 后代表整个行的表示。
        # 不偏向任何一侧：N_cat=20 和 N_num=1 时，[CLS] 的注意力由
        # Transformer 自适应学习，不被 token 数量比例机械决定。
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)

        # ── 4. Unified Transformer ────────────────────────────────────────
        # 所有列（数值 + 类别）的 token 一起处理。
        # 无需手工设计 cross-attn / FiLM：attention 本身学习跨模态路由。
        #
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
            encoder_layer = encoder_layer,
            num_layers    = num_layers,
            norm               = nn.LayerNorm(dim_hidden),
            enable_nested_tensor = False,
        )

        # ── 5. 条件编码（时间步 + 标签）─────────────────────────────────
        # MLP backbone 通过 AdaLN 接收条件向量，而不是注入 Transformer。
        # 好处：Transformer 专注于特征交互（与时间无关），backbone 专注条件化。
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ── 6. 双专属 MLP 主干 + 共享解码器 + 轻量输出头 ─────────────────
        #
        # 原设计缺陷：
        #   out_cat 为每列独立 Sequential(Linear(D,D), SiLU, Linear(D,c))。
        #   所有列头共享同一个 h_cat 输入，但各自持有 D×D 参数，等价于
        #   N_cat 份功能重复的展开层——hepatitis 13 列 dim=512 时
        #   约 3.4M 参数用于此处，而 out_num 只有 0.27M，相差约 13 倍，
        #   是过拟合和 DCR 下降的直接原因。
        #
        # 修复：分离"共享展开"与"列专属投影"
        #   shared_dec  : Linear(D,D) → SiLU → Dropout → LayerNorm
        #                 一个模块，num 侧和 cat 侧各一份
        #   out_num     : Linear(D, N_num)   ← 轻量，单个头
        #   out_cat[i]  : Linear(D, c_i)     ← 轻量，各列独立但仅 D×c 参数
        #
        # 效果（hepatitis, D=512）：
        #   旧 cat 解码层：3,430K → 新：264K + 15K = 279K（减少 12.3 倍）
        #   num 解码层：265K（与新 cat 解码层对等，消除不平衡）
        #   Dropout 在共享层施加正则，直接缓解 DCR 下降的过拟合风险
        if N_num > 0:
            self.num_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_backbone_layers)
            ])
            self.num_shared_dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(dim_hidden),
            )
            self.out_num = nn.Linear(dim_hidden, N_num)
        else:
            self.num_backbone   = None
            self.num_shared_dec = None
            self.out_num        = None

        if N_cat > 0:
            self.cat_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_backbone_layers)
            ])
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
        """
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                prior = true_marginals[i]
                # out_cat[i] 现为 nn.Linear（非 Sequential），直接访问属性
                head.bias.copy_(torch.log(prior + 1e-8))
                head.weight.data.mul_(0.1)

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

        # ── 双专属 MLP 主干 → 共享解码器 → 轻量输出头 ───────────────────
        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h = h_shared.clone()
            for layer in self.num_backbone:
                h = layer(h, emb)
            h     = self.num_shared_dec(h)      # 共享展开 + Dropout
            out_n = self.out_num(h)             # Linear(D, N_num)

        out_c_list = []
        if self.cat_backbone is not None:
            h = h_shared.clone()
            for layer in self.cat_backbone:
                h = layer(h, emb)
            h         = self.cat_shared_dec(h)  # 共享展开 + Dropout，各列公用
            out_c_list = [head(h) for head in self.out_cat]  # Linear(D, c_i) × N_cat

        out_c = (torch.cat(out_c_list, dim=-1)
                 if out_c_list
                 else torch.empty((B, 0), device=x_in.device))

        return torch.cat([out_n, out_c], dim=-1)

# core/denoise_model.py  (v4 — Unified FT-Transformer)
"""
TabMLPDenoise v4：统一 Transformer 架构
=========================================

设计动机
--------
v3 对 cross-attention / FiLM / dual-CLS 的精细调制在 Framingham 数据集上
过度特化，迁移到 Thyroid（N_num=1）、PIMA（label in cat）等数据集时
泛化能力不足。

根本原因：手工设计的跨模态路由（cat→num FiLM、双向 cross-attn）是
对特定 N_num/N_cat 比例的硬编码假设。当比例变化时这些假设失效。

借鉴 FT-Transformer（Gorishniy et al. 2021）的核心思想：
    异构性在 Tokenization 层解决，之后统一处理。
    每一列（无论数值/类别）→ 统一维度 Token → 同一个 Transformer。
    Transformer 本身学习哪些列应互相影响，无需手工设计路由。

架构
----
    x_in
     │
     ├─ num_col_encoders (PeriodicEmbedding × N_num)
     │   └─ + col_type_emb(0)          # 数值类型标识
     │
     └─ cat_embed_layers + cat_col_projs (× N_cat)
         └─ + col_type_emb(1)          # 类别类型标识
     │
     ▼
    [CLS] ‖ num_tokens ‖ cat_tokens    → [B, 1 + N_num + N_cat, D]
     │
     ▼
    unified_transformer (pre-norm TransformerEncoder, L 层)
    ── 所有列在同一 Transformer 中相互 attend ──
     │
     ▼
    h_shared = h_seq[:, 0]             # [CLS] 无偏全局聚合 → [B, D]
     │
     ├─ num_backbone (AdaLN MLPResidualBlock × M)  ← emb(t, y)
     │   └─ out_num
     │
     └─ cat_backbone (AdaLN MLPResidualBlock × M)  ← emb(t, y)
         └─ out_cat_heads

外部接口（与 cat_loss.py / num_loss.py / 训练循环 完全兼容）
------------------------------------------------------------
    model.num_numerical_features  —— cat_loss.py 使用
    model.slices_for_classes      —— cat_loss.py 使用
    model.true_marginals          —— cat_loss.py 使用
    model.model_cfg               —— 由训练循环外部赋值，不变
    forward(x_in, timesteps, y)   —— 签名不变

超参数推荐
----------
    小数据集（N < 500，如 Thyroid）：
        dim_hidden=128, num_layers=2, num_backbone_layers=2, dropout=0.2

    中等数据集（500 < N < 5000，如 Framingham、PIMA）：
        dim_hidden=256, num_layers=3, num_backbone_layers=2, dropout=0.1

    大数据集（N > 5000）：
        dim_hidden=256, num_layers=4, num_backbone_layers=3, dropout=0.1
"""

'''from __future__ import annotations
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数（与历史版本完全相同，保持兼容）
# ─────────────────────────────────────────────────────────────────────────────

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# 基础模块
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicEmbedding(nn.Module):
    """
    单列数值 → [B, embed_dim]。
    可学习频率，sin/cos 展开后线性投影。
    """
    def __init__(self, embed_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_frequencies) * 0.02)
        self.proj  = nn.Linear(n_frequencies * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f   = x.float().unsqueeze(-1) * self.freqs * 2 * math.pi
        x_enc = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
        return self.proj(x_enc)


class MLPResidualBlock(nn.Module):
    """
    AdaLN 风格残差块。
    接收时间步 + 标签条件向量 emb，通过 scale/shift 调制特征。
    cond_proj zero-init：训练初期条件影响为零，保证收敛稳定。
    内部升维 dim → dim*2 → dim，增加表达能力（对比 v3 的 dim → dim）。
    """
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.ff        = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.cond_proj = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h            = self.norm(x)
        scale, shift = self.cond_proj(emb).chunk(2, dim=-1)
        h            = h * (1 + scale) + shift
        return x + self.ff(h)


class ColumnTypeEmbedding(nn.Module):
    """
    列类型标识向量（0=数值列, 1=类别列）。
    small init (std=0.02) 保证训练初期不干扰其他嵌入。
    让 Transformer attention head 可以学习类型感知的路由策略。
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.emb.weight, std=0.02)

    def get(self, type_id: int, device: torch.device) -> torch.Tensor:
        """返回 [D] 类型向量，广播叠加到任意形状 Token 上。"""
        return self.emb(torch.tensor(type_id, dtype=torch.long, device=device))


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class TabMLPDenoise(nn.Module):

    def __init__(
        self,
        task,
        info,
        num_classes_cond:     int   = 2,
        dim_t:                int   = 128,
        dim_hidden:           int   = 256,
        num_layers:      int   = 3,    # Unified Transformer 层数
        num_backbone_layers:  int   = 2,    # MLP backbone 层数
        dropout:              float = 0.1,
        n_frequencies:        int   = 16,
        nhead:                int   = 4,
    ):
        super().__init__()

        self.task          = task
        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']       # list[int]
        true_marginals     = info.get('true_marginals', None)

        N_num = self.num_numerical
        N_cat = len(self.num_classes)

        # ── cat_loss.py 兼容接口 ─────────────────────────────────────────
        # cat_loss.py 通过这三个属性读取模型信息，必须保持。
        self.num_numerical_features = N_num

        _start = 0
        self.slices_for_classes = []
        for n_c in self.num_classes:
            self.slices_for_classes.append(slice(_start, _start + n_c))
            _start += n_c

        self.true_marginals = true_marginals   # list[Tensor] | None
        # model_cfg 由外部训练循环赋值（model.model_cfg = cfg），此处不初始化

        # ── 1. Column Tokenizer ──────────────────────────────────────────
        # 数值列：每列独立 PeriodicEmbedding → dim_hidden
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
        # 为 Token 化完成后的每个 Token 注入类型标识（在 Transformer 之前）。
        # 对 Thyroid / PIMA 等极端 N_num/N_cat 比例的数据集尤其重要：
        # attention head 可以学习"我是 age token，应重点关注哪些 cat token"。
        self.col_type_emb = ColumnTypeEmbedding(dim_hidden)

        # ── 3. [CLS] Token ───────────────────────────────────────────────
        # 全局聚合 token，经过 Transformer 后代表整个行的表示。
        # 不偏向任何一侧：N_cat=20 和 N_num=1 时，[CLS] 的注意力由
        # Transformer 自适应学习，不被 token 数量比例机械决定。
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)

        # ── 4. Unified Transformer ────────────────────────────────────────
        # 所有列（数值 + 类别）的 token 一起处理。
        # 无需手工设计 cross-attn / FiLM：attention 本身学习跨模态路由。
        #
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
            encoder_layer = encoder_layer,
            num_layers    = num_layers,
            norm               = nn.LayerNorm(dim_hidden),
            enable_nested_tensor = False,
        )

        # ── 5. 条件编码（时间步 + 标签）─────────────────────────────────
        # MLP backbone 通过 AdaLN 接收条件向量，而不是注入 Transformer。
        # 好处：Transformer 专注于特征交互（与时间无关），backbone 专注条件化。
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2),
            nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ── 6. 双专属 MLP 主干 ───────────────────────────────────────────
        # 数值和类别输出使用独立的 backbone，保留各自的归纳偏置：
        # 数值列：连续预测（MSE/eps）；类别列：logit 预测（CE/focal）。
        if N_num > 0:
            self.num_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_backbone_layers)
            ])
            self.out_num = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, N_num),
            )
        else:
            self.num_backbone = None
            self.out_num      = None

        if N_cat > 0:
            self.cat_backbone = nn.ModuleList([
                MLPResidualBlock(dim_hidden, dim_t, dropout)
                for _ in range(num_backbone_layers)
            ])
            self.out_cat = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_hidden, dim_hidden),
                    nn.SiLU(),
                    nn.Linear(dim_hidden, c),
                ) for c in self.num_classes
            ])
        else:
            self.cat_backbone = None
            self.out_cat      = None

        if true_marginals is not None and self.out_cat is not None:
            self._apply_prior_initialization(true_marginals)

    # ─────────────────────────────────────────────────────────────────────
    def _apply_prior_initialization(self, true_marginals):
        """
        输出头偏置初始化为 log(prior)，使训练初期类别预测接近先验分布。
        权重乘以 0.1 以减小初始输出幅度，配合 zero-init backbone 使用。
        """
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                prior = true_marginals[i]
                head[-1].bias.copy_(torch.log(prior + 1e-8))
                head[-1].weight.data.mul_(0.1)

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

        # ── 双专属 MLP 主干 ───────────────────────────────────────────────
        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h = h_shared.clone()
            for layer in self.num_backbone:
                h = layer(h, emb)
            out_n = self.out_num(h)

        out_c_list = []
        if self.cat_backbone is not None:
            h = h_shared.clone()
            for layer in self.cat_backbone:
                h = layer(h, emb)
            out_c_list = [head(h) for head in self.out_cat]

        out_c = (torch.cat(out_c_list, dim=-1)
                 if out_c_list
                 else torch.empty((B, 0), device=x_in.device))

        return torch.cat([out_n, out_c], dim=-1)'''

'''# core/denoise_model.py
import torch
import torch.nn as nn
import math

from .denoise_utils import RFFEmbedding
    

# ─────────────────────────────────────────────
# 工具函数（不变）
# ─────────────────────────────────────────────
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ─────────────────────────────────────────────
# 编码模块
# ─────────────────────────────────────────────

class PeriodicEmbedding(nn.Module):
    """
    单列数值编码：sin/cos 展开 → 线性投影到 embed_dim。
    只处理「单列」，方便逐列 Token 化。
    输出：[B] → [B, embed_dim]
    """
    def __init__(self, embed_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_frequencies) * 0.02)
        self.proj  = nn.Linear(n_frequencies * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f   = x.unsqueeze(-1) * self.freqs * 2 * math.pi   # [B, F]
        x_enc = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
        return self.proj(x_enc)


class MultiScaleNumEncoder(nn.Module):
    """
    「少数值列」专用编码器：单列 → n_expand 个多尺度 Token。

    动机：
        当 num_numerical 较少（1~3）时，num_transformer 因序列长度不足而
        被跳过，且 reverse_cross_attn 的 KV 侧信号量不足。
        本模块将同一列在 n_expand 个不同频率尺度下独立编码，构造长度为
        n_expand 的 Token 序列，从而：
          1. 使 num_transformer 可运行（_eff_num_tokens > 1）
          2. 为 cross_attn 提供更丰富的 Key/Value 序列
          3. 不同尺度编码覆盖粗粒度（低频）→ 细粒度（高频）的数值结构

    频率尺度初始化：
        n_expand=1 → [1.0]（退化为标准 PeriodicEmbedding）
        n_expand=2 → [0.1, 10.0]（粗 + 细）
        n_expand=3 → [0.1, 1.0, 10.0]（粗 + 原始 + 细）
        n_expand=4 → [0.1, 0.46, 2.15, 10.0]（对数均匀分布）
        所有尺度均可学习，训练后自适应优化。

    输出：[B] → [B, n_expand, D]
    """
    def __init__(self, embed_dim: int, n_expand: int, n_frequencies: int = 16):
        super().__init__()
        assert n_expand >= 1

        # 每个尺度独立一套 PeriodicEmbedding（共享 n_frequencies 结构，不共享参数）
        self.encoders = nn.ModuleList([
            PeriodicEmbedding(embed_dim, n_frequencies)
            for _ in range(n_expand)
        ])

        # 可学习频率尺度（log 空间，保证始终为正）
        # 初始化：在 [-1, 1] 对数区间均匀分布，即尺度 ∈ [0.1, 10.0]
        init_log = torch.linspace(-1.0, 1.0, n_expand)  # n_expand=1 时为 [0.0] → scale=1.0
        self.log_freq_scales = nn.Parameter(init_log)

        # 可学习 Token 重要性权重（初始均等，训练后自适应）
        self.token_importance = nn.Parameter(torch.zeros(n_expand))

    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        """
        x_col: [B] — 单列数值（已标准化）
        返回:  [B, n_expand, D]
        """
        freq_scales = torch.exp(self.log_freq_scales)              # [n_expand]，正值
        # softmax 后乘以 n_expand，使初始期望幅度 ≈ 1（无偏缩放）
        importance  = torch.softmax(self.token_importance, dim=0) * len(self.encoders)

        tokens = [
            importance[i] * enc(x_col * freq_scales[i])
            for i, enc in enumerate(self.encoders)
        ]
        return torch.stack(tokens, dim=1)                          # [B, n_expand, D]


# ─────────────────────────────────────────────
# 主干模块（不变）
# ─────────────────────────────────────────────

class MLPResidualBlock(nn.Module):
    """AdaLN 风格的残差块（不变）"""
    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm      = nn.LayerNorm(dim)
        self.ff        = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h            = self.norm(x)
        scale, shift = self.cond_proj(emb).chunk(2, dim=-1)
        h            = h * (1 + scale) + shift
        return x + self.ff(h)


class CrossAttentionBlock(nn.Module):
    """
    Q 侧 Token 询问 KV 侧 Token 集合。（不变）
    用途一：数值 Token 询问类别背景（num ← cat）
    用途二：类别 Token 询问数值背景（cat ← num）
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_kv  = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        q_tokens:  torch.Tensor,   # [B, Lq, D]
        kv_tokens: torch.Tensor,   # [B, Lk, D]
    ) -> torch.Tensor:
        q      = self.norm_q(q_tokens)
        kv     = self.norm_kv(kv_tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(q_tokens + out)


class NumToCatFiLM(nn.Module):
    """
    「少数值列」专用：数值 Token 均值 → 类别序列轻量 FiLM 调制。

    动机：
        在 few_num_mode 下，即使通过 MultiScaleNumEncoder 扩展了数值 Token 数量，
        reverse_cross_attn（cat ← num）收到的 KV 信号仍以数值侧为主；
        本模块提供一条更直接的 num→cat 调制路径，以残差方式叠加在
        reverse_cross_attn 之后，作为互补信号。

        gate_scale 初始化为 0（tanh(0)=0），训练初期调制幅度严格为零，
        不干扰正常训练启动；后期逐步学习合适的调制强度。

    输入：
        cat_seq: [B, N_cat, D]
        num_seq: [B, N_eff, D]  （已经过 cross_attn 增强的数值序列）
    输出：
        [B, N_cat, D]（与 cat_seq 形状相同）
    """
    def __init__(self, dim_hidden: int):
        super().__init__()
        self.num_to_cat = nn.Linear(dim_hidden, dim_hidden * 2)
        nn.init.zeros_(self.num_to_cat.weight)
        nn.init.zeros_(self.num_to_cat.bias)

        self.gate_layer = nn.Linear(dim_hidden, dim_hidden)
        nn.init.constant_(self.gate_layer.bias, -1.0)   # sigmoid(-1) ≈ 0.27，初始门控偏小

        self.gate_scale = nn.Parameter(torch.zeros(1))  # tanh(0)=0，初始完全关闭

    def forward(self, cat_seq: torch.Tensor, num_seq: torch.Tensor) -> torch.Tensor:
        h_num       = num_seq.mean(dim=1)                              # [B, D]
        gamma, beta = self.num_to_cat(h_num).chunk(2, dim=-1)         # [B, D] × 2
        gate        = torch.sigmoid(self.gate_layer(h_num))            # [B, D]
        sc          = torch.tanh(self.gate_scale)                      # scalar

        modulation = sc * gate.unsqueeze(1) * (
            cat_seq * gamma.unsqueeze(1) + beta.unsqueeze(1)
        )
        return cat_seq + modulation                                    # [B, N_cat, D]


# ─────────────────────────────────────────────
# 列类型嵌入
# ─────────────────────────────────────────────

class ColumnTypeEmbedding(nn.Module):
    """
    为数值列和类别列的 Token 注入可学习的「列类型标识」向量。

    Type ID：  0 = 数值列   1 = 类别列

    作用：
      1. 让 num_transformer / cat_transformer 感知自己处理的是哪种类型的 Token
      2. cross-attention 中 Q/K 天然携带类型信息，
         注意力头可以分化：「我是 age Token，应重点关注哪些类别 Token」
      3. 对 MultiScaleNumEncoder 扩展出的多个 Token（同属一个数值列）
         统一标注 type=0，让模型知道这些不同频率视角均属数值模态
      4. 当 cat_col 中包含 target label 时（如 PIMA），
         为 label Token 标注 type=1 与普通数值特征区分，
         帮助模型学习「给定 label 的条件数值分布」

    参数量：2 × dim_hidden（极轻量，不影响训练效率）
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(2, embed_dim)          # 0=num, 1=cat
        nn.init.normal_(self.emb.weight, std=0.02)     # 小初始化，不干扰早期训练

    def get(self, type_id: int, device: torch.device) -> torch.Tensor:
        """返回 [D] 类型向量，可直接广播加到 [B, D] 的 Token 上。"""
        return self.emb(torch.tensor(type_id, dtype=torch.long, device=device))


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────

class TabMLPDenoise(nn.Module):
    """
    混合型表格去噪扩散模型（v3）。

    在 v2（few_num_mode / MultiScaleNumEncoder）基础上新增三项修复：

    ① Column Type Embedding (CTE)
       为每个 Token 注入「数值列=0 / 类别列=1」的可学习标识。
       让 Transformer 和 cross-attention 学习类型感知路由，
       对 PIMA（target 加入 cat）和 Thyroid（单 age 列）均有效。

    ② 并行双向 Cross-Attention
       修复原版顺序执行导致的「cat 信息经 num 折射回 cat」问题。
       reverse_cross_attn 使用 cross_attn 之前的 num_seq 快照，
       保证 cat ← num 方向得到真实数值信号。

    ③ 双路 CLS 聚合（Dual CLS Pool）
       cls_num 专门从 num_seq 聚合，cls_cat 专门从 cat_seq 聚合，
       拼接后经 proj_fuse（输入 dim_hidden*2）得到 h_shared。
       解决：单路 pool 在 N_cat >> N_num 时稀释数值信息的问题。

    接口完全不变（forward 签名和外部属性均保持兼容）。
    """

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
        # ── 少数值列自适应（新增可选参数，不影响已有调用）────────────────
        few_num_threshold: int = 3,
        # num_numerical <= few_num_threshold 时激活 few_num_mode
        # 设为 0 可完全禁用（退化为原始行为）
        n_expand: int = 3,
        # 每个数值列扩展为多少个 Token（few_num_mode 生效时）
        # n_expand=1 时等价于原始 PeriodicEmbedding（无扩展）
    ):
        super().__init__()

        self.task          = task
        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']          # list[int]
        true_marginals     = info.get('true_marginals', None)

        # ── 少数值列模式判断 ─────────────────────────────────────────────
        # few_num_mode 激活条件：
        #   1. 确实有数值列（num_numerical > 0）
        #   2. 数值列数量不超过阈值
        #   3. 扩展倍数 > 1（n_expand=1 时无扩展，无需切换编码器）
        self.few_num_mode = (
            0 < self.num_numerical <= few_num_threshold
            and n_expand > 1
        )
        self.n_expand = n_expand if self.few_num_mode else 1

        # 有效数值 Token 数 = num_numerical（正常） 或 num_numerical*n_expand（few_num_mode）
        # 用于决定 num_transformer 是否初始化（需要 > 1 才有意义）
        self._eff_num_tokens = self.num_numerical * self.n_expand

        if self.few_num_mode:
            print(
                f"[TabMLPDenoise] few_num_mode 激活："
                f"num_numerical={self.num_numerical} <= threshold={few_num_threshold}，"
                f"每列扩展为 {n_expand} 个 Token，"
                f"有效数值 Token 数 {self.num_numerical} → {self._eff_num_tokens}"
            )

        # ── 1. 逐列编码器 ────────────────────────────────────────────────
        # 数值列：
        #   正常模式  → PeriodicEmbedding（每列产出 1 个 [B, D] Token）
        #   few_num_mode → MultiScaleNumEncoder（每列产出 n_expand 个 [B, D] Token）
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

        # 类别列：每列独立的 Embedding → 线性投影（不变）
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

        # ── 2. 可学习门控尺度（不变）────────────────────────────────────
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

        # ── 3. 列类型嵌入（新增）────────────────────────────────────────
        # 为 num / cat Token 分别注入 type=0 / type=1 的可学习标识向量。
        # 位置：Token 化完成后、任何 attention 之前施加，
        # 使 Transformer 和 cross-attention 天然感知列类型。
        self.col_type_emb = ColumnTypeEmbedding(dim_hidden)

        # ── 4. Cross-Attention（新增：并行双向）─────────────────────────
        # !! 关键修复：
        #    原版顺序执行 cat_seq = reverse_cross_attn(cat_seq, num_seq_after_cross)
        #    导致 cat 通过 num 读回了自己的信息（「cat 信息的折射」）。
        #    修复方案：在 forward 中保存 cross_attn 前的 num_seq 快照，
        #    reverse_cross_attn 使用该快照作为 KV，实现真正的并行双向。
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cross_attn         = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
            if not self.few_num_mode:
                self.reverse_cross_attn = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
        else:
            self.cross_attn         = None
            if not self.few_num_mode:
                self.reverse_cross_attn = None

        # ── 4b. few_num_mode 补充：NumToCatFiLM ─────────────────────────
        if self.few_num_mode and len(self.num_classes) > 0:
            self.num_to_cat_film = NumToCatFiLM(dim_hidden)
        else:
            self.num_to_cat_film = None

        # ── 5. 列间 Transformer ──────────────────────────────────────────
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

        # ── 6. 双路 CLS 聚合（替换原单一 pool_query）───────────────────
        # !! 关键修复：
        #    原版 pool_query 对所有 num+cat token 软聚合，
        #    当 N_cat >> N_num 时 h_shared 天然被 cat 侧主导（PIMA / Thyroid 的核心问题）。
        #    修复：cls_num 专门聚合 num_seq，cls_cat 专门聚合 cat_seq，
        #    再将两路结果拼接后经 proj_fuse（输入 dim_hidden*2）得到 h_shared。
        #    数值特征始终有独立的「发言权」，不被 cat 数量稀释。
        self.cls_num = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.cls_cat = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.pool_num_attn  = nn.MultiheadAttention(dim_hidden, nhead_col, dropout=dropout, batch_first=True)
        self.pool_cat_attn  = nn.MultiheadAttention(dim_hidden, nhead_col, dropout=dropout, batch_first=True)
        self.pool_norm_num_q  = nn.LayerNorm(dim_hidden)
        self.pool_norm_num_kv = nn.LayerNorm(dim_hidden)
        self.pool_norm_cat_q  = nn.LayerNorm(dim_hidden)
        self.pool_norm_cat_kv = nn.LayerNorm(dim_hidden)

        # ── 7. 条件编码（不变）──────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ── 8. 聚合投影（dim_hidden*2 → dim_hidden，适配双路 CLS 拼接）─
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden),
        )

        # ── 8. 双专属主干（不变）────────────────────────────────────────
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

        # ── 9. 输出头（不变）────────────────────────────────────────────
        if self.num_numerical > 0:
            self.out_num = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, self.num_numerical)
            )

        self.out_cat = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.SiLU(),
                nn.Linear(dim_hidden, c)
            ) for c in self.num_classes
        ])

        if true_marginals is not None:
            self._apply_prior_initialization(true_marginals)

    # ────────────────────────────────────────────────────────────────────
    def _apply_prior_initialization(self, true_marginals):
        """先验初始化：输出头偏置 = log(prior)（不变）"""
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                prior     = true_marginals[i]
                log_prior = torch.log(prior + 1e-8)
                head[-1].bias.copy_(log_prior)
                head[-1].weight.data.mul_(0.1)

    # ────────────────────────────────────────────────────────────────────
    def _encode_num_tokens(self, x_num: torch.Tensor) -> list:
        """
        数值列编码 → Token 列表（每项 [B, D]）。

        正常模式  : 每列 → 1 个 Token（PeriodicEmbedding）
        few_num_mode: 每列 → n_expand 个 Token（MultiScaleNumEncoder）

        返回的 num_tokens 长度：
            正常模式  : num_numerical
            few_num_mode: num_numerical * n_expand = _eff_num_tokens
        """
        num_tokens = []
        if self.num_col_encoders is None:
            return num_tokens

        for i, enc in enumerate(self.num_col_encoders):
            if self.few_num_mode:
                # MultiScaleNumEncoder.forward → [B, n_expand, D]
                expanded = enc(x_num[:, i])                    # [B, n_expand, D]
                for j in range(expanded.shape[1]):
                    num_tokens.append(expanded[:, j])          # [B, D]
            else:
                # PeriodicEmbedding.forward → [B, D]
                num_tokens.append(enc(x_num[:, i]))

        return num_tokens

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x_in: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor):
        B = x_in.shape[0]

        # ── 条件向量（不变）─────────────────────────────────────────────
        t_emb     = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_raw_emb = self.label_emb(y.view(-1).long())
        emb       = t_emb + y_raw_emb                         # [B, dim_t]

        # ── 拆分输入（不变）─────────────────────────────────────────────
        x_num      = x_in[:, :self.num_numerical]             # [B, N_num]
        x_cat_prob = x_in[:, self.num_numerical:].long()

        # ── 逐列 Token 化 ────────────────────────────────────────────────
        num_tokens = self._encode_num_tokens(x_num)

        # 类别侧 Token 化
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
        # 在任何 attention 计算之前，为每个 Token 叠加类型嵌入。
        # num Token（含 MultiScaleNumEncoder 扩展出的多视角 Token）均标记 type=0；
        # cat Token（含 target label 列）均标记 type=1。
        # 效果：attention head 可学习「我是 age Token，应优先关注哪些 cat Token」。
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
        # !! 并行修复：
        #    cross_attn 前保存 num_seq 快照（num_seq_pre）。
        #    num_seq_updated = cross_attn(num_seq, cat_seq)   # num ← cat（原始 cat）
        #    cat_seq_updated = reverse_cross_attn(cat_seq, num_seq_pre)  # cat ← 原始 num
        #    确保两个方向均读取对方 cross-attn 之前的序列，消除 cat 信息折射。
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
            num_seq = self.cross_attn(num_seq, cat_seq)               # num ← cat（原始）
            if not self.few_num_mode:
                cat_seq = self.reverse_cross_attn(cat_seq, num_seq_pre)   # cat ← 原始 num！

        # ── 阶段2b：NumToCatFiLM（仅 few_num_mode）─────────────────────
        if self.num_to_cat_film is not None and num_seq.shape[1] > 0:
            cat_seq = self.num_to_cat_film(cat_seq, num_seq)

        # ── 阶段3：双路 CLS 聚合（替换原单一 pool_query）────────────────
        # cls_num 专门从 num_seq 中提取数值侧信息；
        # cls_cat 专门从 cat_seq 中提取类别侧信息。
        # 两路结果拼接后经 proj_fuse 得到 h_shared。
        # 无论 N_cat 多大，数值特征始终通过独立的 cls_num 路径保留完整表示。
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

        # ── 双专属主干 + 输出头（不变）──────────────────────────────────
        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h_num = h_shared.clone()
            for layer in self.num_backbone:
                h_num = layer(h_num, emb)
            out_n = self.out_num(h_num)

        out_c_list = []
        if self.cat_backbone is not None:
            h_cat = h_shared.clone()
            for layer in self.cat_backbone:
                h_cat = layer(h_cat, emb)
            out_c_list = [head(h_cat) for head in self.out_cat]

        out_c = (
            torch.cat(out_c_list, dim=-1)
            if out_c_list
            else torch.empty((B, 0), device=x_in.device)
        )

        return torch.cat([out_n, out_c], dim=-1)'''