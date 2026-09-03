# core/denoise_model_v1.py  (v1 — Unified FT-Transformer, 隐式跨模态路由)
"""
TabMLPDenoise v1：统一 Transformer 架构
=========================================

定位
----
以此版本为基准（baseline）。数值列与类别列一律 token 化后拼进同一个
Transformer 序列，跨模态交互完全交给 self-attention 自主学习，不做任何
手工路由假设。与 v2（显式 cross-attention + FiLM + 双 CLS）构成一组对照：

    v1 假设：跨模态路由应由 attention 从数据中学出来，
             不应硬编码某个 N_num/N_cat 比例下的先验结构。
    v2 假设：数值/类别的交互模式足够特殊，显式路由（cross-attn/FiLM）
             能提供比隐式 attention 更强的归纳偏置。

两者在你的数据集梯度（N_num/N_cat 比例差异很大）上做消融对比，就是
论文方法部分的核心实验设计。

架构
----
    x_in
     │
     ├─ num_col_encoders (PeriodicEmbedding × N_num)  + col_type_emb(0)
     └─ cat_embed_layers + cat_col_projs (× N_cat)      + col_type_emb(1)
     │
     ▼
    [CLS] ‖ num_tokens ‖ cat_tokens    → [B, 1 + N_num + N_cat, D]
     │
     ▼
    unified_transformer (pre-norm TransformerEncoder, L 层)
     │
     ▼
    h_shared = h_seq[:, 0]             # [CLS] 无偏全局聚合 → [B, D]
     │
     ├─ num_backbone (AdaLN MLPResidualBlock × M) → (可选 shared_dec) → out_num
     └─ cat_backbone (AdaLN MLPResidualBlock × M) → (可选 shared_dec) → out_cat_heads

share_decoder 参数（新增）
--------------------------
    True  （默认）：num/cat 各一个共享展开层(Linear→SiLU→Dropout→LayerNorm)，
                    列头退化为轻量 Linear(D, c)。
                    hepatitis（13类别列, D=512）实测：解码层参数从
                    3,430K 降到 264K + 15K = 279K（减少约12.3倍），
                    直接缓解类别列过拟合 / DCR 隐私分数下降的问题。
    False           每列独立 Sequential(Linear(D,D), SiLU, Linear(D,c))，
                    即修复前的原始版本，参数量大但列间完全独立。

    与 v2 对比消融时，两边用同一个 share_decoder 取值，才能保证差异
    只来自"路由机制"而不是"解码器容量"。

外部接口（与 cat_loss.py / num_loss.py / 训练循环 完全兼容）
------------------------------------------------------------
    model.num_numerical_features  —— cat_loss.py 使用
    model.slices_for_classes      —— cat_loss.py 使用
    model.true_marginals          —— cat_loss.py 使用
    model.model_cfg               —— 由训练循环外部赋值，不变
    forward(x_in, timesteps, y)   —— 签名不变

超参数推荐
----------
    小数据集（N < 500，如 Hepatitis / BCC）：
        dim_hidden=128, num_layers=2, num_backbone_layers=2, dropout=0.2

    中等数据集（500 < N < 5000，如 Framingham、Cleveland、CKD、Cirrhosis）：
        dim_hidden=256, num_layers=3, num_backbone_layers=2, dropout=0.1

    大数据集（N > 5000）：
        dim_hidden=256, num_layers=4, num_backbone_layers=3, dropout=0.1
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
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
    内部升维 dim → dim*2 → dim，增加表达能力（对比 v2 的 dim → dim）。
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
            encoder_layer        = encoder_layer,
            num_layers           = num_layers,
            norm                 = nn.LayerNorm(dim_hidden),
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