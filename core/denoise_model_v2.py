# core/denoise_model_v2.py  (v2 — 双路 Cross-Attention，显式跨模态路由)
"""
TabMLPDenoise v2：数值/类别双路 + 双向 Cross-Attention + FiLM
=================================================================

定位
----
与 v1（统一 Transformer，隐式路由）构成一组对照实验：

    v2 假设：数值列和类别列的分布特性差异足够大，值得先各自过独立的
             intra-modal Transformer，再通过显式的 cross-attention +
             FiLM 交换信息 —— 比让 attention 从零学习跨模态路由更高效，
             尤其在某个固定 N_num/N_cat 比例（如 Framingham）上。

    代价：cross_attn / FiLM / 双 CLS 的路由方式是针对特定列数比例调的，
          迁移到比例差异很大的数据集（Thyroid 几乎全数值、PIMA 目标列
          混在类别里）时可能不稳定 —— 这正是 v1 存在的原因。

架构
----
    x_in
     │
     ├─ num_col_encoders (PeriodicEmbedding，或 few_num_mode 下的
     │   MultiScaleNumEncoder × N_num)         + col_type_emb(0)
     └─ cat_embed_layers + cat_col_projs (× N_cat) + col_type_emb(1)
     │
     ▼
    阶段0：FiLM（cat 均值 → num scale/shift，Intra-Attn 之前）
     │
     ▼
    阶段1：模态内自注意力（num_transformer / cat_transformer，各自独立）
     │
     ▼
    阶段2：并行双向 Cross-Attention
        num_seq_updated = cross_attn(num_seq, cat_seq)          # num ← cat
        cat_seq_updated = reverse_cross_attn(cat_seq, num_seq_pre)  # cat ← num（cross_attn 前快照，避免折射）
     │
     ▼
    阶段3：双路 CLS 聚合 —— cls_num 聚合 num_seq，cls_cat 聚合 cat_seq，
           拼接后经 proj_fuse 得到 h_shared
     │
     ├─ num_backbone (AdaLN MLPResidualBlock × M) → (可选 shared_dec) → out_num
     └─ cat_backbone (AdaLN MLPResidualBlock × M) → (可选 shared_dec) → out_cat_heads

few_num_mode（数值列 ≤ few_num_threshold 时自动激活，v1 无此机制）
--------------------------------------------------------------------
    数值列太少时，num_transformer / cross_attn 的 KV 侧信号量不足。
    MultiScaleNumEncoder 把每个数值列在多个可学习频率尺度下展开成
    n_expand 个 token；NumToCatFiLM 额外提供一条 num→cat 的直接调制
    通路（gate 初始为 0，训练初期不干扰）。

share_decoder 参数（新增，与 v1 对齐）
---------------------------------------
    True  （默认）：num/cat 各一个共享展开层 + 轻量列头，参数量与 v1
                    对齐（约减少 12 倍），消除"v2 解码器容量更大"这个
                    混杂变量，保证 v1/v2 消融只测路由机制本身。
    False           每列独立 Sequential(Linear(D,D), SiLU, Linear(D,c))，
                    即原始版本。

外部接口（与 cat_loss.py / num_loss.py / 训练循环 完全兼容）
------------------------------------------------------------
    model.num_numerical_features  —— cat_loss.py 使用
    model.slices_for_classes      —— cat_loss.py 使用
    model.true_marginals          —— cat_loss.py 使用
    model.model_cfg               —— 由训练循环外部赋值，不变
    forward(x_in, timesteps, y)   —— 签名不变

    注：原始存档版本遗漏了 num_numerical_features / slices_for_classes /
    true_marginals 这三个属性的赋值，若接入 cat_loss.py 会直接报错，
    本文件已补上（见 __init__ 开头）。
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

    输出：[B] → [B, n_expand, D]
    """
    def __init__(self, embed_dim: int, n_expand: int, n_frequencies: int = 16):
        super().__init__()
        assert n_expand >= 1

        self.encoders = nn.ModuleList([
            PeriodicEmbedding(embed_dim, n_frequencies)
            for _ in range(n_expand)
        ])

        init_log = torch.linspace(-1.0, 1.0, n_expand)  # n_expand=1 → [0.0] → scale=1.0
        self.log_freq_scales = nn.Parameter(init_log)

        self.token_importance = nn.Parameter(torch.zeros(n_expand))

    def forward(self, x_col: torch.Tensor) -> torch.Tensor:
        """
        x_col: [B] — 单列数值（已标准化）
        返回:  [B, n_expand, D]
        """
        freq_scales = torch.exp(self.log_freq_scales)              # [n_expand]，正值
        importance  = torch.softmax(self.token_importance, dim=0) * len(self.encoders)

        tokens = [
            importance[i] * enc(x_col * freq_scales[i])
            for i, enc in enumerate(self.encoders)
        ]
        return torch.stack(tokens, dim=1)                          # [B, n_expand, D]


class MLPResidualBlock(nn.Module):
    """AdaLN 风格的残差块（dim → dim → dim，比 v1 简洁，无 zero-init）"""
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
    Q 侧 Token 询问 KV 侧 Token 集合。
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
        gate: torch.Tensor = None
    ) -> torch.Tensor:
        q      = self.norm_q(q_tokens)
        kv     = self.norm_kv(kv_tokens)
        out, _ = self.attn(q, kv, kv)

        if gate is not None:
            out = out * gate

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


class ColumnTypeEmbedding(nn.Module):
    """
    为数值列和类别列的 Token 注入可学习的「列类型标识」向量。
    Type ID：0 = 数值列   1 = 类别列
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.emb = nn.Embedding(2, embed_dim)          # 0=num, 1=cat
        nn.init.normal_(self.emb.weight, std=0.02)     # 小初始化，不干扰早期训练

    def get(self, type_id: int, device: torch.device) -> torch.Tensor:
        """返回 [D] 类型向量，可直接广播加到 [B, D] 的 Token 上。"""
        return self.emb(torch.tensor(type_id, dtype=torch.long, device=device))


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class TabMLPDenoise(nn.Module):
    """
    混合型表格去噪扩散模型（v2：双路 + 显式 Cross-Attention）。
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

        # ── cat_loss.py 兼容接口（原始存档版本遗漏，此处补上）───────────
        self.num_numerical_features = self.num_numerical
        _start = 0
        self.slices_for_classes = []
        for n_c in self.num_classes:
            self.slices_for_classes.append(slice(_start, _start + n_c))
            _start += n_c
        self.true_marginals = true_marginals

        # ── 少数值列模式判断 ─────────────────────────────────────────────
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
        # cross_attn 前保存 num_seq 快照，reverse_cross_attn 用该快照作为
        # KV，确保 cat ← num 方向读到的是真实数值信号，而非 num 已经融合
        # 过 cat 之后的折射结果。
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