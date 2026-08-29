# core/denoise_model.py
import torch
import torch.nn as nn
import math


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
# 主模型
# ─────────────────────────────────────────────

class TabMLPDenoise(nn.Module):
    """
    混合型表格去噪扩散模型。

    架构改进点：
      1. 可学习门控尺度（gate_scale），取代硬编码 0.3
      2. 逐列 Token 化：每一列独享独立编码器，拼成 [B, N_cols, D] 序列
      3. Cross-Attention：数值 Token Query 类别 Token 的语义背景（双向）
      4. Hybrid Transformer-MLP 主干：TransformerEncoderLayer 处理列间关联，
         再经 MLPResidualBlock 栈接收时间步 / 标签条件

    新增：少数值列自适应（few_num_mode）
      当 num_numerical <= few_num_threshold 时自动激活：
        a. 每个数值列改用 MultiScaleNumEncoder，在 n_expand 个频率尺度下编码，
           将有效数值 Token 数量从 num_numerical 扩展到 num_numerical * n_expand，
           使 num_transformer 得以运行
        b. 新增 NumToCatFiLM，在 reverse_cross_attn 之后提供直接的 num→cat 调制，
           弥补数值侧 KV 数量少时跨模态信号不足的问题

    接口不变：
        所有新参数（few_num_threshold / n_expand）均为关键字参数且有默认值，
        不影响任何已有调用代码。forward 签名完全不变。
    """

    def __init__(
        self,
        task,
        info,
        num_classes_cond: int = 2,
        dim_t: int = 128,
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
        self.cat_dropout = nn.Dropout(0.1)
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

        # ── 3. Cross-Attention（不变）───────────────────────────────────
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cross_attn         = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
            self.reverse_cross_attn = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
        else:
            self.cross_attn         = None
            self.reverse_cross_attn = None

        # ── 3b. few_num_mode 补充：NumToCatFiLM ─────────────────────────
        # 在 reverse_cross_attn 之后提供直接 num→cat 调制路径，
        # 弥补少量数值 Token 对 cat 分支的跨模态信号不足。
        # 仅在 few_num_mode 且同时有 cat 特征时初始化。
        if self.few_num_mode and len(self.num_classes) > 0:
            self.num_to_cat_film = NumToCatFiLM(dim_hidden)
        else:
            self.num_to_cat_film = None

        # ── 4. 列间 Transformer ──────────────────────────────────────────
        # 关键变更：
        #   num_transformer 的初始化条件从 "num_numerical > 1"
        #   改为 "_eff_num_tokens > 1"，从而在 few_num_mode 下
        #   即使 num_numerical=1，扩展后的 3 个 Token 也能过 Transformer。
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

        # ── 5. Pool-Query 聚合（不变）────────────────────────────────────
        self.pool_query = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.pool_attn  = nn.MultiheadAttention(
            embed_dim=dim_hidden, num_heads=nhead_col,
            dropout=dropout, batch_first=True,
        )
        self.pool_norm_q  = nn.LayerNorm(dim_hidden)
        self.pool_norm_kv = nn.LayerNorm(dim_hidden)

        # ── 6. 条件编码（不变）──────────────────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ── 7. 聚合投影（不变）──────────────────────────────────────────
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
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
        # 数值侧：通过 _encode_num_tokens 统一处理正常/few_num_mode 两种路径
        # 结果长度：正常=num_numerical，few_num_mode=num_numerical*n_expand
        num_tokens = self._encode_num_tokens(x_num)

        # 类别侧：不变
        cat_tokens, start = [], 0
        for emb_layer, proj, n_class in zip(
            self.cat_embed_layers, self.cat_col_projs, self.num_classes
        ):
            end  = start + n_class
            prob = x_cat_prob[:, start:end]
            assert prob.shape[1] == n_class
            tok  = self.cat_dropout(proj(prob.float() @ emb_layer.weight))
            cat_tokens.append(tok)
            start = end

        # ── 阶段0：Token 化后 FiLM（在 Intra-Attn 之前）────────────────
        # 注：num_tokens 在 few_num_mode 下已有 _eff_num_tokens 个条目，
        #     FiLM 对列表中的每个 Token 独立施加，不受数量影响。
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
        # few_num_mode 下 num_seq 形状：[B, _eff_num_tokens, D]
        # 正常模式      下 num_seq 形状：[B, num_numerical, D]
        # 后续所有操作对序列长度无假设，两种情况均可透明处理。
        num_seq = torch.stack(num_tokens, dim=1)              # [B, L_num, D]
        cat_seq = torch.stack(cat_tokens, dim=1)              # [B, L_cat, D]

        if self.num_transformer is not None:
            num_seq = self.num_transformer(num_seq)
        if self.cat_transformer is not None:
            cat_seq = self.cat_transformer(cat_seq)

        # ── 阶段2：FiLM + 双向 Cross-Attention ──────────────────────────
        if self.cross_attn is not None:
            h_cat_mean = cat_seq.mean(dim=1)
            gamma_beta = self.cat_to_num(h_cat_mean)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gate        = torch.sigmoid(self.gate_layer(h_cat_mean))
            scale_coef  = torch.tanh(self.gate_scale)

            num_seq = num_seq + scale_coef * gate.unsqueeze(1) * (
                num_seq * gamma.unsqueeze(1) + beta.unsqueeze(1)
            )
            num_seq = self.cross_attn(num_seq, cat_seq)           # num ← cat
            cat_seq = self.reverse_cross_attn(cat_seq, num_seq)   # cat ← num

        # ── 阶段2b：NumToCatFiLM 补充调制（仅 few_num_mode）─────────────
        # 在 reverse_cross_attn 之后，为 cat_seq 额外注入数值侧均值信号。
        # gate_scale=0 初始化保证训练早期此路径幅度为零，不影响稳定性。
        if self.num_to_cat_film is not None:
            cat_seq = self.num_to_cat_film(cat_seq, num_seq)

        # ── 阶段3：Pool-Query 聚合（不变）───────────────────────────────
        all_seq = torch.cat([num_seq, cat_seq], dim=1)            # [B, L_total, D]

        q        = self.pool_norm_q(self.pool_query.expand(B, -1, -1))
        kv       = self.pool_norm_kv(all_seq)
        #pool_out, _ = self.pool_attn(q, kv, kv)
        pool_out, attn_weights = self.pool_attn(
            q, kv, kv, need_weights=True, average_attn_weights=False
        )

        h        = (self.pool_query.expand(B, -1, -1) + pool_out).squeeze(1)  # [B, D]
        h_shared = self.proj_fuse(h)

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

        return torch.cat([out_n, out_c], dim=-1)