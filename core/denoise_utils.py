# core/denoise_utils.py
import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# 工具函数
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
# 单列数值编码
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


class RFFEmbedding(nn.Module):
    def __init__(self, embed_dim: int, n_frequencies: int = 16):
        super().__init__()
        # 固定的随机频率，不参与梯度更新
        self.register_buffer('freqs', torch.randn(n_frequencies) * 10.0)
        self.proj = nn.Linear(n_frequencies * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B]
        x_f = x.unsqueeze(-1) * self.freqs * 2 * math.pi
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


# ─────────────────────────────────────────────
# 主干残差模块
# ─────────────────────────────────────────────
class MLPResidualBlock(nn.Module):
    """AdaLN 风格的残差块"""
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
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

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
        gate: torch.Tensor = None
    ) -> torch.Tensor:
        """
        gate: [B, 1, 1] 或 None，对注意力输出进行缩放。
              若为 None，则行为与原来完全一致（gate=1）。
        """
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
