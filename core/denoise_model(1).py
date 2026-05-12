# core/denoise_model.py
import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# 编码模块
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicEmbedding(nn.Module):
    """
    单列数值编码：sin/cos 周期展开 → 线性投影到 embed_dim。
    每列独享一个实例，是逐列 Token 化的基础单元。
    （原版处理全部数值列；此版本每次只处理单列标量 [B]）
    """
    def __init__(self, embed_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_frequencies) * 0.02)
        self.proj  = nn.Linear(n_frequencies * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B] → [B, embed_dim]
        x_f   = x.unsqueeze(-1) * self.freqs * 2 * math.pi
        x_enc = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
        return self.proj(x_enc)


class MLPResidualBlock(nn.Module):
    """AdaLN 风格的条件残差块（与原版相同）"""
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
    数值 Token → Query，类别 Token 集合 → Key/Value。
    数值列在生成时通过 Attention 读取类别背景语义。
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn     = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_q   = nn.LayerNorm(dim)
        self.norm_kv  = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, q_tokens: torch.Tensor, kv_tokens: torch.Tensor) -> torch.Tensor:
        q      = self.norm_q(q_tokens)
        kv     = self.norm_kv(kv_tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(q_tokens + out)


# ─────────────────────────────────────────────────────────────────────────────
# 改进 3：带对称正则的列间 Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SymmetricColumnTransformer(nn.Module):
    """
    列间 Self-Attention + 对称性正则。

    原理
    ----
    真实的统计关联是无向的（A 影响 B 的程度应接近 B 影响 A），
    但 Q/K 分离的 Attention 不天然保证 A[i,j] ≈ A[j,i]。
    正则项 ||A - Aᵀ||²_F 把这一先验软约束编码进训练过程，
    引导模型学习对称的列间依赖图，与具体数据集无关。

    返回
    ----
    (x, sym_loss)
      x:        [B, N, D]，经过 Transformer 的列 Token 序列
      sym_loss: 标量，建议训练时乘以小权重（0.005 ~ 0.02）加入总损失
    """
    def __init__(self, dim: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        # Pre-LN Self-Attention
        h = self.norm1(x)
        out, attn_w = self.attn(
            h, h, h,
            need_weights=True,
            average_attn_weights=False,    # 保留每个 head: [B, nhead, N, N]
        )
        # 对称正则：A 与 Aᵀ 的 Frobenius 距离
        sym_loss = (attn_w - attn_w.transpose(-1, -2)).pow(2).mean()

        x = x + out
        x = x + self.ff(self.norm2(x))
        return x, sym_loss


# ─────────────────────────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────────────────────────

class TabMLPDenoise(nn.Module):
    """
    在逐列 Token 化 + Cross-Attention + 可学习门控的基础上，
    新增三项关联建模改进：

    改进 1 —— 双流解码（Dual-stream Output）
        Column Transformer 之后不立即均值池化，而是保留每列 Token（列专属流）。
        每个输出头同时读「列专属 Token」（含列间交互后的关联信息）
        和「全局向量」（携带时间步 / 标签的扩散条件），两路相加解码。
        均值池化会把联合分布信息压缩为边缘分布的叠加；双流解码保留了
        列特异性，让任意两列之间的条件依赖可以被输出头直接使用。

    改进 2 —— 时间步感知的列间注意力（Timestep-aware Column Attention）
        在列间 Transformer 之前，将时间步嵌入投影并注入每个列 Token。
        Attention 权重因此随噪声水平变化：高噪声阶段各列 Token 的差异
        被时间步信号主导，关联权重趋向均匀（允许独立去噪）；低噪声阶段
        列特异性更强，关联权重更集中（施加联合分布约束）。
        这让列间关联的学习与扩散过程本身的动力学相耦合。

    改进 3 —— 注意力对称正则（Symmetric Attention Regularization）
        SymmetricColumnTransformer 对注意力矩阵施加 ||A - Aᵀ||²_F 正则，
        引导模型学习无向关联图，对任何表格数据适用。
        sym_loss 通过 return_aux=True 或 self.last_sym_loss 暴露给训练循环。

    接口（向后兼容）
    ----
    forward(x_in, timesteps, y)              → Tensor（与原版相同）
    forward(x_in, timesteps, y, return_aux=True) → (Tensor, {'sym_loss': Tensor})
    self.last_sym_loss                       → 上一次 forward 的 sym_loss（detach）
    """

    def __init__(
        self,
        info,
        num_classes_cond: int = 2,
        dim_t: int            = 128,
        dim_hidden: int       = 256,
        num_layers: int       = 3,
        dropout: float        = 0.1,
        n_frequencies: int    = 16,
        nhead_col: int        = 4,
        nhead_cross: int      = 4,
    ):
        super().__init__()

        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']
        true_marginals     = info.get('true_marginals', None)

        # ── 1. 逐列数值编码器 ─────────────────────────────────────────
        if self.num_numerical > 0:
            self.num_col_encoders = nn.ModuleList([
                PeriodicEmbedding(dim_hidden, n_frequencies)
                for _ in range(self.num_numerical)
            ])
        else:
            self.num_col_encoders = None

        # ── 2. 逐列类别编码器 ─────────────────────────────────────────
        self.cat_dropout = nn.Dropout(0.2)
        if len(self.num_classes) > 0:
            self.cat_embed_layers = nn.ModuleList([
                nn.Embedding(c, min(16, c * 2)) for c in self.num_classes
            ])
            self.cat_col_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(min(16, c * 2), dim_hidden),
                    nn.SiLU(),
                    nn.LayerNorm(dim_hidden),
                )
                for c in self.num_classes
            ])
        else:
            self.cat_embed_layers = None
            self.cat_col_projs    = None

        # ── 3. 可学习门控 FiLM ──────────────────────────────────────
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cat_to_num = nn.Linear(dim_hidden, dim_hidden * 2)
            nn.init.zeros_(self.cat_to_num.weight)
            nn.init.zeros_(self.cat_to_num.bias)

            self.gate_layer = nn.Linear(dim_hidden, dim_hidden)
            nn.init.xavier_uniform_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, -1.0)

            # tanh(gate_scale)：初始 0，训练中自适应增长
            self.gate_scale = nn.Parameter(torch.zeros(1))

        # ── 4. Cross-Attention（数值 Query 类别 KV）──────────────────
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cross_attn = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
        else:
            self.cross_attn = None

        # ── 5. 条件编码（时间步 + 标签）─────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.label_emb       = nn.Embedding(num_classes_cond + 1, dim_t)
        self.label_to_hidden = nn.Linear(dim_t, dim_hidden)

        # ── 改进 2：时间步 → 列 Token 投影 ───────────────────────────
        self.time_to_col = nn.Linear(dim_t, dim_hidden)

        # ── 改进 3：带对称正则的列间 Transformer ─────────────────────
        total_cols = self.num_numerical + len(self.num_classes)
        if total_cols > 1:
            self.column_transformer = SymmetricColumnTransformer(
                dim_hidden, nhead_col, dropout
            )
        else:
            self.column_transformer = None

        # ── 改进 1：列专属精炼 FFN ───────────────────────────────────
        # Sequential 的 Linear 对 last dim 操作，自动广播到序列维 N
        self.col_refine = nn.Sequential(
            nn.LayerNorm(dim_hidden),
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )

        # ── 全局流：均值聚合 → 标签注入 → AdaLN 残差主干 ────────────
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden),
        )
        self.layers = nn.ModuleList([
            MLPResidualBlock(dim_hidden, dim_t, dropout)
            for _ in range(num_layers)
        ])

        # ── 改进 1：逐列数值输出头（取代原 Linear → N_num）──────────
        if self.num_numerical > 0:
            self.out_num_heads = nn.ModuleList([
                nn.Linear(dim_hidden, 1) for _ in range(self.num_numerical)
            ])

        # 类别输出头（与原版相同）
        self.out_cat = nn.ModuleList([
            nn.Linear(dim_hidden, c) for c in self.num_classes
        ])

        # sym_loss 缓存，供不使用 return_aux 的训练循环读取
        self.register_buffer('last_sym_loss', torch.zeros(1))

        if true_marginals is not None:
            self._apply_prior_initialization(true_marginals)

    # ──────────────────────────────────────────────────────────────────
    def _apply_prior_initialization(self, true_marginals):
        """将分类输出层的偏置初始化为训练集分布的对数"""
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                prior     = true_marginals[i].cpu()
                log_prior = torch.log(prior + 1e-8)
                head.bias.copy_(log_prior)
                head.weight.data.mul_(0.1)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x_in: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
        return_aux: bool = False,
    ):
        B = x_in.shape[0]

        # ── 条件嵌入 ─────────────────────────────────────────────────
        t_emb     = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_raw_emb = self.label_emb(y.squeeze().long())
        emb       = t_emb + y_raw_emb          # AdaLN 条件向量 [B, dim_t]

        # ── 拆分输入 ─────────────────────────────────────────────────
        x_num = x_in[:, :self.num_numerical]
        x_cat = x_in[:, self.num_numerical:].long()

        # ── 逐列 Token 化 ────────────────────────────────────────────
        num_tokens: list[torch.Tensor] = []
        cat_tokens: list[torch.Tensor] = []

        if self.num_col_encoders is not None:
            for i, enc in enumerate(self.num_col_encoders):
                num_tokens.append(enc(x_num[:, i]))          # [B, D]

        if self.cat_embed_layers is not None:
            for i, (emb_layer, proj) in enumerate(
                zip(self.cat_embed_layers, self.cat_col_projs)
            ):
                tok = proj(emb_layer(x_cat[:, i]))
                cat_tokens.append(self.cat_dropout(tok))     # [B, D]

        # ── 可学习门控 FiLM + Cross-Attention ────────────────────────
        if self.cross_attn is not None and num_tokens and cat_tokens:
            h_cat_mean  = torch.stack(cat_tokens, dim=1).mean(dim=1)  # [B, D]
            gamma_beta  = self.cat_to_num(h_cat_mean)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gate        = torch.sigmoid(self.gate_layer(h_cat_mean))
            scale_coef  = torch.tanh(self.gate_scale)        # 训练初期为 0

            num_tokens = [
                tok + scale_coef * gate * (tok * gamma + beta)
                for tok in num_tokens
            ]

            # 数值 Token 询问类别背景
            num_seq    = torch.stack(num_tokens, dim=1)      # [B, N_num, D]
            cat_seq    = torch.stack(cat_tokens, dim=1)      # [B, N_cat, D]
            num_seq    = self.cross_attn(num_seq, cat_seq)
            num_tokens = [num_seq[:, i] for i in range(num_seq.shape[1])]

        # ── 拼接所有列 Token ─────────────────────────────────────────
        all_tokens = num_tokens + cat_tokens
        if not all_tokens:
            raise ValueError("No tokens to process.")
        token_seq = torch.stack(all_tokens, dim=1)           # [B, N_cols, D]

        # ── 改进 2：注入时间步，使列间注意力感知噪声水平 ─────────────
        # 高噪声（t 大）时各列 Token 被同一时间步向量主导，关联权重趋均匀；
        # 低噪声（t 小）时列特异性更强，关联权重更集中，施加更强的联合约束。
        t_col     = self.time_to_col(t_emb)                  # [B, D]
        token_seq = token_seq + t_col.unsqueeze(1)           # broadcast → [B, N, D]

        # ── 改进 3：带对称正则的列间 Transformer ─────────────────────
        sym_loss = torch.zeros(1, device=x_in.device)
        if self.column_transformer is not None:
            token_seq, sym_loss = self.column_transformer(token_seq)
        self.last_sym_loss.copy_(sym_loss.detach())

        # ── 改进 1-a：列专属精炼 FFN ─────────────────────────────────
        # Linear 对 last dim 广播，token_seq [B, N, D] → h_cols [B, N, D]
        h_cols = self.col_refine(token_seq)

        # ── 全局流 ───────────────────────────────────────────────────
        h_global = token_seq.mean(dim=1)                     # [B, D]
        h_global = h_global + self.label_to_hidden(y_raw_emb)
        h_global = self.proj_fuse(h_global)
        for layer in self.layers:
            h_global = layer(h_global, emb)

        # ── 改进 1-b：双流解码 ───────────────────────────────────────
        # 每个输出头 = 列专属 Token（列级关联）+ 全局向量（扩散条件）
        if self.num_numerical > 0:
            out_n = torch.cat(
                [head(h_cols[:, i] + h_global)
                 for i, head in enumerate(self.out_num_heads)],
                dim=-1,
            )                                                # [B, N_num]
        else:
            out_n = torch.empty((B, 0), device=x_in.device)

        out_c_list = []
        for i, head in enumerate(self.out_cat):
            col_idx = self.num_numerical + i
            out_c_list.append(head(h_cols[:, col_idx] + h_global))

        out_c = (
            torch.cat(out_c_list, dim=-1)
            if out_c_list
            else torch.empty((B, 0), device=x_in.device)
        )

        output = torch.cat([out_n, out_c], dim=-1)

        if return_aux:
            return output, {'sym_loss': sym_loss}
        return output
