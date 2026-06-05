# core/denoise_model.py
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
# 编码模块
# ─────────────────────────────────────────────

class PeriodicEmbedding(nn.Module):
    """
    单列数值编码：sin/cos 展开 → 线性投影到 embed_dim。
    与旧版相比，这里只处理「单列」，方便逐列 Token 化。
    """
    def __init__(self, embed_dim: int, n_frequencies: int = 16):
        super().__init__()
        self.freqs = nn.Parameter(torch.randn(n_frequencies) * 0.02)
        self.proj  = nn.Linear(n_frequencies * 2, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B]  →  [B, embed_dim]
        x_f   = x.unsqueeze(-1) * self.freqs * 2 * math.pi   # [B, F]
        x_enc = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
        return self.proj(x_enc)


# ─────────────────────────────────────────────
# 主干模块
# ─────────────────────────────────────────────

class MLPResidualBlock(nn.Module):
    """AdaLN 风格的残差块（不变，保持与旧版一致）"""
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
        h             = self.norm(x)
        scale, shift  = self.cond_proj(emb).chunk(2, dim=-1)
        h             = h * (1 + scale) + shift
        return x + self.ff(h)


class CrossAttentionBlock(nn.Module):
    """
    数值 Token → Query，类别 Token 集合 → Key/Value。
    让每个数值列在生成时"询问"全部类别背景。
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        q_tokens: torch.Tensor,   # [B, Lq, D]  数值 Token
        kv_tokens: torch.Tensor,  # [B, Lk, D]  类别 Token
    ) -> torch.Tensor:
        """返回与 q_tokens 形状相同的增强张量"""
        q  = self.norm_q(q_tokens)
        kv = self.norm_kv(kv_tokens)
        out, _ = self.attn(q, kv, kv)
        return self.norm_out(q_tokens + out)           # 残差


# ─────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────

class TabMLPDenoise(nn.Module):
    """
    改进点：
      1. 可学习门控尺度（gate_scale），取代硬编码 0.3
      2. 逐列 Token 化：每一列独享独立编码器，拼成 [B, N_cols, D] 序列
      3. Cross-Attention：数值 Token Query 类别 Token 的语义背景
      4. Hybrid Transformer-MLP 主干：TransformerEncoderLayer 处理列间关联，
         再经 MLPResidualBlock 栈接收时间步 / 标签条件
    """

    def __init__(
        self,
        info,
        num_classes_cond: int = 2,
        dim_t: int = 128,
        dim_hidden: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        n_frequencies: int = 16,
        nhead_col: int = 4,       # 列间 Transformer 头数
        nhead_cross: int = 4,     # Cross-Attention 头数
    ):
        super().__init__()

        self.dim_t         = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes   = info['num_classes']          # list[int]
        true_marginals     = info.get('true_marginals', None)

        # ── 1. 逐列编码器 ────────────────────────────────────────────
        # 数值列：每列独立的 PeriodicEmbedding
        if self.num_numerical > 0:
            self.num_col_encoders = nn.ModuleList([
                PeriodicEmbedding(dim_hidden, n_frequencies)
                for _ in range(self.num_numerical)
            ])
        else:
            self.num_col_encoders = None

        # 类别列：每列独立的 Embedding → 线性投影到 dim_hidden
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

        # ── 2. 改进 1：可学习门控尺度 ────────────────────────────────
        # 初始化为 0 → tanh(0)=0 → 训练初期门控完全关闭
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.gate_layer = nn.Linear(dim_hidden, dim_hidden)
            nn.init.xavier_uniform_(self.gate_layer.weight)
            nn.init.constant_(self.gate_layer.bias, -1.0)

            self.cat_to_num   = nn.Linear(dim_hidden, dim_hidden * 2)
            nn.init.zeros_(self.cat_to_num.weight)
            nn.init.zeros_(self.cat_to_num.bias)

            # 可学习标量：控制 FiLM 调制的整体幅度
            #self.gate_scale = nn.Parameter(torch.zeros(1))  # ← 改进 1
            self.gate_scale = nn.Parameter(torch.full((1,), 0.1))

        # ── 3. 改进 3：Cross-Attention（数值 Query 类别 KV）────────────
        if self.num_numerical > 0 and len(self.num_classes) > 0:
            self.cross_attn = CrossAttentionBlock(dim_hidden, nhead_cross, dropout)
        else:
            self.cross_attn = None

        # ── 4. 改进 4：列间 Transformer（改进 2 的自注意力升级）─────────
        # 输入 [B, N_cols, D]，N_cols = num_numerical + len(num_classes)
        total_cols = self.num_numerical + len(self.num_classes)

        if self.num_numerical > 1:
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

        self.num_cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)
        self.cat_cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)

        self.fusion_proj = nn.Sequential(
            nn.Linear(dim_hidden * 2, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden),
        )
        
        '''if total_cols > 1:
            self.column_transformer = nn.TransformerEncoderLayer(
                d_model=dim_hidden,
                nhead=nhead_col,
                dim_feedforward=dim_hidden * 2,
                dropout=dropout,
                batch_first=True,
                norm_first=True,              # Pre-LN 更稳定
            )
        else:
            self.column_transformer = None'''

        # 添加 cls token (classification token) 
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim_hidden) * 0.02)

        # ── 5. 条件编码（时间步 + 类别标签）────────────────────────────
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t),
        )
        self.label_emb      = nn.Embedding(num_classes_cond + 1, dim_t)
        #self.label_to_hidden = nn.Linear(dim_t, dim_hidden)

        # ── 6. 聚合投影（Token 均值 → h）────────────────────────────────
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden),
        )

        # 1. 数值专属塔
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

        # ── 7. MLP 残差主干（接收条件 emb）──────────────────────────────
        '''self.layers = nn.ModuleList([
            MLPResidualBlock(dim_hidden, dim_t, dropout)
            for _ in range(num_layers)
        ])'''

        # ── 8. 输出头 ────────────────────────────────────────────────
        if self.num_numerical > 0:
            #self.out_num = nn.Linear(dim_hidden, self.num_numerical)
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

    # ────────────────────────────────────────────────────────────────
    def _apply_prior_initialization(self, true_marginals):
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                prior     = true_marginals[i]
                log_prior = torch.log(prior + 1e-8)
                head[-1].bias.copy_(log_prior)
                head[-1].weight.data.mul_(0.1)

    # ────────────────────────────────────────────────────────────────
    def forward(self, x_in: torch.Tensor, timesteps: torch.Tensor, y: torch.Tensor):
        B = x_in.shape[0]

        # ── 条件向量 ────────────────────────────────────────────────
        t_emb     = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_raw_emb = self.label_emb(y.view(-1).long())
        emb       = t_emb + y_raw_emb          # [B, dim_t]，用于 AdaLN

        # ── 拆分输入 ─────────────────────────────────────────────────
        x_num = x_in[:, :self.num_numerical]                   # [B, N_num]
        x_cat_prob = x_in[:, self.num_numerical:].long()
        #x_cat = x_in[:, self.num_numerical:].long()            # [B, N_cat]
        

        # ── 改进 2：逐列 Token 化 ─────────────────────────────────────
        num_tokens, cat_tokens = [], []

        if self.num_col_encoders is not None:
            for i, enc in enumerate(self.num_col_encoders):
                num_tokens.append(enc(x_num[:, i]))            # [B, D]

        start = 0
        for emb_layer, proj, n_class in zip(
            self.cat_embed_layers,
            self.cat_col_projs,
            self.num_classes):

            end = start + n_class
            prob = x_cat_prob[:, start:end]
            
            assert prob.shape[1] == n_class

            emb_weight = emb_layer.weight
            tok = prob.float() @ emb_weight
            tok = proj(tok)

            cat_tokens.append(
                self.cat_dropout(tok)
            )
            start = end

        # ── 改进 3：Cross-Attention（数值 Query 类别 KV）────────────────
        # 在 Token 化完成后、拼接前对数值 Token 做跨模态增强
        if self.cross_attn is not None and num_tokens and cat_tokens:
            # 聚合类别侧均值，供 FiLM 调制使用
            h_cat_mean = torch.stack(cat_tokens, dim=1).mean(dim=1)  # [B, D]

            # FiLM 调制（改进 1：可学习 gate_scale）
            gamma_beta = self.cat_to_num(h_cat_mean)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gate        = torch.sigmoid(self.gate_layer(h_cat_mean))
            scale_coef  = torch.tanh(self.gate_scale)          # ← 初始 0，自适应增长

            # 对每个数值 Token 施加 FiLM 调制
            num_tokens = [
                tok + scale_coef * gate * (tok * gamma + beta)
                for tok in num_tokens
            ]

            # 模态内自注意力
            num_seq = torch.stack(num_tokens, dim=1)           # [B, N_num, D]
            cat_seq = torch.stack(cat_tokens, dim=1)           # [B, N_cat, D]

            num_cls = self.num_cls_token.expand(B, -1, -1) # [B, 1, D]
            cat_cls = self.cat_cls_token.expand(B, -1, -1) # [B, 1, D]
            num_seq = torch.cat([num_cls, num_seq], dim=1) # [B, 1+N_num, D]
            cat_seq = torch.cat([cat_cls, cat_seq], dim=1) # [B, 1+N_cat, D]

            if self.num_transformer is not None:
                num_seq = self.num_transformer(num_seq) # num-num 关联
            if self.cat_transformer is not None:
                cat_seq = self.cat_transformer(cat_seq) # cat-cat关联

            # 分离提取cls和 列 token
            num_cls_h = num_seq[:, 0] # [B, D] 数值测聚合
            num_col_seq = num_seq[:, 1:] # [B, N_num, D]
            cat_cls_h   = cat_seq[:, 0]                     # [B, D] 类别侧聚合
            cat_col_seq = cat_seq[:, 1:]                    # [B, N_cat, D]

            # ── FiLM（来源改为 cat_cls_h，比 mean 更有表达力）───────────────
            if self.cross_attn is not None:
                gamma_beta = self.cat_to_num(cat_cls_h)     # cat CLS → 调制系数
                gamma, beta = gamma_beta.chunk(2, dim=-1)
                gate       = torch.sigmoid(self.gate_layer(cat_cls_h))
                scale_coef = torch.tanh(self.gate_scale)
            
                num_col_seq = num_col_seq + scale_coef * gate.unsqueeze(1) * (
                    num_col_seq * gamma.unsqueeze(1) + beta.unsqueeze(1)
                )
            
                # cross_attn：num CLS 也参与 query，让 CLS 直接 attend 类别列
                num_query = torch.cat([num_cls_h.unsqueeze(1), num_col_seq], dim=1)
                num_query = self.cross_attn(num_query, cat_col_seq)
                num_cls_h = num_query[:, 0]                 # 更新后的数值侧 CLS

            #num_seq = self.cross_attn(num_seq, cat_seq)        # [B, N_num, D]
            #num_tokens = [num_seq[:, i] for i in range(num_seq.shape[1])]

        # ── 改进 4：拼接所有列 Token → 列间 Transformer ─────────────────
        '''all_tokens = num_tokens + cat_tokens                   # list of [B, D]

        if all_tokens:
            token_seq = torch.stack(all_tokens, dim=1)        # [B, N_cols, D]
            cls_token = self.cls_token.expand(B, -1, -1)
            token_seq = torch.cat([cls_token, token_seq], dim=1)
        else:
            raise ValueError("No tokens to process.")

        if self.column_transformer is not None:
            token_seq = self.column_transformer(token_seq)    # [B, N_cols, D]

        # Token 均值聚合 → 单一隐向量
        h = token_seq[:,0]'''    

        '''branch_reps = []
        if num_tokens:
            num_seq = torch.stack(num_tokens, dim=1) # [B, N_num, D]
            branch_reps.append(num_seq.mean(dim=1))  # [B, D]
        if cat_tokens:
            cat_seq_final = torch.stack(cat_tokens, dim=1) # [B, N_cat, D]
            branch_reps.append(cat_seq_final.mean(dim=1)) # [B, D]

        if not branch_reps:
            raise ValueError("No tokens to process.")'''

        h = self.fusion_proj(torch.cat([num_cls_h, cat_cls_h], dim=-1)) # [B, D]

        # 注入标签信息
        #h = h + self.label_to_hidden(y_raw_emb)
        h_shared = self.proj_fuse(h)

        # 1. 数值分支前向
        out_n = torch.empty((B, 0), device=x_in.device)
        if self.num_backbone is not None:
            h_num = h_shared.clone()
            for layer in self.num_backbone:
                h_num = layer(h_num, emb)
            out_n = self.out_num(h_num)

        # 2. 分类分支前向
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