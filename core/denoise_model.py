# core/denoise_model.py
import torch
import torch.nn as nn
import math


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

class PeriodicEmbedding(nn.Module):
    """数值特征编码：先进行周期性展开，再映射到隐藏维度"""
    def __init__(self, num_features, embed_dim, n_frequencies=16):
        super().__init__()
        self.n_frequencies = n_frequencies
        # 可学习的频率参数
        self.freqs = nn.Parameter(torch.randn(num_features, n_frequencies)*0.02)
        self.proj = nn.Linear(num_features * n_frequencies * 2, embed_dim)

    def forward(self, x):
        # x: [B, num_features]
        # 扩展到 [B, num_features, n_frequencies]
        x_f = x.unsqueeze(-1) * self.freqs.unsqueeze(0) * 2 * math.pi
        x_enc = torch.cat([torch.sin(x_f), torch.cos(x_f)], dim=-1)
        # 展平后投影
        x_enc = x_enc.reshape(x.shape[0], -1)
        return self.proj(x_enc)

class MLPResidualBlock(nn.Module):
    """Residual block with conditioning injection"""
    def __init__(self, dim, cond_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.cond_proj = nn.Linear(cond_dim, dim*2)

    def forward(self, x, emb):
        # ⭐ 关键：先norm再加cond（稳定很多）
        h = self.norm(x)
        # AdaLN风格：用条件预测scale和shift
        cond = self.cond_proj(emb)
        scale, shift = cond.chunk(2, dim=-1)
        h = h * (1 + scale) + shift #乘法门控初始化接近恒等变换

        h = self.ff(h)
        return x + h


class TabMLPDenoise(nn.Module):
    def __init__(
        self,
        info,
        num_classes_cond=2,
        dim_t=128,
        dim_hidden=256, # 这里的dim_hidden建议作为每个特征embedding的维度
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.dim_t = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes = info['num_classes'] #list [2, 4, 5]
        true_marginals = info.get('true_marginals', None) #接收先验

        # ---------- 1. 数值特征编码 ----------
        if self.num_numerical > 0:
            self.proj_num = PeriodicEmbedding(self.num_numerical, dim_hidden)
        else:
            self.proj_num = None

        # ---------- 2. 类别特征编码 (独立 Embedding) ----------
        self.cat_dropout = nn.Dropout(0.2) # 设置起点较高的dropout
        if len(self.num_classes) > 0:
            # 为每个分类列创建独立的Embedding空间
            self.cat_embed_layers = nn.ModuleList([
                nn.Embedding(c, min(16, c * 2)) for c in self.num_classes
            ])
            cat_total_emb_dim = sum([min(16, c * 2) for c in self.num_classes])

            self.proj_cat = nn.Sequential(
                nn.Linear(cat_total_emb_dim, dim_hidden),
                nn.SiLU(),
                nn.LayerNorm(dim_hidden)
            )
            # FiLM 调制层：通过类别信息生成数值特征的缩放和偏移
            if self.num_numerical > 0:
                self.cat_to_num = nn.Linear(dim_hidden, dim_hidden * 2)
                # 初始化为0， 使得初期融合接近于identity(h = h_num + 0)
                nn.init.zeros_(self.cat_to_num.weight)
                nn.init.zeros_(self.cat_to_num.bias)

        else:
            self.cat_embed_layers = None
            self.proj_cat = None

        #  ---------- 3. 条件编码 (时间 + 标签) ----------
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )
        # +1 用于处理无条件引导 (unconditional guidance)
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)
        # 额外的标签直接注入层
        self.label_to_hidden = nn.Linear(dim_t, dim_hidden)
        self.gate_layer = nn.Linear(
            in_features=dim_hidden,
            out_features=dim_hidden
        )
        self.gate_scale = nn.Parameter(torch.zeros(1)) # 初始化为0
        # 初始化：让模型初始化时倾向于“关闭”门控
        nn.init.xavier_uniform_(self.gate_layer.weight)
        nn.init.constant_(self.gate_layer.bias, -1.0)

        # 新增轻量级Self-Attention模块
        # 在MLP块之间插入一个处理关联性的Attention层
        self.attn_layer = nn.MultiheadAttention(
            embed_dim=dim_hidden,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(dim_hidden)

        # ---------- 4. 融合与主干 ----------
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden)
        )

        self.layers = nn.ModuleList([
            MLPResidualBlock(dim_hidden, dim_t, dropout)
            for _ in range(num_layers)
        ])

        # ---------- 5. 输出头 ----------
        if self.num_numerical > 0:
            self.out_num = nn.Linear(dim_hidden, self.num_numerical)

        self.out_cat = nn.ModuleList([
            nn.Linear(dim_hidden, c) for c in self.num_classes
        ])

        # 使用真实边缘分布初始化Bias
        if true_marginals is not None:
            self._apply_prior_initialization(true_marginals)
    
    def _apply_prior_initialization(self, true_marginals):
        """将分类输出层的偏置初始化为训练集分布的对数"""
        with torch.no_grad():
            for i, head in enumerate(self.out_cat):
                #获取该列的真实分布(p1, p2, ...)
                prior = true_marginals[i].cpu()
                # 将其转换为logits形式作为bias
                log_prior = torch.log(prior + 1e-8)
                head.bias.copy_(log_prior)

                # 为防止初始化权重干扰Bias, 把该层的weight缩小
                head.weight.data.mul_(0.1)


    def forward(self, x_in, timesteps, y):
        B = x_in.shape[0]

        # ---------- 1. embedding ----------
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_raw_emb = self.label_emb(y.squeeze().long())
        emb = t_emb + y_raw_emb   # 用于AdaLN的条件向量 [B, dim_t]

        # ---------- 2. split ----------
        x_num = x_in[:, :self.num_numerical]
        x_cat = x_in[:, self.num_numerical:].long() # 确保是整数型索引

        # ---------- 3. encode ----------
        # A.数值编码
        h_num = self.proj_num(x_num) if self.proj_num is not None else None

        # B.类别编码（遍历每一列）
        h_cat = None
        if self.cat_embed_layers is not None:
            cat_list = []
            for i, emb_layer in enumerate(self.cat_embed_layers):
                cat_list.append(emb_layer(x_cat[:, i]))
            h_cat = torch.cat(cat_list, dim=-1)
            h_cat = self.proj_cat(h_cat)

        # --- 4. 融合 (FiLM 机制) ---
        if h_num is not None and h_cat is not None:
            # 使用类别信息调制数值特征
            h_cat = self.cat_dropout(h_cat) # 在进行FiLM调制前丢弃部分类别信号
            gamma_beta = self.cat_to_num(h_cat)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            # 新增门控机制：让模型自主决定吸收多少类别信息
            gate = torch.sigmoid(self.gate_layer(h_cat))
            h = h_num + torch.tanh(self.gate_scale) * gate * (h_num * gamma + beta)
        elif h_num is not None:
            h = h_num
        else:
            h = h_cat

        # 注入标签信息到隐藏层
        h = h + self.label_to_hidden(y_raw_emb)
        h = self.proj_fuse(h)

        # --- [新增]Self-Attention增强列间关联 ---
        #将 h 视为【B,1， dim_hidden】的序列进行self-attention
        # 这里L=1主要是为了建模特征内部各component的交互，
        # 如果想建模真正的列间关联，需要将Embedding分开输入，但这会大幅增加计算量
        # 当下只做特征内的自注意力
        h_attn = h.unsqueeze(1) # [B, 1, D]
        h_attn, _ = self.attn_layer(h_attn, h_attn, h_attn)
        h = self.attn_norm(h + h_attn.squeeze(1))

        # --- 5. 主干网络 (Residual Stack) ---
        for layer in self.layers:
            h = layer(h, emb)

        # --- 6. 解码输出 ---
        # 数值输出
        if self.num_numerical > 0:
            out_n = self.out_num(h)
        else:
            out_n = torch.empty((B, 0), device=x_in.device)

        # 类别输出（每个字段独立预测logits)
        out_c_list = []
        for head in self.out_cat:
            out_c_list.append(head(h))

        if out_c_list:
            out_c = torch.cat(out_c_list, dim=-1)
        else:
            out_c = torch.empty((B, 0), device=x_in.device)

        return torch.cat([out_n, out_c], dim=-1)