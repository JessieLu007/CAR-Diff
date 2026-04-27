# core/denoise_model.py
import torch
import torch.nn as nn
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:,:1])], dim=-1)
    return embedding

class MLPResidualBlock(nn.Module):
    """标准的残差 MLP 块，对表格数据极其稳定"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            #nn.LayerNorm(dim), # 增加LayerNorm，平衡数值与类别的梯度
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.ff(x)

class TabMLPDenoise(nn.Module):
    def __init__(
            self, 
            info, 
            num_classes_cond=2, 
            dim_t=128,          
            dim_hidden=256,     # MLP 隐藏层维度，256足够应付 Framingham
            num_layers=3,       # 3层残差网络
            dropout=0.1
            ):
        super().__init__()
        self.dim_t = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes = info['num_classes']
        self.in_dim = self.num_numerical + sum(self.num_classes)

        # 1. 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        # 2. 条件嵌入 (保留 +1，用于 CFG 的 Null Token)
        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # 3. 输入映射层 (把 输入特征 + 时间/条件特征 融合)
        self.proj_in = nn.Sequential(
            nn.Linear(self.in_dim + dim_t, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden)
        )

        # 4. 残差 MLP 骨架 (极速且稳定)
        self.layers = nn.ModuleList([
            MLPResidualBlock(dim_hidden, dropout) for _ in range(num_layers)
        ])

        # 5. 输出预测头
        if self.num_numerical > 0:
            self.out_num = nn.Linear(dim_hidden, self.num_numerical)
        else:
            self.out_num = None
            
        self.out_cat = nn.ModuleList([
            nn.Linear(dim_hidden, c) for c in self.num_classes
        ])

    def forward(self, x_in, timesteps, y):
        B = x_in.shape[0]
        
        # --- 1. 时间与条件 ---
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_emb = self.label_emb(y.squeeze().long())
        emb = t_emb + y_emb # [B, dim_t]

        # --- 2. 特征融合 ---
        # 直接把特征和嵌入拼接起来送进 MLP
        h = torch.cat([x_in, emb], dim=-1)
        h = self.proj_in(h)

        # --- 3. 前向传播 ---
        for layer in self.layers:
            h = layer(h)

        # --- 4. 解码输出 ---
        if self.out_num is not None:
            out_n = self.out_num(h)
        else:
            out_n = torch.empty((B, 0), device=x_in.device)
            
        out_c = []
        for head in self.out_cat:
            out_c.append(head(h))
            
        if len(out_c) > 0:
            out_c = torch.cat(out_c, dim=-1)
        else:
            out_c = torch.empty((B, 0), device=x_in.device)

        return torch.cat([out_n, out_c], dim=-1)