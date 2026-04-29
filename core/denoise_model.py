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
        dim_hidden=256,
        num_layers=3,
        dropout=0.1
    ):
        super().__init__()

        self.dim_t = dim_t
        self.num_numerical = info['num_numerical']
        self.num_classes = info['num_classes']

        total_cat_dim = sum(self.num_classes)

        # ---------- 1. num / cat 分开编码 ----------
        if self.num_numerical > 0:
            self.proj_num = nn.Sequential(
                nn.Linear(self.num_numerical, dim_hidden),
                nn.SiLU(),
                nn.LayerNorm(dim_hidden)
            )
        else:
            self.proj_num = None

        if total_cat_dim > 0:
            self.proj_cat = nn.Sequential(
                nn.Linear(total_cat_dim, dim_hidden),
                nn.SiLU(),
                nn.LayerNorm(dim_hidden)
            )
        else:
            self.proj_cat = None

        # ---------- 2. 融合层（不再拼 emb） ----------
        self.proj_fuse = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.SiLU(),
            nn.LayerNorm(dim_hidden)
        )

        # ---------- 3. 时间 / 条件 embedding ----------
        self.time_embed = nn.Sequential(
            nn.Linear(dim_t, dim_t),
            nn.SiLU(),
            nn.Linear(dim_t, dim_t)
        )

        self.label_emb = nn.Embedding(num_classes_cond + 1, dim_t)

        # ---------- 4. residual 主干 ----------
        self.layers = nn.ModuleList([
            MLPResidualBlock(dim_hidden, dim_t, dropout)
            for _ in range(num_layers)
        ])

        # ---------- 5. 输出头 ----------
        if self.num_numerical > 0:
            self.out_num = nn.Linear(dim_hidden, self.num_numerical)
        else:
            self.out_num = None

        self.out_cat = nn.ModuleList([
            nn.Linear(dim_hidden, c) for c in self.num_classes
        ])

    def forward(self, x_in, timesteps, y):
        B = x_in.shape[0]

        # ---------- 1. embedding ----------
        t_emb = self.time_embed(timestep_embedding(timesteps, self.dim_t))
        y_emb = self.label_emb(y.squeeze().long())
        emb = t_emb + y_emb   # [B, dim_t]

        # ---------- 2. split ----------
        x_num = x_in[:, :self.num_numerical]
        x_cat = x_in[:, self.num_numerical:]

        # ---------- 3. encode ----------
        h_num = self.proj_num(x_num) if self.proj_num is not None else None
        h_cat = self.proj_cat(x_cat) if self.proj_cat is not None else None

        if h_num is not None and h_cat is not None:
            h = h_num + h_cat
        elif h_num is not None:
            h = h_num
        elif h_cat is not None:
            h = h_cat
        else:
            raise ValueError("Both numerical and categorical features are missing!")

        # ---------- 4. fuse ----------
        h = self.proj_fuse(h)

        # ---------- 5. residual stack ----------
        for layer in self.layers:
            h = layer(h, emb)

        # ---------- 6. decode ----------
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