import torch.nn as nn
import torch

class LightweightVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()

        hidden = max(input_dim * 2, 32)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden), # 小数量下用LayerNorm比BN稳定
            nn.GELU(),
        )

        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
    

def compute_safe_latent_dim(num_numerical, num_cat_cols, n_samples):
    """
    小数据集的 latent dim 硬约束：
    latent_dim << n_samples, 否则VAE 退化为查找表
    """
    feature_based = num_numerical + num_cat_cols
    data_based = max(4, n_samples // 20)
    return min(feature_based, data_based)