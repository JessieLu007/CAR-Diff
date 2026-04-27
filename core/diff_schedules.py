# core/diff_schedules.py
import numpy as np
import torch
import torch.nn as nn
from .utils import log_1_min_a

class DiffusionScheduleMixin(nn.Module):
    def _init_diffusion_buffers(self, num_timesteps: int):
        def _cosine_beta_schedule(timesteps, s=0.008):
            steps = timesteps + 1
            x = np.linspace(0, timesteps, steps)
            alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return np.clip(betas, 0, 0.999)

        betas = _cosine_beta_schedule(num_timesteps)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        betas_t = torch.tensor(betas, dtype=torch.float32)
        alphas_t = torch.tensor(alphas, dtype=torch.float32)
        alphas_cumprod_t = torch.tensor(alphas_cumprod, dtype=torch.float32)
        alphas_cumprod_prev_t = torch.tensor(alphas_cumprod_prev, dtype=torch.float32)

        self.register_buffer('betas', betas_t)
        self.register_buffer('alphas_cumprod', alphas_cumprod_t)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev_t)

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod_t))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod_t))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod_t))
        self.register_buffer('sqrt_recipml_alphas_cumprod', torch.sqrt(1. / alphas_cumprod_t - 1.))

        self.register_buffer(
            'posterior_variance',
            betas_t * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t)
        )
        self.register_buffer(
            'posterior_mean_coef1',
            betas_t * torch.sqrt(alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev_t) * torch.sqrt(alphas_t) / (1.0 - alphas_cumprod_t)
        )

        #
        self.register_buffer('sqrt_alphas', torch.sqrt(alphas_t)) # 单步sqrt(alpha_t)
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod_recip', # beta_t / sqrt(1-alpha_bar_t)
            betas_t / torch.sqrt(1. - alphas_cumprod_t)
        )

        log_alpha_t = torch.log(torch.clamp(alphas_t, min=1e-10))
        log_cumprod_alpha_t = torch.log(torch.clamp(alphas_cumprod_t, min=1e-10))

        self.register_buffer('log_alpha', log_alpha_t)
        self.register_buffer('log_1_min_alpha', log_1_min_a(log_alpha_t))
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha_t)
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_a(log_cumprod_alpha_t))
