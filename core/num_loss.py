# core/diff_losses.py
import torch
import torch.nn.functional as F
from .utils import extract
import numpy as np

def distance_correlation(x):
    b, d = x.shape
    # change b<=1 to b<=4, 提高最小样本数限制，防止极小batch计算协方差崩溃
    if b <= 4: return torch.zeros((d, d), device=x.device)

    x_reshaped = x.t().unsqueeze(-1)
    a = torch.abs(x_reshaped - x_reshaped.transpose(1, 2))

    A = a - a.mean(dim=2, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(1, 2), keepdim=True)
    A_flat = A.view(d, -1)
    dCov_sq = (A_flat @ A_flat.t()) / (b * b)

    dVar_sq = torch.diag(dCov_sq)
    dVar_prod = torch.sqrt(torch.clamp(dVar_sq.unsqueeze(0) * dVar_sq.unsqueeze(1), min=1e-8) + 1e-8)
    return torch.sqrt(torch.clamp(dCov_sq / dVar_prod, min=1e-8))

def compute_numerical_loss(model, x_num, x_num_t, model_out, t, noise):
    num_feats = model.num_numerical_features
    prediction_type = model.model_cfg.prediction_type

    if prediction_type == 'x0':
        x0_pred = model_out[:, :num_feats]

        # 主损失： 直接拟合 x0
        loss_num = F.mse_loss(x0_pred, x_num)
        loss_corr = torch.tensor(0.0, device=x_num.device)

        return loss_num, loss_corr
    elif prediction_type == 'eps':
        batch_size = x_num.shape[0]

        model_num_out = model_out[:, :num_feats]
        loss_num = F.mse_loss(model_num_out, noise)

        # 一次性计算预测的x0, 供相关损失与分布正则共用
        sqrt_recip = extract(model.sqrt_recip_alphas_cumprod, t, x_num.shape)
        sqrt_recipml = extract(model.sqrt_recipml_alphas_cumprod, t, x_num.shape)
        x0_pred = sqrt_recip * x_num_t - sqrt_recipml * model_num_out
            
        loss_corr = 0.0
        if batch_size > 1 and num_feats > 1 and model.model_cfg.corr_weight > 0:
            threshold = int(model.model_cfg.num_timesteps * model.model_cfg.corr_t_ratio)
            valid_mask = t < threshold

            if valid_mask.sum() > 1:
                corr_real = distance_correlation(x_num[valid_mask])
                corr_pred = distance_correlation(x0_pred[valid_mask])
                loss_corr = F.mse_loss(corr_pred, corr_real)

        # ========== 对预测latent x0的分布做一个极弱正则 ==========
        if getattr(model.model_cfg, 'use_normal_reg', False):
            # 只在中等噪声区间做一点约束，避免极端阶段干扰
            t_float = t.float() / (model.num_timesteps - 1)
            mid_mask = (t_float >= 0.2) & (t_float <= 0.5)

            if mid_mask.any():
                x0_mid = x0_pred[mid_mask] # [B_mid, D]
                mean_mid = x0_mid.mean(dim=0) # [D]
                var_mid = x0_mid.var(dim=0, unbiased=False) #[D]

                # 希望整体接近N(0, 1): 均值接近0，方差接近1
                mean_reg = (mean_mid ** 2).mean() # E[μ^2]
                var_reg = ((var_mid - 1.0)**2).mean() # E[(σ^2-1)^2]

                reg_weight = 1e-3 # 先用很小的权重
                loss_reg = reg_weight * (mean_reg + var_reg)
                loss_num = loss_num + loss_reg
                    
        return loss_num, loss_corr