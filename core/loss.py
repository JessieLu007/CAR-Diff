# core/loss.py
import torch
import torch.nn.functional as F
from .utils import extract


def compute_categorical_loss(model, x_cat: torch.Tensor, model_out: torch.Tensor) -> torch.Tensor:
    """各分类列交叉熵的平均。"""
    logits_all = model_out[:, model.num_numerical_features:]
    slices = model.slices_for_classes

    loss = torch.tensor(0.0, device=model_out.device)
    if not slices:
        return loss

    for i, ix in enumerate(slices):
        loss = loss + F.cross_entropy(logits_all[:, ix], x_cat[:, i])
    return loss / len(slices)


def compute_numerical_loss(model, x_num, x_num_t, model_out, t, noise):
    num_feats = model.num_numerical_features
    prediction_type = model.model_cfg.prediction_type

    if prediction_type == 'x0':
        x0_pred = model_out[:, :num_feats]
        loss_num = F.mse_loss(x0_pred, x_num)
        return loss_num
    elif prediction_type == 'eps':

        model_num_out = model_out[:, :num_feats]
        loss_num = F.mse_loss(model_num_out, noise)

        # 一次性计算预测的x0, 供相关损失与分布正则共用
        sqrt_recip = extract(model.sqrt_recip_alphas_cumprod, t, x_num.shape)
        sqrt_recipml = extract(model.sqrt_recipml_alphas_cumprod, t, x_num.shape)
        x0_pred = sqrt_recip * x_num_t - sqrt_recipml * model_num_out

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

        return loss_num