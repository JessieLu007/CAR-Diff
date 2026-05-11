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
    batch_size = x_num.shape[0]

    model_num_out = model_out[:, :num_feats]
    loss_num = F.mse_loss(model_num_out, noise)

    # 一次性计算预测的x0, 供相关损失与分布正则共用
    sqrt_recip = extract(model.sqrt_recip_alphas_cumprod, t, x_num.shape)
    sqrt_recipml = extract(model.sqrt_recipml_alphas_cumprod, t, x_num.shape)
    x0_pred = sqrt_recip * x_num_t - sqrt_recipml * model_num_out
        
    loss_corr = 0.0
    if batch_size > 1 and num_feats > 1 and model.cfg.corr_weight > 0:
        threshold = int(model.num_timesteps * model.cfg.corr_t_ratio)
        valid_mask = t < threshold

        if valid_mask.sum() > 1:
            corr_real = distance_correlation(x_num[valid_mask])
            corr_pred = distance_correlation(x0_pred[valid_mask])
            loss_corr = F.mse_loss(corr_pred, corr_real)

    # ========== 新增：对预测latent x0的分布做一个极弱正则 ==========
    if getattr(model.cfg, 'use_normal_reg', True):
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

def compute_categorical_loss(model, x_cat, model_out, use_basic=False):
    num_feats = model.num_numerical_features
    model_cat_out = model_out[:, num_feats:]
    slices_for_classes = model.slices_for_classes
    true_marginals = model.true_marginals
    cfg = model.cfg

    if use_basic:
        loss_ce = 0.0
        loss_marginal = 0.0

        for i, ix in enumerate(slices_for_classes):
            logits = model_cat_out[:, ix]
            targets = x_cat[:, i]

            # 1. 主损失：per-sample cross-entropy
            ce = F.cross_entropy(logits, targets)
            loss_ce += ce

            # 2. 辅助：batch级别边缘分布对齐
            # 预测的边缘分布：对batch的softmax概率取平均
            log_probs = F.log_softmax(logits, dim=1)
            log_pred_marginal = torch.logsumexp(log_probs, dim=0) - np.log(logits.size(0))
            true_marginal = true_marginals[i].to(logits.device)

            # 使用 cross-entropy 形式：L_marg = - sum p_true * log p_pred
            # 避免 KL 的各种数值陷阱
            ce_marg = -torch.sum(true_marginal * log_pred_marginal)
            loss_marginal += ce_marg
            
        if len(slices_for_classes) > 0:
            loss_ce /= len(slices_for_classes)
            loss_marginal /= len(slices_for_classes)

        gamma = getattr(cfg, "gma_loss_weight", 0.5)
        return loss_ce + gamma * loss_marginal
    else:
        # 内部定义的微观焦点损失
        def focal_loss(logits, targets):
            ce = F.cross_entropy(logits, targets, reduction='none')
            # 修复：限制pt最大值为0.9999， 防止(1-pt)变成0导致求导Infinity
            pt = torch.clamp(torch.exp(-ce), max=0.9999)
                
            if cfg.use_adaptive_focal:
                with torch.no_grad():
                    probs = F.softmax(logits, dim=1).mean(dim=0)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                    # 严格保证 dtype 为 float32，避免 Double 泄漏
                    max_entropy = torch.log(torch.tensor(len(probs), dtype=torch.float32, device=logits.device))
                    norm_entropy = entropy / (max_entropy + 1e-8)
                    gamma = cfg.gamma_base * (1.0 - norm_entropy)
            else:
                gamma = cfg.gamma_base
            return ((1 - pt) ** gamma * ce).mean()

        loss_local = 0.0
        loss_global = 0.0 # 新增宏观对齐损失

        for i, ix in enumerate(slices_for_classes):
            logits = model_cat_out[:, ix]
            true_marginal = true_marginals[i].to(logits.device)

            # 找到该列中频率最低的类别的占比
            min_freq = true_marginal.min()

            # 1. 计算微观损失
            loss_local += focal_loss(logits, x_cat[:, i])

            # 2. 计算宏观KL对齐损失
            # 当前batch的预测宏观分布
            log_probs = F.log_softmax(logits, dim=1)
            log_pred_marginal = torch.logsumexp(log_probs, dim=0) - np.log(logits.size(0))
            true_marginal = true_marginals[i].to(logits.device)

            # KLDiv: L = target * (log(target) - log(pred))
            kl_div = F.kl_div(
                log_pred_marginal,
                true_marginal,
                reduction='batchmean',
                log_target=False
            )
            loss_global += kl_div
            
        if len(slices_for_classes) > 0:
            loss_local /= len(slices_for_classes)
            loss_global /= len(slices_for_classes)
        return loss_local + cfg.gma_loss_weight * loss_global