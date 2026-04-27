# core/diff_sampling.py
import torch
import torch.nn.functional as F
from .utils import extract, ohe_to_categories
from configs.experiment_configs import SamplingConfig


def sample_core(model, num_samples: int, y_cond: int, cfg: SamplingConfig):
    """
    DDPM 逆扩散采样主循环。
    支持三种 task：full_data / minority_uncond / minority_cond。
    支持两种 objective：fidelity（无CFG）/ clf_boost（三路CFG）。
    """
    b = num_samples
    device = model.device

    # ------------------------------------------------------------------ #
    # 1. 采样超参（由 objective 决定）
    # ------------------------------------------------------------------ #
    if model.train_cfg.objective == 'fidelity':
        temperature     = cfg.temperature_fedility
        cfg_scale       = cfg.cfg_scale_fedility
        repulse_weight  = cfg.repulse_weight_fedility
    else:  # clf_boost
        temperature     = cfg.temperature_boost
        cfg_scale       = cfg.cfg_scale_boost
        repulse_weight  = cfg.repulse_weight_boost

    # full_data 任务一律用 null 条件，不启用 CFG
    if model.train_cfg.task == 'full_data':
        y_cond = model.cfg.null_class_idx

    y_pos = torch.full((b,), y_cond,                    device=device, dtype=torch.long)
    y_unc = torch.full((b,), model.cfg.null_class_idx,  device=device, dtype=torch.long)
    y_neg = torch.full((b,), 1 - y_cond,                device=device, dtype=torch.long)

    use_cfg = (model.train_cfg.task == 'minority_cond'
               and model.train_cfg.objective == 'clf_boost')

    # ------------------------------------------------------------------ #
    # 2. 初始化采样起点
    # ------------------------------------------------------------------ #
    # 数值特征：标准正态
    x_num = torch.randn(b, model.num_numerical_features, device=device)

    # 离散特征：均匀分布（对应训练时 t=T 的完全加噪状态）
    if len(model.num_classes) > 0:
        log_x_cat = torch.log(
            torch.ones(b, sum(model.num_classes), device=device)
            / model.num_classes_expanded
        )
    else:
        log_x_cat = torch.empty(b, 0, device=device)

    posterior_threshold = int(model.num_timesteps * model.cfg.posterior_switch_ratio)

    # ------------------------------------------------------------------ #
    # 3. 逆扩散主循环
    # ------------------------------------------------------------------ #
    for i in reversed(range(model.num_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)

        # ---------- 3a. 去噪网络前向 ----------
        x_in = torch.cat([x_num, log_x_cat.exp()], dim=1)

        if not use_cfg:
            model_out = model._denoise_fn(x_in, t, y_pos)
        else:
            x_in3 = torch.cat([x_in, x_in, x_in], dim=0)
            t3    = torch.cat([t,    t,    t   ], dim=0)
            y3    = torch.cat([y_unc, y_pos, y_neg], dim=0)
            out3  = model._denoise_fn(x_in3, t3, y3)
            out_unc, out_pos, out_neg = torch.chunk(out3, 3, dim=0)
            model_out = (out_unc
                         + cfg_scale      * (out_pos - out_unc)
                         - repulse_weight * (out_neg - out_unc))

        eps_pred     = model_out[:, :model.num_numerical_features]
        model_cat_out = model_out[:, model.num_numerical_features:]

        # ---------- 3b. 数值特征更新（DDPM 后验） ----------
        x0_pred = (extract(model.sqrt_recip_alphas_cumprod,   t, x_num.shape) * x_num
                 - extract(model.sqrt_recipml_alphas_cumprod, t, x_num.shape) * eps_pred)
        x0_pred = torch.clamp(x0_pred, model.cfg.x0_clamp_min, model.cfg.x0_clamp_max)

        mean = (extract(model.posterior_mean_coef1, t, x_num.shape) * x0_pred
              + extract(model.posterior_mean_coef2, t, x_num.shape) * x_num)

        if i > 0:
            noise   = torch.randn_like(x_num) * temperature
            log_var = torch.log(extract(model.posterior_variance, t, x_num.shape).clamp(min=1e-20))
            x_num   = mean + torch.exp(0.5 * log_var) * noise
        else:
            x_num = mean  # 最后一步不加噪

        # ---------- 3c. 离散特征更新（multinomial diffusion 后验） ----------
        new_log_x_cat = torch.empty_like(model_cat_out)

        for ix in model.slices_for_classes:
            logits = model_cat_out[:, ix] / temperature
            p_x0   = F.softmax(logits, dim=1)
            K      = logits.shape[1]

            # ---- 新：所有 t 都只用 p_x0，不再区分 i>posterior_threshold ----
            probs = p_x0.clamp(min=1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)

            idx    = torch.multinomial(probs, 1).squeeze(-1)
            onehot = F.one_hot(idx, num_classes=K).float()

            if i > 0:
                # 与训练前向一致的“软化”：
                safe_prev  = torch.clamp(t - 1, min=0)
                a_bar_prev = extract(model.alphas_cumprod_prev, safe_prev, onehot.shape)
                soft_prob  = a_bar_prev * onehot + (1.0 - a_bar_prev) / K
                new_log_x_cat[:, ix] = torch.log(soft_prob.clamp(min=1e-8))
            else:
                # 最后一步 hard one-hot
                new_log_x_cat[:, ix] = torch.log(onehot + 1e-7)
            
        log_x_cat = new_log_x_cat

    # ------------------------------------------------------------------ #
    # 4. 解码输出
    # ------------------------------------------------------------------ #
    x_cat = ohe_to_categories(log_x_cat.exp(), model.num_classes)
    return torch.cat([x_num, x_cat], dim=1)