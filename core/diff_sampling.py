# core/diff_sampling.py

import torch
import torch.nn.functional as F

from .utils import extract, ohe_to_categories
from configs.experiment_configs import SamplingConfig


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def gumbel_sample(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Gumbel-max 采样，等价于 torch.multinomial(softmax(logits), 1)。
    相比 multinomial，在 CPU / GPU 间有更好的数值一致性。

    参数：
        logits      : [B, K]，原始未缩放的类别 logit
        temperature : 采样温度，>1 更随机，<1 更确定，由外部统一传入
    返回：
        [B]，每行采样得到的类别 index
    """
    gumbel_noise = -torch.log(
        -torch.log(torch.rand_like(logits) + 1e-8) + 1e-8
    )
    return (logits / temperature + gumbel_noise).argmax(dim=-1)


# ═══════════════════════════════════════════════════════════════════
# 采样主函数
# ═══════════════════════════════════════════════════════════════════

def sample_core(
    model,
    num_samples : int,
    y_cond      : int,
) -> torch.Tensor:
    """
    DDPM 逆扩散采样主循环。

    支持三种 task：
        full_data       —— 联合分布生成，y 条件固定为 null
        minority_uncond —— 无条件少数类生成
        minority_cond   —— CFG 引导的少数类生成

    支持两种 objective：
        fidelity    —— 无 CFG，追求分布保真度
        clf_boost   —— 三路 CFG（pos / unc / neg），追求分类器提升

    参数：
        model       : MedAGDiffusion 实例，含 train_cfg / cfg / _denoise_fn
        num_samples : 生成样本数
        y_cond      : 目标类别 index（full_data 模式下忽略）
        cfg         : SamplingConfig，含温度、CFG 权重等超参

    返回：
        torch.Tensor, shape = (num_samples, num_numerical + num_cat_cols)
        数值列在前，类别列（硬 index）在后
    """
    print(">>> ENTER sample_core")
    b      = num_samples
    device = model.device
    sampling_cfg = model.sampling_cfg

    # ── 1. 采样超参（由 objective 决定）─────────────────────────────
    if model.train_cfg.objective == 'fidelity':
        temperature_num = 1.0
        temperature_cat = getattr(sampling_cfg, 'cat_temperature_cap', 0.8)
        cfg_scale       = sampling_cfg.cfg_scale_fidelity
        repulse_weight  = sampling_cfg.repulse_weight_fidelity
    else:   # clf_boost
        temperature_num = sampling_cfg.temperature_boost
        temperature_cat = sampling_cfg.temperature_boost
        cfg_scale       = sampling_cfg.cfg_scale_boost
        repulse_weight  = sampling_cfg.repulse_weight_boost

    # full_data 任务一律使用 null 条件，不启用 CFG
    if model.train_cfg.task == 'full_data':
        y_cond = model.model_cfg.null_class_idx
    
    y_pos = torch.full((b,), y_cond,                   device=device, dtype=torch.long)
    y_unc = torch.full((b,), model.model_cfg.null_class_idx, device=device, dtype=torch.long)
    y_neg = torch.full((b,), 1 - y_cond,               device=device, dtype=torch.long)

    use_cfg = (
        model.train_cfg.task      == 'minority_cond'
        and model.train_cfg.objective == 'clf_boost'
    )

    # ── 2. 初始化采样起点（对应训练时 t = T 的完全加噪状态）─────────
    x_num = torch.randn(b, model.num_numerical_features, device=device)

    if len(model.num_classes) > 0:
        # 类别特征：均匀分布的 log 概率（最大熵状态）
        log_x_cat = torch.log(
            torch.ones(b, sum(model.num_classes), device=device)
            / model.num_classes_expanded
        )
    else:
        log_x_cat = torch.empty(b, 0, device=device)

    # ── 3. 逆扩散主循环：t = T-1 → 0 ───────────────────────────────
    for i in reversed(range(model.model_cfg.num_timesteps)):
        t    = torch.full((b,), i, device=device, dtype=torch.long)
        x_in = torch.cat([x_num, log_x_cat.exp()], dim=1)

        # ── 3a. 去噪网络前向 ─────────────────────────────────────────
        if not use_cfg:
            model_out = model._denoise_fn(x_in, t, y_pos)
        else:
            # 三路 CFG：unconditional / positive / negative
            # 合并为单次前向以节省显存开销
            x_in3 = torch.cat([x_in,  x_in,  x_in ], dim=0)
            t3    = torch.cat([t,     t,     t    ], dim=0)
            y3    = torch.cat([y_unc, y_pos, y_neg], dim=0)
            out3  = model._denoise_fn(x_in3, t3, y3)

            out_unc, out_pos, out_neg = torch.chunk(out3, 3, dim=0)
            model_out = (
                out_unc
                + cfg_scale      * (out_pos - out_unc)
                - repulse_weight * (out_neg - out_unc)
            )

        eps_pred      = model_out[:, :model.num_numerical_features]
        model_cat_out = model_out[:, model.num_numerical_features:]

        # ── 3b. 数值特征更新（DDPM 后验均值 + 噪声）────────────────
        x0_pred = (
            extract(model.sqrt_recip_alphas_cumprod,   t, x_num.shape) * x_num
          - extract(model.sqrt_recipml_alphas_cumprod, t, x_num.shape) * eps_pred
        )
        x0_pred = torch.clamp(x0_pred, model.model_cfg.x0_clamp_min, model.model_cfg.x0_clamp_max)

        mean = (
            extract(model.posterior_mean_coef1, t, x_num.shape) * x0_pred
          + extract(model.posterior_mean_coef2, t, x_num.shape) * x_num
        )

        if i > 0:
            log_var = torch.log(
                extract(model.posterior_variance, t, x_num.shape).clamp(min=1e-20)
            )
            noise = torch.randn_like(x_num) * temperature_num
            x_num = mean + torch.exp(0.5 * log_var) * noise
        else:
            x_num = mean    # t = 0：确定性解码，不加噪声

        # ── 3c. 离散特征更新（multinomial diffusion 后验）────────────
        new_log_x_cat = torch.empty_like(model_cat_out)

        for ix in model.slices_for_classes:
            # 温度缩放仅在 gumbel_sample 内部施加（避免重复缩放）
            logits = model_cat_out[:, ix]
            K      = logits.shape[1]

            idx    = gumbel_sample(logits, temperature_cat)
            onehot = F.one_hot(idx, num_classes=K).float()

            if torch.rand(1).item() < 1e-4:
                print("sample distribution:", torch.bincount(idx, minlength=logits.shape[1]) / idx.shape[0])

            if i > 0:
                # 软化：与训练前向加噪一致，在 one-hot 上叠加均匀噪声
                safe_prev  = torch.clamp(t - 1, min=0)
                a_bar_prev = extract(model.alphas_cumprod_prev, safe_prev, onehot.shape)
                soft_prob  = a_bar_prev * onehot + (1.0 - a_bar_prev) / K
                new_log_x_cat[:, ix] = torch.log(soft_prob.clamp(min=1e-8))
            else:
                # t = 0：hard one-hot，最终离散类别
                new_log_x_cat[:, ix] = torch.log(onehot + 1e-7)

        log_x_cat = new_log_x_cat

    # ── 4. 解码输出 ──────────────────────────────────────────────────
    x_cat = ohe_to_categories(log_x_cat.exp(), model.num_classes)
    return torch.cat([x_num, x_cat], dim=1)

