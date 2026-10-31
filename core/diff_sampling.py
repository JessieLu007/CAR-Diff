# core/diff_sampling.py

import torch
import torch.nn.functional as F

from .utils import extract
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

def _sample_core_once(
    model,
    num_samples : int,
    y_cond      : int,
) -> torch.Tensor:
    b      = num_samples
    device = model.device
    sampling_cfg = model.sampling_cfg

    n_num           = model.num_numerical_features
    cat_ar          = bool(getattr(model.model_cfg, 'cat_ar', False)) and len(model.num_classes) > 0
    temperature_num = 1.0
    temperature_cat = sampling_cfg.cat_temperature_cap
    cfg_scale       = sampling_cfg.cfg_scale_fidelity
    repulse_weight  = sampling_cfg.repulse_weight_fidelity

    if model.train_cfg.task == 'full_data':
        y_cond = model.model_cfg.null_class_idx

    y_pos = torch.full((b,), y_cond,                         device=device, dtype=torch.long)
    y_unc = torch.full((b,), model.model_cfg.null_class_idx, device=device, dtype=torch.long)
    y_neg = torch.full((b,), 1 - y_cond,                     device=device, dtype=torch.long)
    use_cfg = (model.train_cfg.task == 'minority_cond')
    if cat_ar and use_cfg:
        raise NotImplementedError("cat_ar 暂不支持 minority_cond 的 CFG 采样，请设 cat_ar=False")

    x_num = torch.randn(b, n_num, device=device)
    x_cat = torch.empty(b, 0, device=device)

    # ── 逆扩散主循环：只对数值列扩散，类别只在最后一步解码 ──
    for i in reversed(range(model.model_cfg.num_timesteps)):
        t    = torch.full((b,), i, device=device, dtype=torch.long)
        last = (i == 0)

        if cat_ar:
            if last:
                model_out, x_cat_idx = model._denoise_fn.sample_final(x_num, t, y_pos, temperature_cat)
            else:
                model_out = model._denoise_fn(x_num, t, y_pos, decode_cat=False)
        elif not use_cfg:
            model_out = model._denoise_fn(x_num, t, y_pos, decode_cat=last)
        else:
            x3 = torch.cat([x_num, x_num, x_num], dim=0)
            t3 = torch.cat([t,     t,     t    ], dim=0)
            y3 = torch.cat([y_unc, y_pos, y_neg], dim=0)
            out3 = model._denoise_fn(x3, t3, y3, decode_cat=last)
            out_unc, out_pos, out_neg = torch.chunk(out3, 3, dim=0)
            model_out = (
                out_unc
                + cfg_scale      * (out_pos - out_unc)
                - repulse_weight * (out_neg - out_unc)
            )

        eps_pred = model_out[:, :n_num]

        # 数值更新（DDPM 后验均值 + 噪声）
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
            x_num = mean + torch.exp(0.5 * log_var) * torch.randn_like(x_num) * temperature_num
        else:
            x_num = mean

        # 类别：最后一步解码
        if last and len(model.num_classes) > 0:
            if cat_ar:
                x_cat = x_cat_idx.to(x_num.dtype)
            else:
                model_cat_out = model_out[:, n_num:]
                idx_list = [
                    gumbel_sample(model_cat_out[:, ix], temperature_cat)
                    for ix in model.slices_for_classes
                ]
                x_cat = torch.stack(idx_list, dim=1).to(x_num.dtype)

    return torch.cat([x_num, x_cat], dim=1)


def sample_core(model, num_samples: int, y_cond: int,
                chunk_size: int = 2048) -> torch.Tensor:
    """分块采样，避免一次性 forward 爆显存。"""
    if num_samples <= chunk_size:
        return _sample_core_once(model, num_samples, y_cond)

    outs, done, chunk = [], 0, chunk_size
    while done < num_samples:
        n = min(chunk, num_samples - done)
        try:
            out = _sample_core_once(model, n, y_cond)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk = max(256, chunk // 2)
            print(f"[sample_core] OOM, retry with chunk_size={chunk}")
            continue
        outs.append(out.cpu())      # 立刻挪到 CPU，释放显存
        done += n
        torch.cuda.empty_cache()
        print(f"[sample_core] {done}/{num_samples}")

    return torch.cat(outs, dim=0)

