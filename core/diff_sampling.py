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

def _sample_core_once(
    model,
    num_samples : int,
    y_cond      : int,
) -> torch.Tensor:
    print(">>> ENTER sample_core")
    b      = num_samples
    device = model.device
    sampling_cfg = model.sampling_cfg

    cat_mode        = getattr(model.model_cfg, 'cat_input_mode', 'legacy')
    temperature_num = 1.0
    temperature_cat = sampling_cfg.cat_temperature_cap
    cfg_scale       = sampling_cfg.cfg_scale_fidelity
    repulse_weight  = sampling_cfg.repulse_weight_fidelity

    # full_data 任务一律使用 null 条件，不启用 CFG
    if model.train_cfg.task == 'full_data':
        y_cond = model.model_cfg.null_class_idx

    y_pos = torch.full((b,), y_cond,                   device=device, dtype=torch.long)
    y_unc = torch.full((b,), model.model_cfg.null_class_idx, device=device, dtype=torch.long)
    y_neg = torch.full((b,), 1 - y_cond,               device=device, dtype=torch.long)

    use_cfg = (model.train_cfg.task == 'minority_cond')

    # ── 2. 初始化采样起点（对应训练时 t = T 的完全加噪状态）─────────
    x_num = torch.randn(b, model.num_numerical_features, device=device)

    cat_state = None   # 仅 sampled 模式使用：当前 x_t 的 one-hot [B, sum(K)]
    if len(model.num_classes) > 0:
        # 类别特征：均匀分布的 log 概率（最大熵状态）
        log_x_cat = torch.log(
            torch.ones(b, sum(model.num_classes), device=device)
            / model.num_classes_expanded
        )
        if cat_mode == 'sampled':
            # x_T ~ 均匀分布，硬采样
            cat_state = torch.cat([
                F.one_hot(torch.randint(0, K, (b,), device=device), num_classes=K).float()
                for K in model.num_classes
            ], dim=1)
    else:
        log_x_cat = torch.empty(b, 0, device=device)

    # ── 3. 逆扩散主循环：t = T-1 → 0 ───────────────────────────────
    for i in reversed(range(model.model_cfg.num_timesteps)):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        if cat_mode == 'sampled' and cat_state is not None:
            x_in = torch.cat([x_num, cat_state], dim=1)
        else:
            x_in = torch.cat([x_num, log_x_cat.exp()], dim=1)

        # ── 3a. 去噪网络前向 ─────────────────────────────────────────
        if not use_cfg:
            model_out = model._denoise_fn(x_in, t, y_pos)
        else:
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

        # ── 3c. 离散特征更新 ─────────────────────────────────────────
        if cat_mode == 'sampled':
            # 多项式后验 q(x_{t-1}|x_t, x0)，x0 用模型预测分布边缘化；t=0 直接从 x0 预测分布采样
            new_parts = []
            for ix, K in zip(model.slices_for_classes, model.num_classes):
                p0 = F.softmax(model_cat_out[:, ix] / temperature_cat, dim=-1)
                if i > 0:
                    xt         = cat_state[:, ix]
                    alpha_t    = 1.0 - extract(model.betas, t, xt.shape)
                    a_bar_prev = extract(model.alphas_cumprod_prev, t, xt.shape)
                    post = (alpha_t * xt + (1.0 - alpha_t) / K) \
                         * (a_bar_prev * p0 + (1.0 - a_bar_prev) / K)
                    post = post / post.sum(dim=-1, keepdim=True)
                    idx  = gumbel_sample(torch.log(post.clamp(min=1e-30)), 1.0)
                else:
                    idx  = gumbel_sample(torch.log(p0.clamp(min=1e-30)), 1.0)
                new_parts.append(F.one_hot(idx, num_classes=K).float())
            cat_state = torch.cat(new_parts, dim=1)
        else:
            new_log_x_cat = torch.empty_like(model_cat_out)

            for ix in model.slices_for_classes:
                # 温度缩放仅在 gumbel_sample 内部施加（避免重复缩放）
                logits = model_cat_out[:, ix]
                K      = logits.shape[1]

                idx    = gumbel_sample(logits, temperature_cat)
                onehot = F.one_hot(idx, num_classes=K).float()

                if i > 0:
                    if cat_mode == 'prob':
                        # 与训练端对齐：下一步 t-1 的输入使用 ᾱ_{t-1}
                        a_bar = extract(model.alphas_cumprod, t - 1, onehot.shape)
                    else:
                        # legacy：保持原写法
                        safe_prev = torch.clamp(t - 1, min=0)
                        a_bar     = extract(model.alphas_cumprod_prev, safe_prev, onehot.shape)
                    soft_prob = a_bar * onehot + (1.0 - a_bar) / K
                    new_log_x_cat[:, ix] = torch.log(soft_prob.clamp(min=1e-8))
                else:
                    # t = 0：hard one-hot，最终离散类别
                    new_log_x_cat[:, ix] = torch.log(onehot + 1e-7)

            log_x_cat = new_log_x_cat

    # ── 4. 解码输出 ──────────────────────────────────────────────────
    if cat_mode == 'sampled' and cat_state is not None:
        log_x_cat = torch.log(cat_state + 1e-7)
    x_cat = ohe_to_categories(log_x_cat.exp(), model.num_classes)
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

