# core/cat_loss_autodl.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginalEMATracker(nn.Module):
    """EMA 平滑的预测边缘分布，用真实先验初始化。"""

    def __init__(self, true_marginals: list, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        for i, prior in enumerate(true_marginals):
            self.register_buffer(f"ema_{i}", prior.clone().float())

    @torch.no_grad()
    def update(self, model_cat_out: torch.Tensor, slices: list):
        for i, ix in enumerate(slices):
            batch_m = F.softmax(model_cat_out[:, ix].float(), dim=1).mean(0).cpu()
            old = getattr(self, f"ema_{i}")
            setattr(self, f"ema_{i}", self.momentum * old + (1 - self.momentum) * batch_m)

    def __getitem__(self, i: int) -> torch.Tensor:
        return getattr(self, f"ema_{i}")


# ── 损失组件 ─────────────────────────────────────────────────────────────────

def _la_ce(logits, targets, prior, tau_base):
    """Logit Adjustment CE：按不平衡程度自适应 tau。"""
    imbalance = max(0.0, 1.0 - 2.0 * prior.min().item())
    tau = tau_base * imbalance
    adj = tau * torch.log(prior.to(logits.device) + 1e-8)
    return F.cross_entropy(logits + adj, targets)


def _focal(logits, targets, prior, gamma_base, use_adaptive):
    """先验感知 Focal Loss：gamma 由类别频率决定，避免 batch entropy 噪声。"""
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.clamp(torch.exp(-ce), max=0.9999)
    if use_adaptive:
        gamma = (gamma_base * (1.0 - prior.to(logits.device)).pow(0.5))[targets]
    else:
        gamma = gamma_base
    return ((1 - pt) ** gamma * ce).mean()


def _sym_kl(log_pred, true_marginal):
    """对称 KL（≈ 2×JS），比单向 KL 数值更稳定。"""
    p = true_marginal.to(log_pred.device)
    kl_fwd = F.kl_div(log_pred, p, reduction="sum", log_target=False)
    kl_bwd = F.kl_div(torch.log(p + 1e-8), log_pred.exp(), reduction="sum", log_target=False)
    return 0.5 * (kl_fwd + kl_bwd)


def _marginal_loss(logits, prior, ema_marginal):
    """边缘对齐：优先用 EMA 估计，否则退化为 batch 估计。"""
    if ema_marginal is not None:
        log_pred = torch.log(ema_marginal.to(logits.device) + 1e-8)
    else:
        log_pred = torch.logsumexp(
            F.log_softmax(logits, dim=1), dim=0
        ) - math.log(logits.size(0))
    return _sym_kl(log_pred, prior)


# ── 主接口 ───────────────────────────────────────────────────────────────────

def compute_categorical_loss(
    model,
    x_cat: torch.Tensor,
    model_out: torch.Tensor,
    timesteps: torch.Tensor,     # 新增参数，从训练循环传入
    use_basic: bool = False,
    marginal_tracker: MarginalEMATracker | None = None,
) -> torch.Tensor:
    
    logits_all    = model_out[:, model.num_numerical_features:]
    slices        = model.slices_for_classes
    true_marginals = model.true_marginals
    cfg           = model.cfg

    if not slices:
        return torch.tensor(0.0, device=model_out.device)

    tau_base     = getattr(cfg, "logit_adj_tau",     0.3)
    gma_weight   = getattr(cfg, "gma_loss_weight",   0.08)
    gamma_base   = getattr(cfg, "gamma_base",        2.0)
    use_adaptive = getattr(cfg, "use_adaptive_focal", True)
    T            = getattr(cfg, "num_timesteps", 1000)

    # timestep 权重： t小（接近干净数据）时边缘对齐权重高
    # t大（高噪声）时权重衰减，去强迫模型对齐噪声分布的边缘

    if timesteps is not None:
        t_norm  = timesteps.float().mean() / T
        t_weight = (1.0 - t_norm).clamp(0.1, 1.0)
    else:
        t_weight = 1.0

    loss_local = loss_global = torch.tensor(0.0, device=model_out.device)

    for i, ix in enumerate(slices):
        logits = logits_all[:, ix]
        prior  = true_marginals[i]
        ema_m  = marginal_tracker[i] if marginal_tracker is not None else None

        # 微观：per-sample 主损失
        loss_local += (
            _la_ce(logits, x_cat[:, i], prior, tau_base) if use_basic
            else _focal(logits, x_cat[:, i], prior, gamma_base, use_adaptive)
        )

        # 宏观：边缘分布对齐，权重随不平衡程度缩放
        #imbalance   = max(0.0, 1.0 - 2.0 * prior.min().item())
        #loss_global += gma_weight * (imbalance ** 0.5) * t_weight *_marginal_loss(logits, prior, ema_m)

        imbalance = max(0.0, 1.0 - 2.0 * prior.min().item())
        marginal_coef = max(imbalance ** 0.5, 0.2) # 最小0.2， 均衡列不完全关掉
        loss_global += gma_weight * marginal_coef * t_weight * _marginal_loss(logits, prior, ema_m)

    n = len(slices)
    return loss_local / n + loss_global / n