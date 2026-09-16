# core/cat_loss_focal_mode.py
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
    @torch.compiler.disable
    def update(self, model_cat_out: torch.Tensor, slices: list):
        for i, ix in enumerate(slices):
            batch_m = F.softmax(model_cat_out[:, ix].float(), dim=1).mean(0)
            old = getattr(self, f"ema_{i}")
            setattr(self, f"ema_{i}", (self.momentum * old + (1 - self.momentum) * batch_m).clone())

    def __getitem__(self, i: int) -> torch.Tensor:
        return getattr(self, f"ema_{i}")


# ── 损失组件 ─────────────────────────────────────────────────────────────────

def _get_marginal_imbalances(model) -> list:
    """imbalance 只由固定的 true_marginals 决定，训练全程不变，只需算一次。
    缓存后不再需要每个 step、每一列都触发 .item() 同步——这也是 torch.compile
    反复重新编译、图断裂的直接原因之一。"""
    cached = getattr(model, "_marginal_imbalance_cache", None)
    if cached is None:
        cached = [
            max(0.0, 1.0 - 2.0 * prior.min().item())
            for prior in model.true_marginals
        ]
        model._marginal_imbalance_cache = cached
    return cached


def _la_ce(logits, targets, prior, tau_base, imbalance):
    """Logit Adjustment CE：按不平衡程度自适应 tau。"""
    tau = tau_base * imbalance
    adj = tau * torch.log(prior.to(logits.device) + 1e-8)
    return F.cross_entropy(logits + adj, targets)


def _focal(logits, targets, prior, gamma_base, focal_mode, imbalance,
           alpha_scale=1.0, alpha_imbalance_threshold=0.2):
    """
    Focal Loss with optional adaptive gamma and class-balanced alpha.

    focal_mode:
        "fixed"    — 固定 gamma_base，无 alpha 加权
        "adaptive" — 少数类 gamma 更小（保留其梯度），并用 1/sqrt(prior) 加权
    """
    prior = prior.to(logits.device)
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.clamp(torch.exp(-ce), max=0.9999)

    if focal_mode == "adaptive":
        # 少数类 gamma 更小：prior 越小，gamma 越小，少数类梯度不被过度压制
        # gamma ∈ [gamma_base * 0.3, gamma_base]
        gamma_per_class = gamma_base * (0.3 + 0.7 * prior.pow(0.5))
        gamma = gamma_per_class[targets]

        if imbalance > alpha_imbalance_threshold:
            # alpha：少数类损失放大（类比"翻倍"少数类）
            # 归一化到均值为 1，避免整体 loss 量级漂移
            alpha_per_class = (1.0 / (prior + 1e-8)).pow(0.5)
            alpha_per_class = alpha_per_class / alpha_per_class.mean()
            alpha_per_class = 1.0 + alpha_scale * (alpha_per_class - 1.0)
            alpha = alpha_per_class[targets]
        else: 
            alpha = 1.0 # 平衡列只保留gamma，不引入方向偏差
    else:
        gamma = gamma_base
        alpha = 1.0

    return (alpha * (1 - pt) ** gamma * ce).mean()


def _sym_kl(log_pred, true_marginal):
    """对称 KL（≈ 2×JS），比单向 KL 数值更稳定。"""
    p = true_marginal.to(log_pred.device)
    kl_fwd = F.kl_div(log_pred, p, reduction="sum", log_target=False)
    kl_bwd = F.kl_div(torch.log(p + 1e-8), log_pred.exp(), reduction="sum", log_target=False)
    return 0.5 * (kl_fwd + kl_bwd)


def _marginal_loss(logits, prior, ema_marginal):
    """边缘对齐：优先用 EMA 估计，否则退化为 batch 估计。"""
    log_pred = torch.logsumexp(
        F.log_softmax(logits, dim=1), dim=0
    ) - math.log(logits.size(0))

    # target 侧：有 EMA 时用 EMA 平滑（比 true_prior 更稳定），否则用先验
    target = ema_marginal.to(logits.device) if ema_marginal is not None else prior
    return _sym_kl(log_pred, target)

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
    timesteps: torch.Tensor | None = None,
    marginal_tracker: MarginalEMATracker | None = None,
) -> torch.Tensor:
    """
    分类损失统一入口。

    配置项（从 model.cfg 读取）：
        focal_mode: "off" | "fixed" | "adaptive"
            - "off"      : 使用 Logit Adjustment CE
            - "fixed"    : 固定 gamma 的 Focal
            - "adaptive" : prior 感知 gamma + alpha 类权重
        use_marginal_align: bool
            是否启用边缘分布对齐损失（true_marginal 软约束）
        logit_adj_tau:     LA-CE 的 tau 基数
        gamma_base:        Focal 的 gamma 基数
        focal_alpha_scale: adaptive 模式下 alpha 的强度（0=无加权，1=完整 1/sqrt(prior)）
        gma_loss_weight:   边缘对齐损失的权重
        num_timesteps:     扩散总步数（用于 t 加权）
    """
    logits_all     = model_out[:, model.num_numerical_features:]
    slices         = model.slices_for_classes
    true_marginals = model.true_marginals
    cfg            = model.model_cfg

    if not slices:
        return torch.tensor(0.0, device=model_out.device)

    # ── 配置 ─────────────────────────────────────────────
    focal_mode         = getattr(cfg, "focal_mode",         "adaptive")  # "off" | "fixed" | "adaptive"
    use_marginal_align = getattr(cfg, "use_marginal_align", True)
    tau_base           = getattr(cfg, "logit_adj_tau",      0.3)
    gamma_base         = getattr(cfg, "gamma_base",         2.0)
    alpha_scale        = getattr(cfg, "focal_alpha_scale",  1.0)
    gma_weight         = getattr(cfg, "gma_loss_weight",    0.08)
    T                  = getattr(cfg, "num_timesteps",      1000)

    # 兼容旧配置（可选，迁移期保留，确认无引用后删除）
    if hasattr(cfg, "use_basic") and not hasattr(cfg, "focal_mode"):
        if getattr(cfg, "use_basic", False):
            focal_mode = "off"
        elif getattr(cfg, "use_adaptive_focal", False):
            focal_mode = "adaptive"
        else:
            focal_mode = "fixed"

    # ── timestep 权重：低噪声步对齐边缘，高噪声步不对齐 ─
    t_weight = 1.0
    if timesteps is not None and use_marginal_align:
        t_norm = timesteps.float().mean() / T
        t_weight = (1.0 - t_norm).clamp(0.1, 1.0)

    loss_local  = torch.tensor(0.0, device=model_out.device)
    loss_global = torch.tensor(0.0, device=model_out.device)

    imbalances = _get_marginal_imbalances(model)

    for i, ix in enumerate(slices):
        logits = logits_all[:, ix]
        prior  = true_marginals[i]
        imbalance = imbalances[i]

        # 微观：per-sample 主损失
        if focal_mode == "off":
            loss_local += _la_ce(logits, x_cat[:, i], prior, tau_base, imbalance)
        else:
            loss_local += _focal(
                logits, x_cat[:, i], prior,
                gamma_base=gamma_base,
                focal_mode=focal_mode,
                imbalance=imbalance,
                alpha_scale=alpha_scale,
            )

        # 宏观：边缘分布对齐
        if use_marginal_align:
            ema_m = marginal_tracker[i] if marginal_tracker is not None else None

            if imbalance < 0.1:
                # 均衡列：只施加很弱的软约束，避免干扰
                marginal_coef = 0.02 * t_weight
            else:
                marginal_coef = max(imbalance ** 0.5, 0.2) * t_weight

            loss_global += gma_weight * marginal_coef * _marginal_loss(logits, prior, ema_m)

    n = len(slices)
    if use_marginal_align:
        return loss_local / n + loss_global / n
    return loss_local / n