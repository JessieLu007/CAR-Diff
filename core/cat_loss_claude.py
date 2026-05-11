# core/cat_loss.py
"""
分类属性损失函数 —— v2 修订

v1 → v2 的三处关键修复
-----------------------
[Fix 1] MarginalEMATracker 初始化改为「真实先验」而非均匀分布
        均匀初始化会在 EMA 收敛前持续把边缘损失的梯度推向错误方向，
        是 prevalentHyp 偏差 +11.74% 的根本原因。

[Fix 2] Logit Adjustment tau 改为「按特征不平衡程度自适应」
        对近似平衡的特征（currentSmoker 50/50）tau → 0，纠偏力量接近 0；
        对高度不平衡的特征（BPMeds 97/3）tau 自动放大。
        公式：tau_i = tau_base * imbalance_i，其中 imbalance = 1 - 2 * min_freq

[Fix 3] gma_loss_weight 建议从 0.2 降至 0.08，且 per-feature 按不平衡度缩放
        防止稀少类受到双重强调（LA 已经修正 + KL 又推一次）。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1：EMA 追踪器 —— 初始化为真实先验
# ─────────────────────────────────────────────────────────────────────────────

class MarginalEMATracker(nn.Module):
    """
    为每个分类列维护一个预测边缘分布的 EMA。

    v2 修复
    -------
    初始化改为 true_marginals 而非 ones(c)/c。
    均匀初始化在 EMA 收敛前（约 1/(1-0.99)=100 step）会把梯度
    持续推向错误的均匀目标，与真实先验差异越大的列受害越严重。
    用真实先验初始化后，EMA 从一开始就在正确位置附近波动，
    梯度方向可信。

    用法（训练循环）
    ---------------
    tracker = MarginalEMATracker(model.true_marginals, momentum=0.99)
    # 每个 step 在 loss.backward() 之前调用：
    tracker.update(model_cat_out.detach(), model.slices_for_classes)
    loss = compute_categorical_loss(..., marginal_tracker=tracker)
    """

    def __init__(self, true_marginals: list, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        # ★ 用真实先验初始化，而非均匀分布
        for i, prior in enumerate(true_marginals):
            self.register_buffer(f"ema_{i}", prior.clone().float())

    @torch.no_grad()
    def update(self, model_cat_out: torch.Tensor, slices_for_classes: list):
        for i, ix in enumerate(slices_for_classes):
            logits = model_cat_out[:, ix]
            batch_marginal = F.softmax(logits.float(), dim=1).mean(dim=0).cpu()
            ema: torch.Tensor = getattr(self, f"ema_{i}")
            setattr(self, f"ema_{i}",
                    self.momentum * ema + (1.0 - self.momentum) * batch_marginal)

    def __getitem__(self, i: int) -> torch.Tensor:
        return getattr(self, f"ema_{i}")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2：自适应 tau 计算
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_tau(prior: torch.Tensor, tau_base: float) -> float:
    """
    根据类别不平衡程度动态计算 Logit Adjustment 的 tau 值。

    公式
    ----
    imbalance = 1 - 2 * min(prior)
    tau_i     = tau_base * imbalance

    各列的实际 tau（tau_base=0.3）
    --------------------------------
    currentSmoker (min≈0.492) → imbalance=0.016 → tau≈0.005  几乎不修正
    prevalentHyp  (min≈0.310) → imbalance=0.380 → tau≈0.11   轻度修正
    BPMeds        (min≈0.029) → imbalance=0.942 → tau≈0.28   充分修正
    diabetes      (min≈0.026) → imbalance=0.948 → tau≈0.28   充分修正
    """
    min_freq  = prior.min().item()
    imbalance = max(0.0, 1.0 - 2.0 * min_freq)
    return tau_base * imbalance


# ─────────────────────────────────────────────────────────────────────────────
# 损失函数组件
# ─────────────────────────────────────────────────────────────────────────────

def logit_adjusted_ce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prior: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """
    Logit Adjustment Loss (Menon et al., ICLR 2021)
    L = CE(logits + tau * log(prior), targets)
    tau 由调用方传入（v2 改为 per-feature 自适应值）。
    """
    adj = tau * torch.log(prior.to(logits.device) + 1e-8)
    return F.cross_entropy(logits + adj, targets)


def symmetric_kl(
    log_pred: torch.Tensor,
    true_marginal: torch.Tensor,
) -> torch.Tensor:
    """
    对称 KL 散度：0.5 * [KL(true‖pred) + KL(pred‖true)]
    两项互相制衡，避免 pred≈0 时单向 KL 梯度爆炸。
    """
    true_m   = true_marginal.to(log_pred.device)
    log_true = torch.log(true_m + 1e-8)
    pred     = torch.exp(log_pred.clamp(min=-30))

    kl_fwd = F.kl_div(log_pred, true_m, reduction="sum", log_target=False)
    kl_bwd = F.kl_div(log_true, pred,   reduction="sum", log_target=False)
    return 0.5 * (kl_fwd + kl_bwd)


def marginal_alignment_loss(
    logits: torch.Tensor,
    true_marginal: torch.Tensor,
    ema_marginal: torch.Tensor | None,
) -> torch.Tensor:
    """用 EMA 平滑的预测边缘分布计算 sym-KL 对齐损失。"""
    if ema_marginal is not None:
        log_pred = torch.log(ema_marginal.to(logits.device) + 1e-8)
    else:
        log_probs = F.log_softmax(logits, dim=1)
        log_pred  = torch.logsumexp(log_probs, dim=0) - math.log(logits.size(0))

    return symmetric_kl(log_pred, true_marginal)


def focal_loss_with_prior_gamma(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prior: torch.Tensor,
    gamma_base: float,
    use_adaptive: bool,
) -> torch.Tensor:
    """
    先验感知 Focal Loss：gamma_c = gamma_base * (1 - pi_c)^0.5
    多数类获得较小 gamma，少数类获得较大 gamma。
    """
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.clamp(torch.exp(-ce), max=0.9999)

    if use_adaptive:
        prior_dev        = prior.to(logits.device)
        gamma_per_class  = gamma_base * (1.0 - prior_dev).pow(0.5)
        gamma_for_sample = gamma_per_class[targets]
    else:
        gamma_for_sample = gamma_base

    return ((1.0 - pt) ** gamma_for_sample * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 主函数（接口与 v1 完全兼容）
# ─────────────────────────────────────────────────────────────────────────────

def compute_categorical_loss(
    model,
    x_cat: torch.Tensor,
    model_out: torch.Tensor,
    use_basic: bool = False,
    marginal_tracker: MarginalEMATracker | None = None,
) -> torch.Tensor:
    """
    关键超参（cfg 属性，带默认值）
    ------------------------------
    logit_adj_tau     : float = 0.3   LA loss 的 tau_base（会被自适应缩放）
    gma_loss_weight   : float = 0.08  边缘对齐损失权重（↓ 从 v1 的 0.2）
    gamma_base        : float = 2.0   focal gamma 基础值
    use_adaptive_focal: bool  = False
    """
    num_feats          = model.num_numerical_features
    model_cat_out      = model_out[:, num_feats:]
    slices_for_classes = model.slices_for_classes
    true_marginals     = model.true_marginals
    cfg                = model.cfg

    tau_base     = getattr(cfg, "logit_adj_tau",      0.3)
    gma_weight   = getattr(cfg, "gma_loss_weight",    0.08)
    gamma_base   = getattr(cfg, "gamma_base",         2.0)
    use_adaptive = getattr(cfg, "use_adaptive_focal", False)

    n = len(slices_for_classes)
    if n == 0:
        return torch.tensor(0.0, device=model_out.device)

    loss_local  = torch.tensor(0.0, device=model_out.device)
    loss_global = torch.tensor(0.0, device=model_out.device)

    for i, ix in enumerate(slices_for_classes):
        logits  = model_cat_out[:, ix]
        targets = x_cat[:, i]
        prior   = true_marginals[i]
        ema_m   = marginal_tracker[i] if marginal_tracker is not None else None

        # ── 主损失（微观，per-sample）────────────────────────────────
        if use_basic:
            tau_i = compute_adaptive_tau(prior, tau_base)   # Fix 2
            loss_local += logit_adjusted_ce(logits, targets, prior, tau=tau_i)
        else:
            loss_local += focal_loss_with_prior_gamma(
                logits, targets, prior,
                gamma_base=gamma_base,
                use_adaptive=use_adaptive,
            )

        # ── 辅助损失（宏观，边缘对齐）────────────────────────────────
        # Fix 3：gma 权重按不平衡度缩放，平衡列权重趋近 0，
        #        避免对已被 LA 修正过的少数类再次施加强梯度
        min_freq  = prior.min().item()
        imbalance = max(0.0, 1.0 - 2.0 * min_freq)
        gma_i     = gma_weight * (imbalance ** 0.5)   # 开方使中间衰减不过快

        loss_global += gma_i * marginal_alignment_loss(logits, prior, ema_m)

    loss_local  /= n
    loss_global /= n

    return loss_local + loss_global
