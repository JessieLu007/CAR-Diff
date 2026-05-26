'''# core/cat_loss.py
"""
分类属性损失函数 —— 修订版

解决「CE 低估少数类 / Focal 高估少数类」的两端翻车问题，核心改动：
  1. Logit Adjustment (LA) Loss           —— 代替原始 CE，对少数类施加先验 margin
  2. MarginalEMATracker                   —— 用 EMA 平滑边缘分布估计，代替嘈杂的 batch 估计
  3. 对称 KL（Jensen-Shannon）            —— 代替单向 KL，避免数值爆炸
  4. 先验感知 Focal gamma                  —— 用类别频率替代 batch entropy 决定 gamma

接口与原版完全兼容：compute_categorical_loss(model, x_cat, model_out, use_basic)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 工具 1：指数移动平均边缘分布追踪器
# ─────────────────────────────────────────────────────────────────────────────

class MarginalEMATracker(nn.Module):
    """
    为每个分类列维护一个预测边缘分布的 EMA。

    动机
    ----
    batch 级别的 softmax 均值方差极大（尤其 batch 里少数类可能一个都没有），
    直接用它计算 KL/CE 边缘损失会引入剧烈的噪声梯度。
    EMA 相当于用过去 1/(1-momentum) 个 step 的数据做滑动平均，
    让边缘估计稳定后，对齐损失的梯度方向才可信。

    用法
    ----
    # 在训练循环里，每次 forward 后调用 update
    tracker = MarginalEMATracker(model.num_classes, momentum=0.99)
    tracker.update(model_cat_out.detach(), model.slices_for_classes)

    # 把 tracker 传给 compute_categorical_loss
    loss = compute_categorical_loss(model, x_cat, model_out,
                                    marginal_tracker=tracker)
    """

    def __init__(self, true_marginals: list, momentum: float = 0.99):
        super().__init__()
        self.momentum = momentum
        # ★ 用真实先验初始化，而非均匀分布
        for i, prior in enumerate(true_marginals):
            self.register_buffer(f"ema_{i}", prior.clone().float())

    @torch.no_grad()
    def update(self, model_cat_out: torch.Tensor, slices_for_classes: list):
        """在每个训练 step 调用，传入 detach 后的模型输出"""
        for i, ix in enumerate(slices_for_classes):
            logits = model_cat_out[:, ix]
            batch_marginal = F.softmax(logits.float(), dim=1).mean(dim=0).cpu()
            ema: torch.Tensor = getattr(self, f"ema_{i}")
            # EMA 更新：新值 = momentum * 旧值 + (1 - momentum) * 当前 batch
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

    原理
    ----
    标准 CE 的贝叶斯最优决策边界不是 f(x) = 0，而是 f(x) + log(pi_y) = 0。
    LA Loss 在 logit 上直接加入先验对数，把决策边界修正到贝叶斯最优位置，
    让少数类获得与频率成正比的隐式 margin。

    tau 由调用方传入
    ---
    
    """
    adj = tau * torch.log(prior.to(logits.device) + 1e-8)   # [num_classes]
    return F.cross_entropy(logits + adj, targets)


def symmetric_kl(
    log_pred: torch.Tensor,      # 预测对数概率 [num_classes]，已经是 log 形式
    true_marginal: torch.Tensor, # 真实先验概率 [num_classes]
) -> torch.Tensor:
    """
    对称 KL 散度（≈ 2 × Jensen-Shannon Divergence）
    ------------------------------------------------
    sym_KL = 0.5 * KL(true || pred) + 0.5 * KL(pred || true)

    为什么不用单向 KL
    -----------------
    F.kl_div(log_pred, true) = KL(true || pred)
    当 true_marginal 某个类为 0 时，KL 正常；
    但当 pred 某个类概率极小时，KL(pred || true) 的 pred * log(pred/true) → 0，
    而 KL(true || pred) 的 true * log(true/pred) → ∞。
    对称形式两项互相制衡，数值更稳定。
    """
    true_m = true_marginal.to(log_pred.device)
    log_true = torch.log(true_m + 1e-8)
    pred = torch.exp(log_pred.clamp(min=-30))

    # KL(true || pred)：true * (log_true - log_pred)
    kl_fwd = F.kl_div(log_pred, true_m, reduction="sum", log_target=False)
    # KL(pred || true)：pred * (log_pred - log_true)
    kl_bwd = F.kl_div(log_true, pred,   reduction="sum", log_target=False)
    return 0.5 * (kl_fwd + kl_bwd)


def marginal_alignment_loss(
    logits: torch.Tensor,
    true_marginal: torch.Tensor,
    ema_marginal: torch.Tensor | None,
) -> torch.Tensor:
    """
    边缘分布对齐损失。
    如果提供了 ema_marginal，用 EMA 分布作为「当前预测分布的代理」，
    计算 sym_KL(true || EMA_pred)，梯度更稳定；
    否则退化为 batch 估计（嘈杂，不推荐）。

    注意：此损失只对模型的边缘分布施加软约束，不参与 per-sample 的梯度，
    权重应比主损失小（建议 gma_loss_weight ≤ 0.3）。
    """

    if ema_marginal is not None:
        # 用 EMA 作为预测边缘的平滑估计
        log_pred = torch.log(ema_marginal.to(logits.device) + 1e-8)
    else:
        # batch 估计：对 batch 维取 logsumexp 再减 log(B)
        log_probs = F.log_softmax(logits, dim=1)
        log_pred = torch.logsumexp(log_probs, dim=0) - math.log(logits.size(0))

    return symmetric_kl(log_pred, true_marginal)


def focal_loss_with_prior_gamma(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prior: torch.Tensor,
    gamma_base: float,
    use_adaptive: bool,
) -> torch.Tensor:
    """
    先验感知 Focal Loss
    -------------------
    原版问题：用 batch entropy 估计自适应 gamma 同样受 batch 噪声影响。
    改进：gamma 直接由类别先验频率决定。

        gamma_c = gamma_base * (1 - pi_c)^0.5

    - 多数类（pi_c 大）→ gamma_c 小 → 惩罚权重轻
    - 少数类（pi_c 小）→ gamma_c 大 → 惩罚权重重
    - (1 - pi_c)^0.5 起到「压缩」作用，避免极稀少类 gamma 爆炸

    当 use_adaptive=False 时退化为 gamma = gamma_base（标量），
    与原版 non-adaptive 分支行为一致。
    """
    ce = F.cross_entropy(logits, targets, reduction="none")   # [B]
    pt = torch.clamp(torch.exp(-ce), max=0.9999)              # 防止梯度 inf

    if use_adaptive:
        prior_dev = prior.to(logits.device)
        # gamma 按类别计算，再按每个样本的 target 类别索引取值
        gamma_per_class  = gamma_base * (1.0 - prior_dev).pow(0.5)  # [C]
        gamma_for_sample = gamma_per_class[targets]                 # [B]
    else:
        gamma_for_sample = gamma_base

    return ((1.0 - pt) ** gamma_for_sample * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 主函数（保持原始接口）
# ─────────────────────────────────────────────────────────────────────────────

def compute_categorical_loss(
    model,
    x_cat: torch.Tensor,
    model_out: torch.Tensor,
    use_basic: bool = False,
    marginal_tracker: MarginalEMATracker | None = None,
) -> torch.Tensor:
    """
    参数
    ----
    model             : 与原版相同，需有 num_numerical_features / slices_for_classes /
                        true_marginals / cfg 属性
    x_cat             : 真实类别标签 [B, N_cat]
    model_out         : 模型完整输出 [B, N_num + sum(N_cat_i)]
    use_basic         : True → LA-CE + 对称 KL 边缘对齐
                        False → 先验感知 Focal + 对称 KL 边缘对齐
    marginal_tracker  : MarginalEMATracker 实例，提供 EMA 平滑的预测边缘分布；
                        为 None 时退化为 batch 估计（不推荐）
    """
    num_feats          = model.num_numerical_features
    model_cat_out      = model_out[:, num_feats:]
    slices_for_classes = model.slices_for_classes
    true_marginals     = model.true_marginals
    cfg                = model.cfg

    # ── 超参读取（带默认值，向后兼容）───────────────────────────────────
    tau_base            = getattr(cfg, "logit_adj_tau",   0.3)   # LA loss 先验修正强度
    gma_weight      = getattr(cfg, "gma_loss_weight", 0.08)   # 边缘对齐损失权重
    gamma_base      = getattr(cfg, "gamma_base",      2.0)   # focal gamma 基础值
    use_adaptive    = getattr(cfg, "use_adaptive_focal", False)

    n = len(slices_for_classes)
    if n == 0:
        return torch.tensor(0.0, device=model_out.device)

    loss_local    = torch.tensor(0.0, device=model_out.device)
    loss_global   = torch.tensor(0.0, device=model_out.device)

    for i, ix in enumerate(slices_for_classes):
        logits     = model_cat_out[:, ix]                        # [B, C_i]
        targets    = x_cat[:, i]                                 # [B]
        #print(targets.unique())
        prior      = true_marginals[i]                           # [C_i]
        ema_m      = marginal_tracker[i] if marginal_tracker is not None else None

        # ── 主损失（微观，per-sample）───────────────────────────────────
        if use_basic:
            tau_i = compute_adaptive_tau(prior, tau_base)
            loss_local += logit_adjusted_ce(logits, targets, prior, tau=tau_i)
        else:
            # 先验感知 Focal：基于类别频率决定 gamma，而非嘈杂的 batch entropy
            loss_local += focal_loss_with_prior_gamma(
                logits, targets, prior,
                gamma_base=gamma_base,
                use_adaptive=use_adaptive,
            )

        # ── 辅助损失（宏观，边缘对齐）──────────────────────────────
        # 
        min_freq = prior.min().item()
        imbalance = max(0.0, 1.0 - 2.0 * min_freq)

        gma_i = gma_weight * (imbalance ** 0.5) # 开方使中间衰减不过快
        loss_global += gma_i * marginal_alignment_loss(logits, prior, ema_m)

    loss_local  /= n
    loss_global /= n

    return loss_local + loss_global'''



# core/cat_loss.py
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
    use_adaptive = getattr(cfg, "use_adaptive_focal", False)

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
        imbalance   = max(0.0, 1.0 - 2.0 * prior.min().item())
        loss_global += gma_weight * (imbalance ** 0.5) * _marginal_loss(logits, prior, ema_m)

    n = len(slices)
    return loss_local / n + loss_global / n
