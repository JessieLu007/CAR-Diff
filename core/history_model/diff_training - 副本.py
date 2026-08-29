# core/diff_training.py
import math
import torch
from torch.optim.lr_scheduler import LambdaLR

from .cat_loss import MarginalEMATracker


# ----------------------------------------------------------------
# Loss 历史记录器
# ----------------------------------------------------------------
class LossHistory:
    """轻量loss记录容器，新增对验证损失的记录和最佳模型追踪"""

    def __init__(self):
        self.train_total: list[float] = []
        self.train_num: list[float] = []
        self.train_cat: list[float] = []
        self.val_total: list[float] = [] # 新增验证总损失
        self.best_monitor_loss: float = float('inf') # 追踪用于早停的最佳损失（通常是验证损失）
        self.best_epoch: int = 0

    def append(self, train_total: float, train_num: float, train_cat: float,
               val_total: float | None, epoch: int): # 新增 val_total 参数
        self.train_total.append(train_total)
        self.train_num.append(train_num)
        self.train_cat.append(train_cat)
        if val_total is not None:
            self.val_total.append(val_total)

    def set_best(self, epoch: int, monitor_loss: float) -> None:
        self.best_epoch         = epoch
        self.best_monitor_loss  = monitor_loss


# ----------------------------------------------------------------
# 学习率调度 
# ----------------------------------------------------------------
def get_warmup_cosine_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs:  int,
    min_lr_ratio:  float = 0.05,   # 最低 lr = base_lr * 0.05
) -> LambdaLR:
    """
    线性warmup + Cosine Decay调度器
    0 → warmup_epochs : 线性 warmup，lr 从 0 升到 base_lr
    warmup_epochs → total_epochs : cosine decay，lr 降到 base_lr * min_lr_ratio
    """
    def lr_lambda(epoch) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)

# ----------------------------------------------------------------
# 标签平滑（可选，用于minority_cond模式）
# 以小概率将 y 替换为随机类别，减少模型对 label 条件的过度依赖
# ----------------------------------------------------------------

def smooth_labels(
        y           : torch.Tensor, 
        num_classes : int,
        smooth_prob : float = 0.05
        ) -> torch.Tensor:
    """
    以smooth_prob的概率将标签替换为随机类别，
    用于减少模型对 label 条件的过度依赖，缓解采样侧的类别偏移。
    """
    mask     = torch.rand_like(y.float()) < smooth_prob
    random_y = torch.randint_like(y, low=0, high=num_classes)
    return torch.where(mask, random_y, y)

# ----------------------------------------------------------------
# 训练主循环
# ----------------------------------------------------------------
def train_loop(model, train_dataloader, val_dataloader, train_cfg) -> None:
    task_name = train_cfg.task
    is_minority = task_name in ('minority_uncond', 'minority_cond')

    # ------- 两阶段训练开关 ---------------------
    # use_two_phase=True 且 is_minority=True 时启用分阶段冻结
    # Phase 1：冻结数值编码器，优先稳定类别表示
    # Phase 2：解冻全部参数，联合微调
    use_two_phase = getattr(train_cfg, 'use_two_phase', False)
    PHASE1_EPOCHS = getattr(train_cfg, 'phase1_epochs', 100)
    in_phase1     = use_two_phase and is_minority

    # ------ Checkpoint Ensemble 设置 -------------
    # minority 模式下保存val loss最优的K个checkpoint， 训练结束后做权重平均
    ENSEMBLE_K = getattr(train_cfg, 'ensemble_k', 5) if is_minority else 1
    top_k_states = [] # list of (monitor_loss, state_dict)
 
    # ------ Optimizer --------------------------
    optimizer = torch.optim.AdamW(
        list(model._denoise_fn.parameters()) + [model.log_vars], 
        lr=train_cfg.lr, 
        weight_decay=1e-4 if is_minority else 1e-5) # minority下加强 weight decay

    # ------ Scheduler --------------------------
    warmup_epochs = 10 if is_minority else 20
    scheduler = get_warmup_cosine_scheduler(
        optimizer,
        warmup_epochs = warmup_epochs,
        total_epochs = train_cfg.epochs,
        min_lr_ratio = 0.05,  # 最终 lr ≈ 0.0005 * 0.05 = 2.5e-5
    )
    
    # ------ 验证集监控策略 ---------------------------------
    val_size = sum(len(b[0]) for b in val_dataloader) if val_dataloader else 0

    if val_dataloader is not None and val_size >0:
        USE_VAL_MONITOR = True # 只要有验证集就用
        if val_size < 200:
            EMA_ALPHA = 0.15  # 更平滑，抑制小验证集的噪声
            print(f"⚠️  验证集仅 {val_size} 条，强制使用 val 监控，EMA 平滑加强至 {EMA_ALPHA}")
        else:
            EMA_ALPHA = 0.1
    else:
        USE_VAL_MONITOR = False
        EMA_ALPHA = 0.1

    # ----- MarginalEMATracker ----------------------------
    marginal_tracker = MarginalEMATracker(
        true_marginals=model.true_marginals,
        momentum=0.99,
    ).to(model.device)
    
    # —————训练状态初始化—————————————————————————————————————————————
    history             = LossHistory()
    model.loss_history  = history

    best_monitor_loss   = float('inf')
    best_state          = None # 用于保存最佳模型权重
    patience_counter    = 0
    ema_monitor_loss    = None
    stop_training       = False

    MIN_RELATIVE_IMPROVEMENT = 0.001

    print(f"\n⏳ 开始训练 (Epochs: {train_cfg.epochs})..."
          f"Early-stop patience: {train_cfg.patience}...")
    
    # ————————Phase 1 初始化（可选）————————————————————————————————————
    if in_phase1:
        for name, p in model._denoise_fn.named_parameters():
            if 'num_col_encoders' in name or 'num_transformer' in name:
                p.requires_grad_(False)
        print(f" Phase 1 (前{PHASE1_EPOCHS} epochs): 冻结数值编码器，优先稳定类别表示")

    # =======================================================================
    # 训练主循环
    # =======================================================================
    for epoch in range(train_cfg.epochs):
        if stop_training:
            break

        # ———————— Phase 切换（可选）——————————————————————————————————————
        if in_phase1 and epoch == PHASE1_EPOCHS:
            for p in model._denoise_fn.parameters():
                p.requires_grad_(True)
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5 # 解冻后降低lr,防止震荡
            in_phase1 = False
            print(f" Phase 2 (Epoch {epoch+1}): 解冻所有参数，联合微调， lr x 0.5")

        # ————————训练阶段 （Training Phase）——————————————————————————————
        model._denoise_fn.train() 
        if marginal_tracker is not None:
            marginal_tracker.train()

        epoch_train_loss = 0.0
        epoch_train_loss_num = 0.0
        epoch_train_loss_cat = 0.0
        train_valid_batches = 0
        
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(model.device), batch_y.to(model.device)
            optimizer.zero_grad()
            
            loss, loss_num, loss_cat, _ = model.compute_loss(
                batch_x, batch_y, marginal_tracker=marginal_tracker)
            
            # ----- NaN/Inf 检测 -----
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"跳过 NaN 或 Inf 批次 (Epoch {epoch})")
                stop_training = True
                break
            
            loss.backward()
            # 检查反向传播梯度中是否含有NaN或Inf
            has_invalid_grad = any(
                param.grad is not None and not torch.isfinite(param.grad).all()
                for param in model._denoise_fn.parameters()
            )

            if has_invalid_grad:
                optimizer.zero_grad()
                print(f"Warning: 无效梯度，跳过本批次（Epoch {epoch + 1}")
                continue

            torch.nn.utils.clip_grad_norm_(model._denoise_fn.parameters(), 1.0)
            optimizer.step()
            
            epoch_train_loss        += loss.item()
            epoch_train_loss_num    += loss_num.item()
            epoch_train_loss_cat    += loss_cat.item()
            train_valid_batches     += 1
            
        if train_valid_batches == 0:
            if not stop_training:
                print(f"Warning: Epoch {epoch} 无有效训练批次。跳过。")
            continue

        avg_train_loss = epoch_train_loss    / train_valid_batches
        avg_train_num = epoch_train_loss_num / train_valid_batches
        avg_train_cat = epoch_train_loss_cat / train_valid_batches

        # ——————————验证阶段 （Validation Phase）————————————————————————
        avg_val_loss = None
        if val_dataloader is not None:
            model._denoise_fn.eval()
            if marginal_tracker is not None:
                marginal_tracker.eval()

            epoch_val_loss      = 0.0
            val_valid_batches   = 0

            with torch.no_grad():
                for batch_x_val, batch_y_val in val_dataloader:
                    batch_x_val, batch_y_val = batch_x_val.to(model.device), batch_y_val.to(model.device)
                    val_loss, _, _, _ = model.compute_loss(
                        batch_x_val, batch_y_val, marginal_tracker=marginal_tracker) 
                    
                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        print(f"跳过 NaN 或 Inf 验证批次 (Epoch {epoch + 1})") 
                        continue

                    epoch_val_loss += val_loss.item()
                    val_valid_batches += 1

            if val_valid_batches > 0:
                avg_val_loss = epoch_val_loss / val_valid_batches
            else:
                print(f"Warning: Epoch {epoch + 1} 无有效验证批次。")

        # ————————————记录损失历史 ——————————————————————————————————————————————————————————————
        history.append(avg_train_loss, avg_train_num, avg_train_cat, avg_val_loss, epoch)

        # ————————————EMA监控 ———————————————————————————————————————————————————————————————————
        raw_monitor = (avg_val_loss if (USE_VAL_MONITOR and avg_val_loss is not None) 
                       else avg_train_loss
        )

        ema_monitor_loss = (
            raw_monitor if ema_monitor_loss is None
            else EMA_ALPHA * raw_monitor + (1 - EMA_ALPHA) * ema_monitor_loss
        )
        current_monitor_loss = ema_monitor_loss

        # ———————————————— Checkpoint Ensemble 候选池更新——————————————————————————————————————————
        current_state = {k: v.cpu().clone() for k, v in model._denoise_fn.state_dict().items()}
        top_k_states.append((current_monitor_loss, current_state))
        top_k_states.sort(key=lambda x: x[0])
        if len(top_k_states) > ENSEMBLE_K:
            top_k_states.pop() #移除最差的， 保留最好的 K 个

        # ———————————————— 早停判断—————————————————————————————————————————————————————————————————
        if current_monitor_loss < best_monitor_loss * (1- MIN_RELATIVE_IMPROVEMENT): # 引入一个小的余量，避免过于敏感
            best_monitor_loss = current_monitor_loss
            best_state        = current_state
            history.set_best(epoch, best_monitor_loss)
            patience_counter = 0
        else:
            patience_counter += 1

        # ————————Scheduler 步进 （每epoch一次）——————————————————————————————————————————————————————
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss if avg_val_loss is not None else avg_train_loss)
        else:
            scheduler.step()

        #scheduler.step()
        
        # ———————— 定期打印 ————————————————————————————————————————————————————————————————————————————
        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_str = (
                f"Epoch [{epoch+1:>4}/{train_cfg.epochs}] "
                f"Train Tot: {avg_train_loss:.4f}  "
                f"(Num: {avg_train_num:.4f}, Cat: {avg_train_cat:.4f})"
                f"lr: {current_lr}"
            )
            if avg_val_loss is not None:
                log_str += f" | Val Tot: {avg_val_loss:.4f}"
            log_str += (
                f" | Monitor(EMA): {current_monitor_loss:.4f} " # EMA值
                f" | Raw: {raw_monitor: .4f}"  # 原始值
                f"| Best Mon: {best_monitor_loss:.4f} "
                f"| Patience: {patience_counter}/{train_cfg.patience}"
            )
            
            print(log_str)
            
        if patience_counter >= train_cfg.patience:
            print(f"⏹️  Early stopping @ Epoch {epoch + 1}  "
                  f"(最优: Epoch {history.best_epoch + 1}, "
                  f"监控损失 {history.best_monitor_loss:.4f})"
                  )
            break
    
    # =======================================================================
    # 训练结束，权重恢复
    # =======================================================================
    if is_minority and len(top_k_states) > 1:
        # minority 模式：对最优K个checkpoint做加权平均
        print(f"🔀 对 {len(top_k_states)} 个最优 checkpoint 做权重平均...")
        avg_state = {}
        weights = [1.0 / (s[0] + 1e-8) for s in top_k_states]
        total_w = sum(weights)
        for key in top_k_states[0][1].keys():
            # 按monitor_loss的倒数加权（Loss越小权重越大 ）
            avg_state[key] = sum(
                w / total_w * s[1][key].float()
                for w, s in zip(weights, top_k_states)
                )
        model._denoise_fn.load_state_dict(avg_state)
        print(f"✅ 权重平均完成")
    elif best_state is not None:
        model._denoise_fn.load_state_dict(best_state)
        print(f"🔁 已恢复 Epoch {history.best_epoch + 1} 的最优权重")
            
    print(f"\n✅ 训练完成 | {history.summary()}")