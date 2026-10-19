# core/diff_training.py
import math
import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LambdaLR

from .loss_history import LossHistory


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int,
    total_steps:  int, min_lr_ratio: float = 0.05,) -> LambdaLR:
    def lr_lambda(step) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda)


def smooth_labels(y: torch.Tensor, num_classes: int, smooth_prob: float = 0.05) -> torch.Tensor:
    mask     = torch.rand_like(y.float()) < smooth_prob
    random_y = torch.randint_like(y, low=0, high=num_classes)
    return torch.where(mask, random_y, y)


def train_loop(model, train_dataloader, val_dataloader, train_cfg) -> None:
    task_name = train_cfg.task
    is_minority = task_name in ('minority_uncond', 'minority_cond')

    # 两阶段训练开关
    use_two_phase = getattr(train_cfg, 'use_two_phase', False)
    PHASE1_STEPS  = getattr(train_cfg, 'phase1_steps', 5000)
    in_phase1     = use_two_phase

    # Checkpoint Ensemble 设置
    ENSEMBLE_K = getattr(train_cfg, 'ensemble_k', 5) if is_minority else 1
    top_k_states = []  # list of (monitor_loss, state_dict)

    # NaN/Inf 检查开关（速度相关）
    ENABLE_NAN_CHECK = getattr(train_cfg, 'enable_nan_check', False)

    ENABLE_AMP = getattr(train_cfg, 'enable_amp', False)
    if ENABLE_AMP:
        print("ℹ️  已启用 AMP(bfloat16)混合精度训练")

    compiled_compute_loss = model.compute_loss
    if getattr(train_cfg, 'enable_compile', False):
        print("ℹ️  已启用 torch.compile(mode='reduce-overhead')，前几步会有编译预热开销")
        compiled_compute_loss = torch.compile(model.compute_loss, mode="reduce-overhead")

    steps_per_epoch = len(train_dataloader.dataset) // train_dataloader.batch_size
    if getattr(train_cfg, 'total_steps', None) is not None:
        total_steps = train_cfg.total_steps
    else:
        total_steps = max(getattr(train_cfg, 'step_floor', 0), 
                          train_cfg.target_epochs * steps_per_epoch)
    print(f"ℹ️  steps_per_epoch={steps_per_epoch} → total_steps={total_steps}"
          f"（≈{total_steps/steps_per_epoch:.1f} epoch）")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model._denoise_fn.parameters()) + [model.log_vars],
        lr=train_cfg.lr, weight_decay=1e-4 if is_minority else 1e-5, fused=(model.device.type=="cuda"))

    # Scheduler
    default_warmup_steps = 500 if is_minority else 1000
    warmup_steps = getattr(train_cfg, 'warmup_steps', default_warmup_steps)
    warmup_steps = min(warmup_steps, max(1, total_steps // 5))  # 避免 warmup 吃掉小数据集的大部分预算

    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps = warmup_steps,
        total_steps  = total_steps, min_lr_ratio = 0.05,)

    eval_every = getattr(train_cfg, 'eval_every', None)
    if eval_every is None:
        eval_every = max(1, total_steps // 30)   # 约30个评估点，随数据集自动适配
    log_every  = getattr(train_cfg, 'log_every', eval_every)

    # 验证集监控策略
    val_size = sum(len(b[0]) for b in val_dataloader) if val_dataloader else 0

    if val_dataloader is not None and val_size > 0:
        USE_VAL_MONITOR = True
        if val_size < 200:
            EMA_ALPHA = 0.25
            print(f"⚠️  验证集仅 {val_size} 条，强制使用 val 监控，EMA 平滑加强至 {EMA_ALPHA}")
        else:
            EMA_ALPHA = 0.1
    else:
        USE_VAL_MONITOR = False
        EMA_ALPHA = 0.1

    # ------------------------------------------------------------------
    # DataLoader 配置提示（速度排查用，不影响训练）
    # ------------------------------------------------------------------
    _nw   = getattr(train_dataloader, 'num_workers', 'N/A')
    _pin  = getattr(train_dataloader, 'pin_memory', 'N/A')
    _bsz  = getattr(train_dataloader, 'batch_size', 'N/A')
    print(f"🔧 DataLoader 配置: num_workers={_nw}, pin_memory={_pin}, batch_size={_bsz}")
    if _nw == 0:
        print("   ⚠️  num_workers=0，数据加载会阻塞 GPU，建议改为 2~4")
    if _pin is False:
        print("   ⚠️  pin_memory=False，H2D 拷贝无法与计算重叠，建议改为 True")

    # 训练状态初始化
    history             = LossHistory()
    model.loss_history  = history
    best_monitor_loss   = float('inf')
    best_state          = None
    best_monitor_step   = None
    patience_counter    = 0             # 距离最近一次“改善”过了多少个 eval 间隔
    ema_monitor_loss    = None
    stop_training       = False

    MIN_RELATIVE_IMPROVEMENT = 0.001

    patience_steps = train_cfg.patience
    USE_EARLY_STOP = getattr(train_cfg, 'use_early_stop', True)

    print(f"\n⏳ 开始训练 (Total steps: {total_steps}, 每 {eval_every} step 验证/记录一次)...")
    if not ENABLE_NAN_CHECK:
        print("ℹ️  NaN/Inf 检查已关闭（如需开启设置 train_cfg.enable_nan_check=True）")

    # Phase 1 初始化（可选）
    if in_phase1:
        for name, p in model._denoise_fn.named_parameters():
            if 'cat_col_encoders' in name or 'cat_transformer' in name:
                p.requires_grad_(False)
        print(f" Phase 1 (前{PHASE1_STEPS} steps): 冻结分类编码器，优先稳定数值表示")

    model._denoise_fn.train()

    train_iter = iter(train_dataloader)

    running_loss     = torch.zeros((), device=model.device)
    running_loss_num = torch.zeros((), device=model.device)
    running_loss_cat = torch.zeros((), device=model.device)
    running_batches  = 0    

    for step in range(1, total_steps + 1):
        if stop_training:
            break

        # Phase 切换
        if in_phase1 and step == PHASE1_STEPS:
            for p in model._denoise_fn.parameters():
                p.requires_grad_(True)
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.75
            in_phase1 = False
            print(f" Phase 2 (Step {step}): 解冻所有参数，联合微调， lr x 0.75")

        try:
            batch_x, batch_y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch_x, batch_y = next(train_iter)

        batch_x = batch_x.to(model.device, non_blocking=True)
        batch_y = batch_y.to(model.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if ENABLE_AMP else nullcontext()
        with amp_ctx:
            loss, loss_num, loss_cat = compiled_compute_loss(
                batch_x, batch_y)

        if ENABLE_NAN_CHECK:
            if not torch.isfinite(loss).item():
                print(f"跳过 NaN 或 Inf 批次 (Step {step})")
                stop_training = True
                break

        loss.backward()

        if ENABLE_NAN_CHECK:
            grads_finite = torch.ones((), dtype=torch.bool, device=model.device)
            for param in model._denoise_fn.parameters():
                if param.grad is not None:
                    grads_finite = grads_finite & torch.isfinite(param.grad).all()
            if not grads_finite.item():
                optimizer.zero_grad(set_to_none=True)
                print(f"Warning: 无效梯度，跳过本批次（Step {step}）")
                continue

        torch.nn.utils.clip_grad_norm_(model._denoise_fn.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss     += loss.detach()
        running_loss_num += loss_num.detach()
        running_loss_cat += loss_cat.detach()
        running_batches  += 1

        if step % eval_every != 0:
            continue
        if running_batches == 0:
            print(f"Warning: Step {step} 附近无有效训练批次，跳过本次验证。")
            continue

        avg_train_loss = (running_loss     / running_batches).item()
        avg_train_num  = (running_loss_num / running_batches).item()
        avg_train_cat  = (running_loss_cat / running_batches).item()
        running_loss     = torch.zeros((), device=model.device)
        running_loss_num = torch.zeros((), device=model.device)
        running_loss_cat = torch.zeros((), device=model.device)
        running_batches  = 0

        # 验证阶段
        avg_val_loss = None
        if USE_VAL_MONITOR and val_dataloader is not None:
            model._denoise_fn.eval()
            epoch_val_loss    = None
            val_valid_batches = 0

            with torch.no_grad():
                for batch_x_val, batch_y_val in val_dataloader:
                    batch_x_val = batch_x_val.to(model.device, non_blocking=True)
                    batch_y_val = batch_y_val.to(model.device, non_blocking=True)
                    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if ENABLE_AMP else nullcontext()
                    with amp_ctx:
                        val_loss, _, _ = compiled_compute_loss(
                            batch_x_val, batch_y_val)
                    if ENABLE_NAN_CHECK:
                        if not torch.isfinite(val_loss).item():
                            print(f"跳过 NaN 或 Inf 验证批次 (Step {step})")
                            continue
                    if epoch_val_loss is None:
                        epoch_val_loss = val_loss.detach()
                    else:
                        epoch_val_loss = epoch_val_loss + val_loss.detach()
                    val_valid_batches += 1

            if val_valid_batches > 0:
                avg_val_loss = (epoch_val_loss / val_valid_batches).item()
            else:
                print(f"Warning: Step {step} 无有效验证批次。")

            model._denoise_fn.train()

        history.append(avg_train_loss, avg_train_num, avg_train_cat, avg_val_loss, step)
        raw_monitor = (avg_val_loss if (USE_VAL_MONITOR and avg_val_loss is not None) else avg_train_loss)
        ema_monitor_loss = (raw_monitor if ema_monitor_loss is None
            else EMA_ALPHA * raw_monitor + (1 - EMA_ALPHA) * ema_monitor_loss)
        current_monitor_loss = ema_monitor_loss

        if not math.isfinite(best_monitor_loss):
            is_better_than_best = True
        else:
            is_better_than_best = (
                current_monitor_loss
                < best_monitor_loss - MIN_RELATIVE_IMPROVEMENT * abs(best_monitor_loss))

        top_k_needs_update = (
            len(top_k_states) < ENSEMBLE_K or current_monitor_loss < top_k_states[-1][0])

        current_state = None
        if top_k_needs_update or is_better_than_best:
            current_state = {
                k: v.detach().cpu().clone()
                for k, v in model._denoise_fn.state_dict().items()}

        if top_k_needs_update:
            top_k_states.append((current_monitor_loss, current_state))
            top_k_states.sort(key=lambda x: x[0])
            if len(top_k_states) > ENSEMBLE_K:
                top_k_states.pop()

        if is_better_than_best:
            best_monitor_loss = current_monitor_loss
            best_monitor_step = step
            best_state        = current_state
            history.set_best(step, best_monitor_loss)
            patience_counter = 0
        else:
            patience_counter += 1

        if step % log_every == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_str = (
                f"Step [{step:>6}/{total_steps}] "
                f"Train Tot: {avg_train_loss:.4f}  "
                f"(Num: {avg_train_num:.4f}, Cat: {avg_train_cat:.4f}) "
                f"lr: {current_lr}"
            )
            if avg_val_loss is not None:
                log_str += f" | Val Tot: {avg_val_loss:.4f}"
            log_str += (
                f" | Monitor(EMA): {current_monitor_loss:.4f} "
                f" | Raw: {raw_monitor: .4f}"
                f" | Best Mon: {best_monitor_loss:.4f} "
                f"| Patience: {patience_counter}/{patience_steps}"
            )
            print(log_str)

        if USE_EARLY_STOP and patience_counter >= patience_steps:
            print(f"⏹️  Early stopping @ Step {step}  "
                  f"(最优: Step {best_monitor_step}, "
                  f"监控损失 {best_monitor_loss:.4f}, "
                  f"已 {patience_counter} 个评估间隔未改善)")
            stop_training = True

    if is_minority and len(top_k_states) > 1:
        print(f"🔀 对 {len(top_k_states)} 个最优 checkpoint 做权重平均...")
        avg_state = {}
        weights = [1.0 / (s[0] + 1e-8) for s in top_k_states]
        total_w = sum(weights)
        for key in top_k_states[0][1].keys():
            avg_state[key] = sum(
                w / total_w * s[1][key].float()
                for w, s in zip(weights, top_k_states)
            )
        model._denoise_fn.load_state_dict(avg_state)
        print(f"✅ 权重平均完成")
    elif USE_EARLY_STOP and best_state is not None:
        model._denoise_fn.load_state_dict(best_state)
        print(f"🔁 已恢复 Step {best_monitor_step} 的最优权重")

    print(f"\n✅ 训练完成 | {history.summary()}")
