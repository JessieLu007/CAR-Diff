# core/diff_training.py
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


# ----------------------------------------------------------------
# Loss 历史记录器
# ----------------------------------------------------------------

class LossHistory:
    """轻量loss记录容器，支持绘图与保存, 新增对验证损失的记录和最佳模型追踪"""

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
            current_monitor_loss = val_total
        else:# 若没有验证集，则使用训练损失作为监控损失
            current_monitor_loss = train_total

        if current_monitor_loss < self.best_monitor_loss:
            self.best_monitor_loss = current_monitor_loss
            self.best_epoch = epoch
 
    def plot(self, save_path: str | None = None, show: bool = False):
        """
        绘制 loss 曲线并保存（默认不弹窗，适合服务器环境）。
 
        :param save_path: 图片保存路径，None 则自动命名
        :param show:      是否调用 plt.show()（本地调试时设为 True）
        """
        if not self.train_total:
            print("⚠️  没有可绘制的 loss 数据")
            return
 
        epochs = list(range(1, len(self.train_total) + 1))
 
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training and Validation Loss Curves", fontsize=13, fontweight='bold')
 
        # 左图：总 loss
        ax = axes[0]
        ax.plot(epochs, self.train_total, color='#2c7bb6', linewidth=1.5, label='Total Loss')
        if self.val_total: # 如果有验证损失，则绘制
            val_epochs = list(range(1, len(self.val_total) + 1))
            ax.plot(val_epochs, self.val_total, color='#ff7f0e', linewidth=1.5, linestyle='--', label='Validation Total Loss')
        
        # 指示最佳 epoch （基于监控损失）
        ax.axvline(self.best_epoch + 1, color='#d7191c', linestyle='--',
                   linewidth=1, label=f'Best Epoch ({self.best_epoch + 1})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
 
        # 右图：num / cat 分量
        ax = axes[1]
        ax.plot(epochs, self.train_num, color='#1a9641', linewidth=1.2, label='Numerical Loss')
        ax.plot(epochs, self.train_cat, color='#fdae61', linewidth=1.2, label='Categorical Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Numerical vs Categorical Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
 
        plt.tight_layout()
 
        if save_path is None:
            ts = datetime.now().strftime("%m%d_%H%M%S")
            save_path = f"loss_curve_{ts}.png"
 
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📈 Loss 曲线已保存至: {save_path}")
 
        if show:
            plt.show()
        plt.close(fig)
 
    def summary(self) -> str:
        if not self.train_total:
            return "（无训练记录）"
        return (
            f"总 Epoch: {len(self.train_total)} | "
            f"最优监控: {self.best_monitor_loss:.4f} (Epoch {self.best_epoch + 1}) | "
            f"末尾训练Loss: {self.train_total[-1]:.4f}"
        )

# --- 为 train_loop 增加val_dataloader 参数
def train_loop(model, train_dataloader, val_dataloader, train_cfg):
    """
    扩散模型训练主循环。
 
    接口不变：train_loop(model, dataloader, train_cfg)
    新增：
      - model.loss_history (LossHistory)  训练后可调用 .plot() 绘图
      - 每 50 epoch 打印 num/cat 分量
      - 基于验证损失的早停和最佳模型保存
    """
    optimizer = torch.optim.AdamW(
        list(model._denoise_fn.parameters()) + [model.log_vars], 
        lr=train_cfg.lr, 
        weight_decay=1e-5)

    """ den = model._denoise_fn
    
    optimizer = torch.optim.AdamW(
        [
            {'params': den.time_embed.parameters(),  'lr': train_cfg.lr},
            {'params': den.label_emb.parameters(),   'lr': train_cfg.lr},
            # input projection
            {'params': den.proj_num.parameters(), 'lr': train_cfg.lr},
            {'params': den.proj_cat.parameters(), 'lr': train_cfg.lr},
            {'params': den.proj_fuse.parameters(), 'lr': train_cfg.lr},
            # backbone
            {'params': den.layers.parameters(),      'lr': train_cfg.lr},
            # output
            {'params': den.out_num.parameters(),     'lr': train_cfg.lr},
            {'params': den.out_cat.parameters(),     'lr': train_cfg.lr * 1.0},
        ],
        weight_decay=1e-4,
    ) """

    # 挂载 loss历史到model， 方便外部访问
    history = LossHistory()
    model.loss_history = history

    best_monitor_loss = float('inf')
    best_state = None # 用于保存最佳模型权重
    patience_counter = 0

    print(f"\n⏳ 开始训练 (Epochs: {train_cfg.epochs})..."
          f"Early-stop patience: {train_cfg.patience}...")
    model._denoise_fn.train()

    stop_training = False
    for epoch in range(train_cfg.epochs):
        if stop_training:
            break

        # -------------------------------
        # 训练阶段 （Training Phase）
        # -------------------------------
        model._denoise_fn.train() # 设置为训练模式
        epoch_train_loss = 0.0
        epoch_train_loss_num = 0.0
        epoch_train_loss_cat = 0.0
        train_valid_batches = 0
        
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(model.device), batch_y.to(model.device)
            optimizer.zero_grad()
            
            loss, loss_num, loss_cat, _ = model.compute_loss(batch_x, batch_y)
            
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
                print(f"Warning: Invalid gradient detected at Epoch {epoch}, skipping gradient update for this batch.")
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

        avg_train_loss = epoch_train_loss / train_valid_batches
        avg_train_num = epoch_train_loss_num / train_valid_batches
        avg_train_cat = epoch_train_loss_cat / train_valid_batches

        # -------------------------------
        # 验证阶段 （Validation Phase）
        # -------------------------------
        avg_val_loss = None
        if val_dataloader is not None:
            model._denoise_fn.eval() # 设置为评估模式
            epoch_val_loss = 0.0
            val_valid_batches = 0
            with torch.no_grad(): # 禁用梯度计算
                for batch_x_val, batch_y_val in val_dataloader:
                    batch_x_val, batch_y_val = batch_x_val.to(model.device), batch_y_val.to(model.device)
                    val_loss, _, _, _ = model.compute_loss(batch_x_val, batch_y_val) # 在验证数据上计算损失
                    if torch.isnan(val_loss) or torch.isinf(val_loss):
                        print(f"跳过 NaN 或 Inf 验证批次 (Epoch {epoch})") 
                        continue
                    epoch_val_loss += val_loss.item()
                    val_valid_batches += 1

            if val_valid_batches > 0:
                avg_val_loss = epoch_val_loss / val_valid_batches
            else:
                print(f"Warning: Epoch {epoch} 无有效验证批次。avg_val_loss 将为 None。")

        # ----- 记录损失历史 -----
        history.append(avg_train_loss, avg_train_num, avg_train_cat, avg_val_loss, epoch)

        # ----- 早期停止逻辑 (现在基于验证损失，若无则退化为训练损失) -----
        current_monitor_loss = avg_val_loss if avg_val_loss is not None else avg_train_loss

        if current_monitor_loss < best_monitor_loss - 1e-4: # 引入一个小的余量，避免过于敏感
            best_monitor_loss = current_monitor_loss
            best_epoch_current_run = epoch # 记录当前运行中的最佳 epoch
            # 保存 denoise_fn的状态字典，而不是整个模型
            best_state = {k: v.cpu().clone() for k, v in model._denoise_fn.state_dict().items()}
            patience_counter = 0
            #print(f"  --> 监控损失改善！保存最佳模型. ({best_monitor_loss:.4f})")
        else:
            patience_counter += 1
            #print(f"  --> 监控损失未改善 ({current_monitor_loss:.4f}). Patience: {patience_counter}/{train_cfg.patience}")
        
        # ----- 定期打印 -----
        if (epoch + 1) % 50 == 0 or epoch == 0:
            log_str = (
                f"Epoch [{epoch+1:>4}/{train_cfg.epochs}] "
                f"Train Tot: {avg_train_loss:.4f}  "
                f"(Num: {avg_train_num:.4f}, Cat: {avg_train_cat:.4f})"
            )
            if avg_val_loss is not None:
                log_str += f" | Val Tot: {avg_val_loss:.4f}"
            log_str += (
                f" | Monitor Loss: {current_monitor_loss:.4f} "
                f"| Best Mon: {best_monitor_loss:.4f} "
                f"| Patience: {patience_counter}/{train_cfg.patience}"
            )
            print(log_str)
            
        if patience_counter >= train_cfg.patience:
            print(f"⏹️  Early stopping @ Epoch {epoch + 1}  "
                  f"(最优: Epoch {history.best_epoch + 1}, 监控损失 {history.best_monitor_loss:.4f})")
            break

    if best_state is not None:
        model._denoise_fn.load_state_dict(best_state)
        print(f"🔁 已恢复 Epoch {history.best_epoch + 1} 的最优权重")
            
    # ── 训练结束摘要 ──
    print(f"\n✅ 训练完成 | {history.summary()}")