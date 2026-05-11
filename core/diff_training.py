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
    """轻量loss记录容器，支持绘图与保存"""

    def __init__(self):
        self.total: list[float] = []
        self.num: list[float] = []
        self.cat: list[float] = []
        self.best_loss: float = float('inf')
        self.best_epoch: int = 0

    def append(self, total: float, num: float, cat: float, epoch: int):
        self.total.append(total)
        self.num.append(num)
        self.cat.append(cat)
        if total < self.best_loss:
            self.best_loss = total
            self.best_epoch = epoch
 
    def plot(self, save_path: str | None = None, show: bool = False):
        """
        绘制 loss 曲线并保存（默认不弹窗，适合服务器环境）。
 
        :param save_path: 图片保存路径，None 则自动命名
        :param show:      是否调用 plt.show()（本地调试时设为 True）
        """
        if not self.total:
            print("⚠️  没有可绘制的 loss 数据")
            return
 
        epochs = list(range(1, len(self.total) + 1))
 
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training Loss Curves", fontsize=13, fontweight='bold')
 
        # 左图：总 loss
        ax = axes[0]
        ax.plot(epochs, self.total, color='#2c7bb6', linewidth=1.5, label='Total Loss')
        ax.axvline(self.best_epoch + 1, color='#d7191c', linestyle='--',
                   linewidth=1, label=f'Best Epoch ({self.best_epoch + 1})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
 
        # 右图：num / cat 分量
        ax = axes[1]
        ax.plot(epochs, self.num, color='#1a9641', linewidth=1.2, label='Numerical Loss')
        ax.plot(epochs, self.cat, color='#fdae61', linewidth=1.2, label='Categorical Loss')
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
        if not self.total:
            return "（无训练记录）"
        return (
            f"总 Epoch: {len(self.total)} | "
            f"最优 Loss: {self.best_loss:.4f} (Epoch {self.best_epoch + 1}) | "
            f"末尾 Loss: {self.total[-1]:.4f}"
        )

def train_loop(model, dataloader, train_cfg):
    """
    扩散模型训练主循环。
 
    接口不变：train_loop(model, dataloader, train_cfg)
    新增：
      - model.loss_history (LossHistory)  训练后可调用 .plot() 绘图
      - 每 50 epoch 打印 num/cat 分量
      - Early stopping 提示包含最优 epoch 信息
    """
    #optimizer = torch.optim.AdamW(model._denoise_fn.parameters(), lr=train_cfg.lr, weight_decay=1e-4)

    den = model._denoise_fn

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
    )

    # 挂载 loss历史到model， 方便外部访问
    history = LossHistory()
    model.loss_history = history

    best_loss = float('inf')
    patience_counter = 0

    print(f"\n⏳ 开始训练 (Epochs: {train_cfg.epochs})..."
          f"Early-stop patience: {train_cfg.patience}...")
    model._denoise_fn.train()

    stop_training = False
    for epoch in range(train_cfg.epochs):
        if stop_training:
            break

        epoch_loss = 0.0
        epoch_loss_num = 0.0
        epoch_loss_cat = 0.0
        valid_batches = 0
        
        for batch_x, batch_y in dataloader:
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
                continue

            torch.nn.utils.clip_grad_norm_(model._denoise_fn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss      += loss.item()
            epoch_loss_num  += loss_num.item()
            epoch_loss_cat  += loss_cat.item()
            valid_batches   += 1
            
        if valid_batches == 0: continue
        avg_loss = epoch_loss / valid_batches
        avg_num = epoch_loss_num / valid_batches
        avg_cat = epoch_loss_cat / valid_batches

        # ----- 记录 epoch -----
        history.append(avg_loss, avg_num, avg_cat, epoch)

        # ----- early stopping -----
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model._denoise_fn.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # ----- 定期打印 -----
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1:>4}/{train_cfg.epochs}] "
                f"Tot: {avg_loss:.4f}  "
                f"(Num: {avg_num:.4f}, Cat: {avg_cat:.4f})  "
                f"Best: {best_loss:.4f}  "
                f"Patience: {patience_counter}/{train_cfg.patience}"
            )
            
        if patience_counter >= train_cfg.patience:
            print(f"⏹️  Early stopping @ Epoch {epoch + 1}  "
                  f"(最优: Epoch {history.best_epoch + 1}, Loss {history.best_loss:.4f})")
            break

    if best_state is not None:
        model._denoise_fn.load_state_dict(best_state)
        print(f"🔁 已恢复 Epoch {best_epoch + 1} 的最优权重")
            
    # ── 训练结束摘要 ──
    print(f"\n✅ 训练完成 | {history.summary()}")