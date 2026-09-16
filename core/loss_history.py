import csv
import math
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# ----------------------------------------------------------------
# Loss 历史记录器
# ----------------------------------------------------------------
class LossHistory:
    """轻量loss记录容器，新增对验证损失的记录和最佳模型追踪（按 step 记录）"""

    def __init__(self):
        self.steps: list[int] = []        # 新增：每条记录对应的真实 step 数，而不是记录点序号
        self.train_total: list[float] = []
        self.train_num: list[float] = []
        self.train_cat: list[float] = []
        self.val_total: list[float] = []  # 与 steps 严格等长，没有验证值的位置填 NaN（而不是跳过不记）
        self.best_monitor_loss: float = float('inf')  # 追踪用于早停的最佳损失（通常是验证损失）
        self.best_step: int = 0

    def append(self, train_total: float, train_num: float, train_cat: float,
               val_total: float | None, step: int):
        self.steps.append(step)
        self.train_total.append(train_total)
        self.train_num.append(train_num)
        self.train_cat.append(train_cat)
        self.val_total.append(val_total if val_total is not None else math.nan)

    def set_best(self, step: int, monitor_loss: float) -> None:
        self.best_step        = step
        self.best_monitor_loss = monitor_loss

    def to_csv(self, path: str) -> None:
        """导出为CSV，字段：step, train_total, train_num, train_cat, val_total（缺失记为空）"""
        if not self.steps:
            print("没有可导出的loss数据")
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_total", "train_num", "train_cat", "val_total"])
            for i, step in enumerate(self.steps):
                val = "" if math.isnan(self.val_total[i]) else self.val_total[i]
                writer.writerow([step, self.train_total[i], self.train_num[i], self.train_cat[i], val])
        print(f"Loss历史已导出至：{path}")

    def plot(self, save_path: str | None = None, show: bool = False) -> None:
        """绘制loss曲线并保存，默认不弹窗（横轴为真实 step 数）"""
        if not self.train_total:
            print("没有可绘制的loss数据")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Training and Validation Loss Curves", fontsize=13, fontweight='bold')

        # 左图： 总 loss
        ax = axes[0]
        ax.plot(self.steps, self.train_total, color='#2c7bb6', linewidth=1.5, label='Train Total')
        if any(not math.isnan(v) for v in self.val_total):
            # NaN 会被 matplotlib 自动处理为断线，不需要单独筛选长度
            ax.plot(self.steps, self.val_total, color='#ff7f0e', linewidth=1.5, linestyle='--', label='Val Total')
        ax.axvline(self.best_step, color='#d7191c', linestyle='--', linewidth=1, label=f'Best Step ({self.best_step})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 右图： num / cat 分量
        ax = axes[1]
        ax.plot(self.steps, self.train_num, color='#1a9641', linewidth=1.2, label='Numerical Loss')
        ax.plot(self.steps, self.train_cat, color='#fdae61', linewidth=1.2, label='Categorical Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Numerical vs Categorical Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout

        if save_path is None:
            ts        = datetime.now().strftime("%m%d_%H%M%S")
            save_path = f"loss_curve_{ts}.png"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Loss曲线已保存至：{save_path}")

        if show:
            plt.show()
        plt.close(fig)

    def summary(self) -> str:
        if not self.train_total:
            return "（无训练记录）"
        return (
            f"总记录点数: {len(self.train_total)} | "
            f"最优监控: {self.best_monitor_loss:.4f} (Step {self.best_step}) | "
            f"末尾训练Loss: {self.train_total[-1]:.4f}"
        )