# experiments/base_synthesizer.py
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class TargetDistInfo:
    target_class: int
    target_count: int
    background_count: int
    total: int
    IR: float

class BaseSynthesizer:
    """
    生成算法统一接口：
      - fit(df_train): 拟合生成器
      - generate(strategy, target_class, manual_n): 按策略生成数据
      - get_config_dict(): 返回当前实验配置（用于保存）
      - save/load: 可选，用于持久化模型
    """

    def __init__(self, target_col: str, cat_cols: list[str], training_cfg=None, sampling_cfg=None):
        self.target_col = target_col
        self.cat_cols = cat_cols
        self.training_cfg = training_cfg # 包含task和objective
        self.sampling_cfg = sampling_cfg # num_samples等参数
        self.target_dist: TargetDistInfo | None = None        

    # ========= 公共 API =========
    def fit(self, df_train: pd.DataFrame):
        """
        统一外壳：
          1. 统计 label=1 的分布信息
          2. 调用子类的 _fit_algorithm(df_train)
        """
        counts = df_train[self.target_col].value_counts()
        total = len(df_train)

        # 这里约定 label=1 为“目标类/病人类”，与数量多少无关
        target_class = 1
        target_count = int(counts.get(target_class, 0))
        background_count = int(total - target_count)
        ir = (background_count / target_count) if target_count > 0 else float('inf')

        self.target_dist = TargetDistInfo(
            target_class=target_class,
            target_count=target_count,
            background_count=background_count,
            total=total,
            IR=ir,
        )

        self._fit_algorithm(df_train)

    def _decide_n_samples(self, target_class: int) -> int:
        """根据 task + objective + 训练集分布自动决定生成数量"""
        assert self.target_dist is not None, "请先 fit() 再 generate()"

        task = getattr(self.training_cfg, "task", "full_data")
        objective = getattr(self.training_cfg, "objective", "fidelity")

        t_count = self.target_dist.target_count
        b_count = self.target_dist.background_count
        total = self.target_dist.total
        ir = self.target_dist.IR

        if task == "full_data":
            # 学整表：默认生成与训练集同规模的数据
            return total

        if task in ("minority_uncond", "minority_cond"):
            if objective == "fidelity":
                # 拟真：生成一个与真实少数类规模相近的集合
                return t_count if t_count > 0 else 1
            else:  # 'clf_boost'
                # 分类增强：少数类多生成一些，例如补齐到接近多数类，或者翻倍
                if t_count == 0:
                    return 1
                # 方案1：补到与多数类相当
                if ir >= 2:
                    return max(b_count - t_count, 1)
                else:
                    # 方案2：直接翻倍少数类数量
                    return 2 * t_count
        # 兜底
        return total

    def generate(self, target_class: int = 1) -> pd.DataFrame:
        """
        生成数量优先级：
        - 若 sampling_cfg.num_samples不为None, 优先用它
        - 否则用_decide_n_samples()自动计算
        """
        samp_cfg = self.sampling_cfg
        if samp_cfg is not None and getattr(samp_cfg, "num_samples", None) is not None:
            n_samples = samp_cfg.num_samples
        else:
            n_samples = self._decide_n_samples(target_class)

        print(f"   ⚙️ task={self.training_cfg.task}, "
            f"objective={self.training_cfg.objective}, "
            f"target={target_class}, 数量: {n_samples}")
        return self._generate_algorithm(n_samples, target_class)


    def get_config_dict(self) -> dict:
        """默认返回算法名称；具体子类可以扩展"""
        return {"algorithm": self.__class__.__name__}

    # ========= 可选：模型持久化 =========
    def save(self, path: str):
        """子类可按需实现：保存模型权重/processor等"""
        raise NotImplementedError

    def load(self, path: str):
        """子类可按需实现：从磁盘加载模型"""
        raise NotImplementedError

    # ========= 需子类实现的内部方法 =========
    def _fit_algorithm(self, df_train: pd.DataFrame):
        raise NotImplementedError

    def _generate_algorithm(self, n_samples: int, target_class: int) -> pd.DataFrame:
        raise NotImplementedError
