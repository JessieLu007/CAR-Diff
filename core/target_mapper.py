# core/target_mapper.py
import numpy as np
import pandas as pd


class TargetMapper:
    """
    类别标签映射：原始标签（str/int/任意 hashable）↔ 0..K-1 整数。
    与 TabularDataProcessor 平行：fit / transform / inverse_transform 三件套。

    用途：minority_cond / minority_uncond 模式下，target 从特征矩阵里剥离后，
    y 张量需要整数化；生成结果若要把 target 还原成原始标签，用它反变换。
    """

    def __init__(self, class_order: list | None = None):
        """
        class_order: 显式指定类序（跨 fold / 跨 run 保持一致时用）。
                     None 则 fit 时按 "数值优先、字符串兜底" 排序。
        """
        self.class_order_ = list(class_order) if class_order is not None else None
        self.mapping_ = (
            {c: i for i, c in enumerate(self.class_order_)} if class_order else None
        )

    # ---------------------------------------------------------
    def fit(self, y):
        s = pd.Series(y)
        if s.isna().any():
            raise ValueError(f"target 含缺失值，请先处理: {s[s.isna()].index.tolist()[:5]} ...")

        if self.class_order_ is None:
            unique = s.unique()
            try:
                # 先尝试按数值排（兼容 int / 数字字符串）
                self.class_order_ = sorted(unique, key=lambda v: (0, float(v)))
            except (ValueError, TypeError):
                # 兜底按字符串排，保证稳定
                self.class_order_ = sorted(unique, key=str)
            self.mapping_ = {c: i for i, c in enumerate(self.class_order_)}
        else:
            unknown = set(s.unique()) - set(self.mapping_.keys())
            if unknown:
                raise ValueError(f"发现未在 class_order 里的类: {unknown}")
        return self

    # ---------------------------------------------------------
    def transform(self, y) -> np.ndarray:
        s = pd.Series(y)
        mapped = s.map(self.mapping_)
        if mapped.isna().any():
            bad = s[mapped.isna()].unique().tolist()
            raise ValueError(f"transform 遇到未注册的类: {bad}")
        return mapped.astype("int64").to_numpy()

    def inverse_transform(self, y_int) -> np.ndarray:
        inv = {v: k for k, v in self.mapping_.items()}
        arr = np.asarray(y_int).ravel().astype(int)
        return np.array([inv[v] for v in arr], dtype=object)

    # ---------------------------------------------------------
    @property
    def classes_(self):
        return np.array(self.class_order_, dtype=object)

    @property
    def num_classes_(self) -> int:
        return len(self.class_order_)

    def to_dict(self) -> dict:
        """落盘用"""
        def _to_py(x):
            # numpy 标量 → Python 原生类型
            if isinstance(x, np.integer):
                return int(x)
            if isinstance(x, np.floating):
                return float(x)
            if isinstance(x, np.bool_):
                return bool(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        class_order = [_to_py(c) for c in self.class_order_]
        mapping = {_to_py(k): _to_py(v) for k, v in self.mapping_.items()}
        return {"class_order": class_order, "mapping": mapping}

    # def to_dict(self) -> dict:
    #     """落盘用"""
    #     return {"class_order": list(self.class_order_), "mapping": dict(self.mapping_)}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetMapper":
        return cls(class_order=d["class_order"])