# evaluation/align.py
"""把生成数据对齐到真实训练数据的格式：列顺序、类别取值、数值精度/量化步长/取值范围。
规则直接复用 TabularDataProcessor.canonicalize，所有方法（含自己的）在评估读取时统一调用。"""
import numpy as np
import pandas as pd
from core.data_processor import TabularDataProcessor


def make_aligner(train: pd.DataFrame, meta: dict):
    cols = list(train.columns)
    num_idx = [cols.index(c) for c in meta['num_cols']]
    cat_idx = [cols.index(c) for c in meta['cat_cols']]
    proc = TabularDataProcessor(num_idx, cat_idx).fit(train)

    def align(fake: pd.DataFrame) -> pd.DataFrame:
        fake = fake.copy()
        fake.columns = fake.columns.str.strip()
        fake = fake[cols].copy().reset_index(drop=True)          # 缺列或改名会在此报 KeyError
        for c in meta['cat_cols']:                               # 数值编码的类别列：吸附到真实取值
            if pd.api.types.is_numeric_dtype(train[c]):
                vals = np.sort(train[c].unique()).astype(float)
                v = fake[c].to_numpy(dtype=float)
                fake[c] = vals[np.abs(v[:, None] - vals[None, :]).argmin(1)]
        return proc.canonicalize(fake)

    return align