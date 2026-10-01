# evaluation/dep_metrics.py
"""独立的新指标：prdc（调用 prdc 包）与强相关对保持情况（Spearman，仅数值列）。
每个函数：输入 d，输出一个 {指标名: 数值} 字典。以验证集 d.test 为真实参照。
"""
import numpy as np
import pandas as pd
from evaluation.fidelity_utils import _active_cols

def _matrix(real, fake, num_cols, cat_cols):
    """类别列 one-hot、数值列按 real 的均值/方差标准化；返回 (X_real, X_fake)"""
    cols = num_cols + cat_cols
    both = pd.concat([real[cols], fake[cols]], keys=['r', 'f'])
    for c in cat_cols:
        both[c] = both[c].astype(str)
    X = pd.get_dummies(both, columns=cat_cols, dtype=float)
    mu, sd = X.loc['r'].mean(), X.loc['r'].std(ddof=0).replace(0, 1.0)
    X = (X - mu) / sd
    return X.loc['r'].to_numpy(), X.loc['f'].to_numpy()


def metric_prdc(real_df, fake_df, num_cols, cat_cols, config, k: int = 5, **kw) -> dict:
    from prdc import compute_prdc
    n, c = _active_cols(num_cols, cat_cols, config.get_mode('prdc'), config)
    xr, xf = _matrix(real_df, fake_df, n, c)
    names = ('precision', 'recall', 'density', 'coverage')
    if min(len(xr), len(xf)) <= k:
        return {f'prdc_{x}': float('nan') for x in names}
    r = compute_prdc(real_features=xr, fake_features=xf, nearest_k=k)
    return {f'prdc_{x}': float(r[x]) for x in names}


def metric_dep(real_df, fake_df, num_cols, cat_cols, config,
               threshold: float = 0.5, tolerance: float = 0.1, **kw) -> dict:
    cols, _ = _active_cols(num_cols, cat_cols, config.get_mode('dep'), config)
    iu = np.triu_indices(len(cols), k=1)
    r = real_df[cols].corr(method='spearman').to_numpy()[iu]
    f = np.nan_to_num(fake_df[cols].corr(method='spearman').to_numpy()[iu])
    m = np.abs(r) >= threshold
    nan = float('nan')
    if not m.any():
        return {'dep_n_pairs': 0, 'dep_err_mean': nan, 'dep_err_median': nan,
                'dep_preserve_rate': nan, 'dep_direction_rate': nan}
    err = np.abs(r[m] - f[m])
    return {
        'dep_n_pairs': int(m.sum()),
        'dep_err_mean': float(err.mean()),
        'dep_err_median': float(np.median(err)),
        'dep_preserve_rate': float((err <= tolerance).mean()),
        'dep_direction_rate': float((np.sign(r[m]) == np.sign(f[m])).mean()),
    }