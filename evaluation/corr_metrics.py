# evaluation/corr_metrics.py
"""把 correlation_utils 的两个函数适配为统一的 metric 签名。

输出键（均为汇总量，无逐属性明细）：
    crps_F, crps_S          —— 需要 real_df 含 >=2 个类别
    shap_corr, shap_cos     —— 需要 real_df 与 fake_df 均含 >=2 个类别（即 full_data / tstr 场景）
"""
from __future__ import annotations
import warnings
import pandas as pd

from evaluation.fidelity_utils import EvalConfig
from evaluation.correlation_utils import compute_crps, compute_shap_preservation

_NAN_CRPS = {'crps_F': float('nan'), 'crps_S': float('nan')}
_NAN_SHAP = {'shap_corr': float('nan'), 'shap_cos': float('nan')}


def _y_str(s: pd.Series) -> pd.Series:
    return s.round().astype(int).astype(str) if s.dtype.kind in 'if' else s.astype(str)


def _prep(real_df, fake_df, num_cols, cat_cols, label_col):
    """全部转为数值；标签放最后一列（correlation_utils 的约定）。
    类别/标签编码以 real_df 为参照；fake 中未见过的类别 -> -1，未见过的标签行丢弃。"""
    feats = [c for c in num_cols + cat_cols if c != label_col]
    cats = [c for c in cat_cols if c != label_col]
    cmaps = {c: {v: i for i, v in enumerate(sorted(real_df[c].astype(str).unique()))} for c in cats}
    ymap = {v: i for i, v in enumerate(sorted(_y_str(real_df[label_col]).unique()))}

    def enc(df):
        X = df[feats].copy().reset_index(drop=True)
        for c in cats:
            X[c] = X[c].astype(str).map(cmaps[c]).fillna(-1)
        y = _y_str(df[label_col]).map(ymap).reset_index(drop=True)
        keep = y.notna()
        out = X[keep].astype(float)
        out[label_col] = y[keep].astype(int)
        return out.reset_index(drop=True)

    return enc(real_df), enc(fake_df)


def metric_crps(real_df, fake_df, num_cols, cat_cols, config: EvalConfig,
                top_k: int = 10, **kwargs) -> dict:
    if config.minority_col is None:
        warnings.warn("crps: EvalConfig.minority_col (label) must be set, skipping.")
        return dict(_NAN_CRPS)
    tr, fk = _prep(real_df, fake_df, num_cols, cat_cols, config.minority_col)
    if tr.iloc[:, -1].nunique() < 2 or tr.shape[1] - 1 < 3:
        warnings.warn("crps: real_df needs >=2 classes and >=3 features "
                      "(pre_filtered real_df is single-class), skipping.")
        return dict(_NAN_CRPS)
    F, S = compute_crps(tr, fk, top_k=min(top_k, tr.shape[1] - 1))
    return {'crps_F': float(F), 'crps_S': float(S)}


def metric_shap(real_df, fake_df, num_cols, cat_cols, config: EvalConfig,
                **kwargs) -> dict:
    if config.minority_col is None:
        warnings.warn("shap: EvalConfig.minority_col (label) must be set, skipping.")
        return dict(_NAN_SHAP)
    tr, fk = _prep(real_df, fake_df, num_cols, cat_cols, config.minority_col)
    if tr.iloc[:, -1].nunique() < 2 or fk.iloc[:, -1].nunique() < 2:
        warnings.warn("shap: both real_df and fake_df need >=2 classes "
                      "(not applicable to minority-only fake data), skipping.")
        return dict(_NAN_SHAP)
    c, s = compute_shap_preservation(tr, fk)
    return {'shap_corr': float(c), 'shap_cos': float(s)}