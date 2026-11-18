# evaluation/rare_ratio.py
import numpy as np
import pandas as pd


def rare_class_ratio(real_cat: pd.DataFrame, syn_cat: pd.DataFrame,
                     thr: float = 0.10, min_count: int = 5):
    """
    real_cat / syn_cat：仅含分类列的原始标签 DataFrame（列名一致）。
    返回 (detail_df, summary_dict)。
    """
    n_real, n_syn = len(real_cat), len(syn_cat)
    rows = []
    for col in real_cat.columns:
        vc_r = real_cat[col].value_counts()
        vc_s = syn_cat[col].value_counts()
        for cls, cnt in vc_r.items():
            p_real = cnt / n_real
            if p_real >= thr or cnt < min_count:
                continue
            p_syn = vc_s.get(cls, 0) / n_syn
            ratio = (p_syn + 0.5 / n_syn) / p_real
            rows.append(dict(col=col, cls=cls, p_real=p_real, p_syn=p_syn,
                             ratio=ratio, log2_ratio=float(np.log2(ratio))))
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, dict(rare_bias=np.nan, rare_dev=np.nan, n_rare=0)
    summary = dict(
        rare_bias=float(detail['log2_ratio'].mean()),
        rare_dev=float(detail['log2_ratio'].abs().mean()),
        n_rare=int(len(detail)),
    )
    return detail, summary