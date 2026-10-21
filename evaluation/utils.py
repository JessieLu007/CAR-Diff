# evaluation/utils.py
from typing import Optional
import warnings

import shap
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from xgboost import XGBClassifier


# =================================================================
# internal helper
# =================================================================

def _encode_scale(
    ref_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
    cat_cols: list[str],
) -> list[np.ndarray]:
    cats = {col: np.unique(ref_df[col].astype(str).to_numpy()) for col in cat_cols}  # 已排序，同 LabelEncoder.classes_

    arrays: list[np.ndarray] = []
    for df in (ref_df, *other_dfs):
        enc = df.copy().reset_index(drop=True)
        for col in cat_cols:
            codes = pd.Categorical(df[col].astype(str).to_numpy(), categories=cats[col]).codes
            enc[col] = codes.astype(float)          # 未见类别为 -1
        arrays.append(enc.values.astype(float))

    mu  = arrays[0].mean(axis=0)
    sig = arrays[0].std(axis=0) + 1e-8
    return [(a - mu) / sig for a in arrays]


# =================================================================
# statistical computation
# =================================================================

# ── Low-level statistics ───────────────────────────────────────────────────────

def _as_str(x) -> np.ndarray:
    """类别值转字符串；数值型先转 float，避免 int 的 '1' 与 float 的 '1.0' 被当成不同类别。"""
    x = np.asarray(x)
    return x.astype(float).astype(str) if x.dtype.kind in 'iufb' else x.astype(str)


def _ks_stat(a: np.ndarray, b: np.ndarray) -> float:
    """两样本 KS 统计量（1 − 该值 = SDMetrics 的 KSComplement）。Range: [0, 1]."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    v = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, v, side='right') / len(a)
                        - np.searchsorted(b, v, side='right') / len(b)).max())


def _codes_pair(a, b):
    """两组类别值按并集编码，返回 (codes_a, codes_b, K)。"""
    a, b = _as_str(a), _as_str(b)
    cats = np.union1d(np.unique(a), np.unique(b))
    return (pd.Categorical(a, categories=cats).codes.astype(np.int64),
            pd.Categorical(b, categories=cats).codes.astype(np.int64), len(cats))


def _tvd_cat(a, b) -> float:
    """类别列 TVD（1 − 该值 = SDMetrics 的 TVComplement）。Range: [0, 1]."""
    ca, cb, k = _codes_pair(a, b)
    p = np.bincount(ca, minlength=k) / len(ca)
    q = np.bincount(cb, minlength=k) / len(cb)
    return float(0.5 * np.abs(p - q).sum())


def _num_bin_codes(a, b, n_bins: int = 10):
    """数值列按 a（ref）的取值范围等宽分箱，b 超出范围的落入首/末箱（同 SDMetrics 的离散化）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    lo, hi = a.min(), a.max()
    if not hi > lo:
        return np.zeros(len(a), np.int64), np.zeros(len(b), np.int64), 1
    inner = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return np.digitize(a, inner).astype(np.int64), np.digitize(b, inner).astype(np.int64), n_bins


def _contingency_sim(xr, xf, kx, yr, yf, ky) -> float:
    """SDMetrics ContingencySimilarity：1 − ½ Σ|p_ref(a,b) − p_fake(a,b)|。"""
    m = kx * ky
    p = np.bincount(xr * ky + yr, minlength=m) / len(xr)
    q = np.bincount(xf * ky + yf, minlength=m) / len(xf)
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def _cpt_scores(ref, fake, num_cols, cat_cols, n_bins: int = 10):
    """SDMetrics Column Pair Trends 的逐对得分，返回 [(col_a, col_b, kind, score)]，kind ∈ nn / nc / cc。
    num–num: 1 − |ρ_ref − ρ_fake| / 2（Pearson；常数列相关置 0）；含类别列: 列联相似度（数值列分箱）。"""
    out = []
    p = len(num_cols)
    if p > 1:
        def _pearson(df):
            X = df[num_cols].to_numpy(dtype=float)
            with np.errstate(all='ignore'):
                C = np.nan_to_num(np.corrcoef(X, rowvar=False))
            const = X.std(axis=0) < 1e-8
            C[const, :] = 0.0
            C[:, const] = 0.0
            return C
        iu = np.triu_indices(p, k=1)
        s = 1.0 - np.abs(_pearson(ref) - _pearson(fake))[iu] / 2.0
        out += [(num_cols[i], num_cols[j], 'nn', float(v)) for i, j, v in zip(iu[0], iu[1], s)]
    if not cat_cols:
        return out

    codes = {c: _codes_pair(ref[c].to_numpy(), fake[c].to_numpy()) for c in cat_cols}
    codes.update({c: _num_bin_codes(ref[c].to_numpy(), fake[c].to_numpy(), n_bins) for c in num_cols})
    for a in num_cols:
        for b in cat_cols:
            xr, xf, kx = codes[a]
            yr, yf, ky = codes[b]
            out.append((a, b, 'nc', _contingency_sim(xr, xf, kx, yr, yf, ky)))
    for i, a in enumerate(cat_cols):
        for b in cat_cols[i + 1:]:
            xr, xf, kx = codes[a]
            yr, yf, ky = codes[b]
            out.append((a, b, 'cc', _contingency_sim(xr, xf, kx, yr, yf, ky)))
    return out


def _mixed_corr(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """
    num–num: Spearman（秩的 Pearson 相关，一次矩阵运算）
    cat–cat: bias-corrected Cramér's V（bincount 列联表）
    num–cat: η²（bincount 分组求和）
    """
    all_cols = num_cols + cat_cols
    p, q = len(num_cols), len(cat_cols)
    C = np.eye(len(all_cols))

    # ---- num–num ----
    if p > 1:
        X = df[num_cols].to_numpy(dtype=float)
        const = X.std(axis=0) < 1e-8
        R = df[num_cols].astype(float).rank(method='average').to_numpy()
        S = np.nan_to_num(np.corrcoef(R, rowvar=False))
        S[const, :] = 0.0
        S[:, const] = 0.0
        np.fill_diagonal(S, 1.0)
        C[:p, :p] = S

    # ---- 类别列：只编码一次 ----
    codes, levels = [], []
    for c in cat_cols:
        cd, uniq = pd.factorize(df[c].to_numpy(), sort=False)
        codes.append(cd.astype(np.int64))
        levels.append(len(uniq))
    n = len(df)

    # ---- cat–cat ----
    for i in range(q):
        for j in range(i + 1, q):
            r, k = levels[i], levels[j]
            if n == 0 or r < 2 or k < 2:
                v = 0.0
            else:
                ct = np.bincount(codes[i] * k + codes[j], minlength=r * k).reshape(r, k).astype(float)
                exp = np.outer(ct.sum(1), ct.sum(0)) / n
                chi2 = ((ct - exp) ** 2 / exp).sum()
                phi2 = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
                r_c = r - (r - 1) ** 2 / (n - 1)
                k_c = k - (k - 1) ** 2 / (n - 1)
                denom = min(r_c - 1, k_c - 1)
                v = float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0
            C[p + i, p + j] = C[p + j, p + i] = v

    # ---- num–cat ----
    for a in range(p):
        x = df[num_cols[a]].to_numpy(dtype=float)
        gm = x.mean()
        sst = np.sum((x - gm) ** 2)
        for i in range(q):
            if sst <= 1e-10:
                v = 0.0
            else:
                cnt = np.bincount(codes[i], minlength=levels[i]).astype(float)
                sm = np.bincount(codes[i], weights=x, minlength=levels[i])
                ssb = np.sum(cnt * (sm / cnt - gm) ** 2)
                v = float(ssb / sst)
            C[a, p + i] = C[p + i, a] = v

    return pd.DataFrame(C, index=all_cols, columns=all_cols)


# ── Classification helpers ─────────────────────────────────────────────────────
 
def _g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Geometric mean of per-class recalls.
      Binary    : sqrt(sensitivity × specificity)
      Multiclass: (∏ recall_i)^{1/K}
    Returns 0.0 if any class is entirely missed, nan if < 2 classes present.
    """
    classes = np.unique(y_true)
    if len(classes) < 2:
        return float('nan')
    recalls = recall_score(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    return float(np.prod(recalls) ** (1.0 / len(classes)))


def _score_clf(
    clf,
    X_test:    np.ndarray,
    y_test:    np.ndarray,
    pos_label: Optional[int] = None,
) -> dict[str, float]:
    """
    Compute all 6 classification metrics for a fitted sklearn-compatible clf.
 
    Parameters
    ----------
    clf       : fitted classifier with predict / predict_proba
    X_test    : encoded feature matrix
    y_test    : integer-encoded ground-truth labels
    pos_label : encoded integer for the minority (positive) class;
                governs which predict_proba column is used in binary AUC.
                Defaults to the largest class integer when None.
 
    Returns
    -------
    dict with keys: acc, recall, f1, gmean, roc_auc, pr_auc
    """
    y_pred      = clf.predict(X_test)
    clf_classes = np.array(clf.classes_)
    n_cls       = len(np.unique(y_test))
 
    out: dict[str, float] = dict(
        acc     = float(accuracy_score(y_test, y_pred)),
        recall  = float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        f1      = float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        gmean   = _g_mean(y_test, y_pred),
        roc_auc = float('nan'),
        pr_auc  = float('nan'),
    )
 
    try:
        proba = clf.predict_proba(X_test)           # (n_samples, n_clf_classes)
        if n_cls == 2:
            pos = pos_label if pos_label is not None else int(clf_classes[-1])
            col = int(np.where(clf_classes == pos)[0][0])
            # convert y_test to binary: minority=1, other=0 so AUC direction is correct
            y_bin = (y_test == pos).astype(int)
            out['roc_auc'] = float(roc_auc_score(y_bin, proba[:, col]))
            out['pr_auc']  = float(average_precision_score(y_bin, proba[:, col]))
        else:
            y_bin = label_binarize(y_test, classes=clf_classes)
            out['roc_auc'] = float(roc_auc_score(
                y_bin, proba, multi_class='ovr', average='macro'
            ))
            out['pr_auc'] = float(average_precision_score(
                y_bin, proba, average='macro'
            ))
    except Exception as exc:
        warnings.warn(f"_score_clf: AUC computation failed ({exc})")
 
    return out


def _shap_imp(sv):
    if isinstance(sv, list):                 # 旧版 shap：每类一个数组
        sv = np.stack(sv, axis=-1)           # (n, f, k)
    imp = np.abs(sv).mean(axis=0)            # (f,) 或 (f, k)
    if imp.ndim > 1:
        imp = imp.mean(axis=-1)              # 对类别取平均 -> (f,)
    return imp


def compute_shap_preservation(train: pd.DataFrame, syn: pd.DataFrame):
    x_real = train.iloc[:, :-1]
    y_real = train.iloc[:, -1]

    x_syn = syn.iloc[:, :-1]
    y_syn = syn.iloc[:, -1]

    clf_real = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    clf_real.fit(x_real, y_real)

    clf_syn = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )
    clf_syn.fit(x_syn, y_syn)

    explainer_real = shap.TreeExplainer(clf_real)
    explainer_syn = shap.TreeExplainer(clf_syn)

    shap_real = explainer_real.shap_values(x_real)
    shap_syn = explainer_syn.shap_values(x_real)

    imp_real = _shap_imp(shap_real)
    imp_syn  = _shap_imp(shap_syn)

    shap_corr = spearmanr(imp_real, imp_syn).correlation
    cos = cosine_similarity(imp_real.reshape(1, -1), imp_syn.reshape(1, -1))[0,0]

    return shap_corr, cos
    