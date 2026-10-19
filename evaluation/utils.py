# evaluation/utils.py
from typing import Optional
import warnings

import shap
import numpy as np
import pandas as pd
from scipy import stats

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

def _tvd_cont(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """TVD for a continuous column via equal-width histogram. Range: [0, 1]."""
    bins = np.histogram_bin_edges(np.concatenate([a, b]), bins=n_bins)
    p, _ = np.histogram(a, bins=bins)
    q, _ = np.histogram(b, bins=bins)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(0.5 * np.abs(p - q).sum())


def _tvd_cat(a: np.ndarray, b: np.ndarray) -> float:
    """TVD for a categorical column. Range: [0, 1]."""
    cats = np.union1d(np.unique(a), np.unique(b))
    p = np.array([np.mean(a == c) for c in cats])
    q = np.array([np.mean(b == c) for c in cats])
    return float(0.5 * np.abs(p - q).sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramér's V for two categorical arrays."""
    ct   = pd.crosstab(a, b).values.astype(float)
    n    = ct.sum()
    if n == 0:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    r, k = ct.shape
    phi2  = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_c   = r - (r - 1) ** 2 / (n - 1)
    k_c   = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_c - 1, k_c - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _eta_sq(num: np.ndarray, cat: np.ndarray) -> float:
    """η²: fraction of variance in a continuous col explained by a categorical col."""
    gm  = num.mean()
    ssb = sum(
        np.sum(cat == c) * (num[cat == c].mean() - gm) ** 2
        for c in np.unique(cat) if np.sum(cat == c) > 0
    )
    sst = np.sum((num - gm) ** 2)
    return float(ssb / sst) if sst > 1e-10 else 0.0


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


def _corr_diff(
    ref_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[float, float]:
    """Return (mean absolute diff, Frobenius norm) of upper-triangle correlation diff."""
    C_ref  = _mixed_corr(ref_df,  num_cols, cat_cols)
    C_fake = _mixed_corr(fake_df, num_cols, cat_cols)
    diff   = (C_ref - C_fake).abs()
    mask   = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    upper  = diff.values[mask]
    return float(upper.mean()), float(np.linalg.norm(diff.values))


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


def _kendall_mat(df, cols):
    K = df[cols].astype(float).corr(method='kendall').fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(K, 1.0)
    return pd.DataFrame(K, index=cols, columns=cols)


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


def compute_crps_old(train: pd.DataFrame, syn: pd.DataFrame, top_k: int = 10):
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    x_syn = syn.iloc[:, :-1]
    y_syn = syn.iloc[:, -1]

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    clf.fit(x_train, y_train)
    importance = clf.feature_importances_
    
    top_idx = np.argsort(importance)[::-1][:top_k]
    top_features = x_train.columns[top_idx]

    corr_real = (x_train[top_features].corr(method='spearman').values)
    corr_syn  = (x_syn[top_features].corr(method='spearman').values)

    fro_diff = np.linalg.norm(corr_real - corr_syn, ord='fro')
    fro_real = np.linalg.norm(corr_real, ord='fro')

    crps_F = 1 - fro_diff / (fro_real + 1e-8)

    idx = np.triu_indices_from(corr_real, k=1)
    r_real = corr_real[idx]
    r_syn  = corr_syn[idx]

    crps_S = spearmanr(r_real, r_syn).correlation

    return crps_F, crps_S


def compute_crps(train, syn, top_k=10, cat_cols=()):
    x_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    x_syn = syn.iloc[:, :-1]

    clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)
    clf.fit(x_train, y_train)
    top = list(x_train.columns[np.argsort(clf.feature_importances_)[::-1][:top_k]])

    n = [c for c in top if c not in cat_cols]
    c = [c for c in top if c in cat_cols]
    corr_real = _mixed_corr(x_train, n, c).to_numpy()
    corr_syn  = _mixed_corr(x_syn,  n, c).to_numpy()

    fro_diff = np.linalg.norm(corr_real - corr_syn, ord='fro')
    fro_real = np.linalg.norm(corr_real, ord='fro')
    crps_F = 1 - fro_diff / (fro_real + 1e-8)

    idx = np.triu_indices_from(corr_real, k=1)
    r_real, r_syn = corr_real[idx], corr_syn[idx]
    crps_S = spearmanr(r_real, r_syn).correlation if np.std(r_real) > 0 and np.std(r_syn) > 0 else float('nan')
    return crps_F, crps_S


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

    # imp_real = np.abs(shap_real).mean(axis=0)
    # imp_syn  = np.abs(shap_syn).mean(axis=0)

    imp_real = _shap_imp(shap_real)
    imp_syn  = _shap_imp(shap_syn)

    shap_corr = spearmanr(imp_real, imp_syn).correlation
    cos = cosine_similarity(imp_real.reshape(1, -1), imp_syn.reshape(1, -1))[0,0]

    return shap_corr, cos