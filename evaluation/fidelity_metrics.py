import warnings
import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
 
try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

from evaluation.fidelity_config import EvalConfig
from evaluation.fidelity_utils import (
    _subset, _active_cols, _tvd_cont, _tvd_cat, _corr_diff, _encode_scale, 
    _score_clf, _kendall_mat
)

# ══════════════════════════════════════════════════════════════════════════════
# Metric functions
# (all share the same signature for uniform registration and dispatch)
# ══════════════════════════════════════════════════════════════════════════════

def metric_tvd(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    config:   EvalConfig,
    **kwargs,
) -> dict:
    """
    Total Variation Distance per column and average.

    Comparison : fake_df vs test_df
    Mode       : controls full vs minority subset
    Outputs    : tvd_avg, tvd_<col> for each feature column
    Interpretation : lower is better; 0 = identical distributions
    """
    mode = config.get_mode('tvd')
    ref  = _subset(test_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)

    per_col: dict[str, float] = {}
    for col in n_cols:
        per_col[col] = _tvd_cont(ref[col].values, fake[col].values)
    for col in c_cols:
        per_col[col] = _tvd_cat(ref[col].values, fake[col].values)

    result = {
        'tvd_avg': float(np.mean(list(per_col.values()))) if per_col else float('nan')
    }
    for col, v in per_col.items():
        result[f'tvd_{col}'] = v
    return result


def metric_corr_diff(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    config:   EvalConfig,
    **kwargs,
) -> dict:
    """
    Mixed-type correlation matrix difference.

    Comparison : fake_df vs test_df
    Mode       : controls full vs minority subset
    Outputs    : corr_diff_mean (avg absolute diff), corr_diff_fro (Frobenius norm)
    Interpretation : lower is better; captures pairwise dependency structure gaps
    """
    mode = config.get_mode('corr_diff')
    ref  = _subset(test_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)

    if len(n_cols) + len(c_cols) < 2:
        warnings.warn("corr_diff: fewer than 2 active feature columns, skipping.")
        return {'corr_diff_mean': float('nan'), 'corr_diff_fro': float('nan')}

    mean_d, fro_d = _corr_diff(ref, fake, n_cols, c_cols)
    return {'corr_diff_mean': mean_d, 'corr_diff_fro': fro_d}


def metric_alpha_beta(
    train_df:    pd.DataFrame,
    test_df:     pd.DataFrame,
    fake_df:     pd.DataFrame,
    num_cols:    list[str],
    cat_cols:    list[str],
    config:      EvalConfig,
    k_neighbors: int = 5,
    **kwargs,
) -> dict:
    """
    α-Precision : fraction of fake samples that fall within the real support.
                  Low α → fake samples are unrealistic (hallucinated modes).
    β-Recall    : fraction of real local patterns covered by fake samples.
                  Low β → real modes are missing in fake data (mode dropping).
    Outputs : alpha_precision, beta_recall
    Interpretation : higher is better; both > 0.8 is a good target
    """
    mode = config.get_mode('alpha_beta')
    ref  = _subset(test_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)
    all_feat = n_cols + c_cols

    if not all_feat:
        warnings.warn("alpha_beta: no active feature columns.")
        return {'alpha_precision': float('nan'), 'beta_recall': float('nan')}

    k = k_neighbors
    if len(ref) < k + 1 or len(fake) < k + 1:
        warnings.warn(
            f"alpha_beta: insufficient samples "
            f"(ref={len(ref)}, fake={len(fake)}, k={k}). "
            f"Try reducing k_neighbors."
        )
        return {'alpha_precision': float('nan'), 'beta_recall': float('nan')}

    X_ref, X_fake = _encode_scale(ref[all_feat], fake[all_feat], cat_cols=c_cols)

    # local radius of each real sample = distance to its k-th nearest neighbor
    dist_rr = NearestNeighbors(n_neighbors=k + 1).fit(X_ref).kneighbors(X_ref)[0]
    radius  = dist_rr[:, k]                # shape: [N_ref]

    # α-Precision: for each fake sample, check if it falls within the
    # local radius of its nearest real neighbor
    dist_fr, idx_fr = NearestNeighbors(n_neighbors=k).fit(X_ref).kneighbors(X_fake)
    alpha = float(np.mean(dist_fr[:, 0] <= radius[idx_fr[:, 0]]))

    # β-Recall: for each real sample, check if a fake sample is close enough
    dist_rf = NearestNeighbors(n_neighbors=k).fit(X_fake).kneighbors(X_ref)[0]
    beta    = float(np.mean(dist_rf[:, 0] <= radius))

    return {'alpha_precision': alpha, 'beta_recall': beta}


def metric_c2st(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    config:       EvalConfig,
    n_folds:      int = 5,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """
    Interpretation :
        AUC ≈ 0.5 → indistinguishable (ideal)
        AUC > 0.7 → detectable gap
        AUC > 0.9 → highly distinguishable (poor quality)
    """
    if not _HAS_XGB and not _HAS_LGB:
        warnings.warn("c2st: xgboost/lightgbm not installed, skipping.")
        return {'c2st_auc': float('nan')}

    mode = config.get_mode('c2st')
    ref  = _subset(test_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)
    all_feat = n_cols + c_cols

    r = ref[all_feat].copy();  r['__y__'] = 0
    f = fake[all_feat].copy(); f['__y__'] = 1
    combined = pd.concat([r, f], ignore_index=True)
    for col in c_cols:
        combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
    y = combined.pop('__y__').values
    X = combined.values.astype(float)

    clf = (
        XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', verbosity=0, random_state=random_state,
        )
        if _HAS_XGB else
        LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=random_state,
        )
    )

    cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    auc = float(cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)

    result = {'c2st_auc': auc}
    if hasattr(clf, 'feature_importances_'):
        for col, imp in zip(all_feat, clf.feature_importances_):
            result[f'c2st_fimp_{col}'] = float(imp)
    return result


def metric_dcr(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    config:       EvalConfig,
    max_samples:  int = 2000,
    random_state: int = 42,
    **kwargs,
) -> dict:
    
    mode  = config.get_mode('dcr')
    fake  = _subset(fake_df,  mode, config)
    train = _subset(train_df, mode, config)
    test  = _subset(test_df,  mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)
    all_feat = n_cols + c_cols

    # normalize using train as reference (consistent scale across all sets)
    X_train, X_test, X_fake = _encode_scale(
        train[all_feat], test[all_feat], fake[all_feat], cat_cols=c_cols
    )

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_fake), min(max_samples, len(X_fake)), replace=False)
    X_s = X_fake[idx]

    dcr_train = NearestNeighbors(n_neighbors=1).fit(X_train).kneighbors(X_s)[0][:, 0]
    dcr_test  = NearestNeighbors(n_neighbors=1).fit(X_test).kneighbors(X_s)[0][:, 0]

    ks_stat, ks_pval = stats.ks_2samp(dcr_train, dcr_test)

    return {
        'dcr_train_mean': float(dcr_train.mean()),
        'dcr_test_mean':  float(dcr_test.mean()),
        'dcr_ks_stat':    float(ks_stat),
        'dcr_ks_pval':    float(ks_pval),
    }


def metric_cls(
        train_df:       pd.DataFrame,
        test_df:        pd.DataFrame,
        fake_df:        pd.DataFrame,
        num_cols:       list[str],
        cat_cols:       list[str],
        config:         EvalConfig,
        random_state:   int = 42,
        **kwargs,
) -> dict:
    """
    Downstream classification utility - measures whether synthetic minority samples improve a downstream
    classifier trained on the augmented set.

    Protocol
    --------
    Baseline:   XGB/LGB trained on the original trian_df, scored on test_df.
    Augmented:  identical architecture trained on train_df ∪ fake_df (all fake
                samples treated as minority class), scored on the same test_df.

    Both classifiers share identical hyperparameters; the only variable is training-set composition,
    making making Δ = aug − baseline a clean signal of augmentation utility.

    Design note
    --------
    mode / _subset() is intentionally not used here.  Downstream evaluation
    always uses the full train_df and test_df so that accuracy is measured
    across all classes.  fake_df supplies minority-class augmentation only.
 
    fake_df does not need to contain minority_col; when absent, minority_val
    is assigned to all rows automatically.
 
    Requires
    --------
    config.minority_col  – column name of the classification target
    config.minority_val  – value identifying the minority class
 
    Output keys
    -----------
    cls_baseline_{acc,recall,f1,gmean,roc_auc,pr_auc}
    cls_aug_{acc,recall,f1,gmean,roc_auc,pr_auc}
    cls_delta_{acc,recall,f1,gmean,roc_auc,pr_auc}   (aug − baseline)
 
    Interpretation
    --------------
    acc, recall, f1  – macro-averaged; higher is better
    gmean            – geometric mean of per-class recalls; higher is better;
                       sensitive to both majority and minority coverage
    roc_auc, pr_auc  – minority class as positive in binary; macro-OvR in
                       multiclass; higher is better
    cls_delta_*      – positive = augmentation improved the metric
    
    """
    
    _SFXS = ('acc', 'recall', 'f1', 'gmean', 'roc_auc', 'pr_auc')
    _nan_result: dict[str, float] = {
        f'cls_{p}_{s}': float('nan')
        for p in ('baseline', 'aug', 'delta') for s in _SFXS
    }

    if not (_HAS_XGB or _HAS_LGB):
        warnings.warn("cls: xgboost/lightgbm not installed, skipping.")
        return _nan_result
 
    if config.minority_col is None:
        warnings.warn("cls: EvalConfig.minority_col must be set, skipping.")
        return _nan_result
 
    label_col = config.minority_col
    cat_feats = [c for c in cat_cols if c != label_col]
    num_feats = [c for c in num_cols if c != label_col]
    feat_cols = num_feats + cat_feats
 
    if not feat_cols:
        warnings.warn("cls: no feature columns after excluding label, skipping.")
        return _nan_result
    
    # ── 1.  Attach label to fake_df if absent ────────────────────────────────
    fake_labeled = fake_df.copy()
    if label_col not in fake_labeled.columns:
        fake_labeled[label_col] = config.minority_val
 
    if len(fake_labeled) == 0:
        warnings.warn("cls: fake_df is empty — augmented metrics will equal baseline.")
 
    # ── 2.  Fit feature encoders on train_df (avoids test leakage) ───────────
    feat_encoders: dict[str, LabelEncoder] = {}
    for col in cat_feats:
        feat_encoders[col] = LabelEncoder().fit(train_df[col].astype(str))
 
    le_label = LabelEncoder().fit(train_df[label_col].astype(str))
 
    try:
        minority_enc: Optional[int] = int(
            le_label.transform([str(config.minority_val)])[0]
        )
    except Exception:
        warnings.warn(
            f"cls: minority_val={config.minority_val!r} not found in "
            f"train_df['{label_col}']; pos_label will default to largest class."
        )
        minority_enc = None
 
    # ── 3.  Encode features & labels consistently ────────────────────────────
    def _enc_X(df: pd.DataFrame) -> np.ndarray:
        enc = df[feat_cols].copy().reset_index(drop=True)
        for col in cat_feats:
            le = feat_encoders[col]
            enc[col] = df[col].astype(str).map(
                lambda v, _le=le: int(_le.transform([v])[0])
                if v in _le.classes_ else -1
            )
        return enc.values.astype(float)
 
    def _enc_y(df: pd.DataFrame) -> np.ndarray:
        return le_label.transform(df[label_col].astype(str))
 
    X_train = _enc_X(train_df);  y_train = _enc_y(train_df)
    X_test  = _enc_X(test_df);   y_test  = _enc_y(test_df)
    X_fake  = _enc_X(fake_labeled)
 
    # label for fake samples: use column if caller provided it, else minority_enc
    y_fake = (
        _enc_y(fake_labeled) if label_col in fake_df.columns
        else np.full(
            len(X_fake),
            minority_enc if minority_enc is not None else 1,
            dtype=int,
        )
    )
 
    X_aug = np.vstack([X_train, X_fake])
    y_aug = np.concatenate([y_train, y_fake])
 
    # ── 4.  Build classifier (same hyperparams for both runs) ─────────────────
    def _build() -> object:
        return (
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0, random_state=random_state,
            ) if _HAS_XGB else
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=random_state,
            )
        )
 
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf_base = _build(); clf_base.fit(X_train, y_train)
        clf_aug  = _build(); clf_aug.fit(X_aug,   y_aug)
 
    # ── 5.  Score both classifiers on the same held-out test set ─────────────
    base_s = _score_clf(clf_base, X_test, y_test, pos_label=minority_enc)
    aug_s  = _score_clf(clf_aug,  X_test, y_test, pos_label=minority_enc)
 
    # ── 6.  Assemble result dict ──────────────────────────────────────────────
    result: dict[str, float] = {}
    for k in _SFXS:
        bv, av = base_s[k], aug_s[k]
        result[f'cls_baseline_{k}'] = bv
        result[f'cls_aug_{k}']      = av
        result[f'cls_delta_{k}']    = (
            av - bv
            if not (np.isnan(bv) or np.isnan(av))
            else float('nan')
        )
    return result


def metric_tstr(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    config:       EvalConfig,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """
    TSTR evaluation with robust handling of unknown categories and label rounding.
    """
    _SFXS = ('acc', 'recall', 'f1', 'gmean', 'roc_auc', 'pr_auc')
    _nan  = {f'tstr_{p}_{s}': float('nan')
             for p in ('trtr', 'tstr', 'delta') for s in _SFXS}

    if not (_HAS_XGB or _HAS_LGB):
        warnings.warn("tstr: xgboost/lightgbm not installed, skipping.")
        return _nan
    if config.minority_col is None:
        warnings.warn("tstr: EvalConfig.minority_col must be set, skipping.")
        return _nan

    label_col = config.minority_col
    if label_col not in fake_df.columns:
        warnings.warn(f"tstr: fake_df missing '{label_col}'. Skipping.")
        return _nan

    # 特征列（排除标签）
    cat_feats = [c for c in cat_cols if c != label_col]
    num_feats = [c for c in num_cols if c != label_col]
    feat_cols = num_feats + cat_feats
    if not feat_cols:
        warnings.warn("tstr: no feature columns after excluding label, skipping.")
        return _nan

    # ----- 准备编码器及未知类别 fallback -----
    # 为每个类别特征计算众数（用于填充未知值）
    cat_modes = {}
    for col in cat_feats:
        # 转字符串，找到出现最多的类别
        mode_val = train_df[col].astype(str).mode()
        if len(mode_val) > 0:
            cat_modes[col] = mode_val.iloc[0]
        else:
            cat_modes[col] = "UNKNOWN"

    # Label encoder for features (fit on train)
    feat_encoders = {}
    for col in cat_feats:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        feat_encoders[col] = le

    # Label encoder for target
    le_label = LabelEncoder()
    # 将标签转为字符串，同时做舍入（如果标签是数值）
    train_labels = train_df[label_col].copy()
    if train_labels.dtype.kind in 'if':  # 浮点或整数
        train_labels = train_labels.round().astype(int).astype(str)
    else:
        train_labels = train_labels.astype(str)
    le_label.fit(train_labels)
    # minority 编码
    try:
        minority_enc = le_label.transform([str(config.minority_val)])[0]
    except Exception:
        # 如果 minority_val 不在标签中，尝试使用二分类的 pos_label 推断
        uniq = le_label.classes_
        if len(uniq) == 2:
            # 默认将较小的类作为负类，较大的作为正类？不确定，但至少不报错
            minority_enc = 1  # 假定第二个类为正
            warnings.warn(f"tstr: minority_val '{config.minority_val}' not found; using class '{uniq[1]}' as positive.")
        else:
            warnings.warn("tstr: cannot determine minority encoding; returning NaN.")
            return _nan

    def _enc_X(df: pd.DataFrame) -> np.ndarray:
        """编码特征矩阵，未知类别用众数填充后编码"""
        df_enc = df[feat_cols].copy().reset_index(drop=True)
        for col in cat_feats:
            # 转换为字符串
            col_vals = df[col].astype(str)
            # 未知类别 -> 众数
            unknown_mask = ~col_vals.isin(feat_encoders[col].classes_)
            if unknown_mask.any():
                col_vals[unknown_mask] = cat_modes[col]
            # 编码
            df_enc[col] = feat_encoders[col].transform(col_vals)
        # 数值列直接保留
        return df_enc.values.astype(float)

    def _enc_y(df: pd.DataFrame) -> np.ndarray:
        """编码标签，并做舍入（如果是数值）"""
        y = df[label_col].copy()
        if y.dtype.kind in 'if':
            y = y.round().astype(int).astype(str)
        else:
            y = y.astype(str)
        return le_label.transform(y)

    X_real = _enc_X(train_df); y_real = _enc_y(train_df)
    X_fake = _enc_X(fake_df);  y_fake = _enc_y(fake_df)
    X_test = _enc_X(test_df);  y_test = _enc_y(test_df)

    def _build() -> object:
        return (
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0, random_state=random_state,
            ) if _HAS_XGB else
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=random_state,
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf_trtr = _build(); clf_trtr.fit(X_real, y_real)
        clf_tstr = _build(); clf_tstr.fit(X_fake, y_fake)

    trtr_s = _score_clf(clf_trtr, X_test, y_test, pos_label=minority_enc)
    tstr_s = _score_clf(clf_tstr, X_test, y_test, pos_label=minority_enc)

    result = {}
    for k in _SFXS:
        rv, sv = trtr_s[k], tstr_s[k]
        result[f'tstr_trtr_{k}'] = rv
        result[f'tstr_tstr_{k}'] = sv
        result[f'tstr_delta_{k}'] = sv - rv if not (np.isnan(rv) or np.isnan(sv)) else float('nan')
    return result


def metric_tstr_old(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    config:       EvalConfig,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """
    Train on Synthetic, Test on Real (TSTR) — full-dataset utility evaluation.

    TRTR : train on train_df (real),  score on test_df  [reference]
    TSTR : train on fake_df (full synthetic set), score on same test_df
    Δ    = TSTR − TRTR;  near 0 means synthetic generalises as well as real

    Unlike metric_cls (minority-only augmentation), fake_df here is the
    complete synthetic dataset and MUST contain minority_col.
    mode / _subset() intentionally not applied — always full-distribution.

    Requires
    --------
    config.minority_col  – classification target column
    config.minority_val  – minority class value (binary pos_label)

    Output keys
    -----------
    tstr_trtr_{acc,recall,f1,gmean,roc_auc,pr_auc}
    tstr_tstr_{acc,recall,f1,gmean,roc_auc,pr_auc}
    tstr_delta_{acc,recall,f1,gmean,roc_auc,pr_auc}
    """
    _SFXS = ('acc', 'recall', 'f1', 'gmean', 'roc_auc', 'pr_auc')
    _nan  = {f'tstr_{p}_{s}': float('nan')
             for p in ('trtr', 'tstr', 'delta') for s in _SFXS}

    if not (_HAS_XGB or _HAS_LGB):
        warnings.warn("tstr: xgboost/lightgbm not installed, skipping.")
        return _nan
    if config.minority_col is None:
        warnings.warn("tstr: EvalConfig.minority_col must be set, skipping.")
        return _nan

    label_col = config.minority_col

    if label_col not in fake_df.columns:
        warnings.warn(
            f"tstr: fake_df must contain '{label_col}'. "
            f"TSTR requires the full synthetic dataset with labels. Skipping."
        )
        return _nan

    cat_feats = [c for c in cat_cols if c != label_col]
    num_feats = [c for c in num_cols if c != label_col]
    feat_cols = num_feats + cat_feats

    if not feat_cols:
        warnings.warn("tstr: no feature columns after excluding label, skipping.")
        return _nan

    # ── Encoders fitted on train_df (real) as reference ──────────────────────
    feat_encoders: dict[str, LabelEncoder] = {}
    for col in cat_feats:
        feat_encoders[col] = LabelEncoder().fit(train_df[col].astype(str))

    le_label = LabelEncoder().fit(train_df[label_col].astype(str))
    try:
        minority_enc: Optional[int] = int(
            le_label.transform([str(config.minority_val)])[0]
        )
    except Exception:
        minority_enc = None

    def _enc_X(df: pd.DataFrame) -> np.ndarray:
        enc = df[feat_cols].copy().reset_index(drop=True)
        for col in cat_feats:
            le = feat_encoders[col]
            enc[col] = df[col].astype(str).map(
                lambda v, _le=le: int(_le.transform([v])[0])
                if v in _le.classes_ else -1
            )
        return enc.values.astype(float)

    def _enc_y(df: pd.DataFrame) -> np.ndarray:
        y = df[label_col].copy()
        if y.dtype.kind in 'if':          # 如果是浮点数或整数
            y = y.round().astype(int).astype(str)   # 四舍五入取整后转字符串
        else:
            y = y.astype(str)
        return le_label.transform(y)

    def _enc_y(df: pd.DataFrame) -> np.ndarray:
        return le_label.transform(df[label_col].astype(str))

    X_real = _enc_X(train_df); y_real = _enc_y(train_df)
    X_fake = _enc_X(fake_df);  y_fake = _enc_y(fake_df)
    X_test = _enc_X(test_df);  y_test = _enc_y(test_df)

    def _build() -> object:
        return (
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0, random_state=random_state,
            ) if _HAS_XGB else
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=random_state,
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf_trtr = _build(); clf_trtr.fit(X_real, y_real)
        clf_tstr = _build(); clf_tstr.fit(X_fake, y_fake)

    trtr_s = _score_clf(clf_trtr, X_test, y_test, pos_label=minority_enc)
    tstr_s = _score_clf(clf_tstr, X_test, y_test, pos_label=minority_enc)

    result: dict[str, float] = {}
    for k in _SFXS:
        rv, sv = trtr_s[k], tstr_s[k]
        result[f'tstr_trtr_{k}'] = rv
        result[f'tstr_tstr_{k}'] = sv
        result[f'tstr_delta_{k}'] = (
            sv - rv if not (np.isnan(rv) or np.isnan(sv)) else float('nan')
        )
    return result


def metric_kendall(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    config:   EvalConfig,
    **kwargs,
) -> dict:
    """
    Kendall τ-b inter-attribute association heatmap.

    Comparison  : fake_df vs test_df (numerical columns only)
    Mode        : configurable via config / metric_modes['kendall']
    Outputs     : summary stats + per-pair values for heatmap reconstruction

    kendall_diff_mean / _fro          – summary (lower is better)
    kendall_real_{ci}__{cj}           – real pairwise τ (upper triangle)
    kendall_fake_{ci}__{cj}           – fake pairwise τ
    kendall_diff_{ci}__{cj}           – |real − fake| per pair
    """
    mode      = config.get_mode('kendall')
    ref       = _subset(test_df, mode, config)
    fake      = _subset(fake_df, mode, config)
    n_cols, _ = _active_cols(num_cols, cat_cols, mode, config)

    if len(n_cols) < 2:
        warnings.warn("kendall: fewer than 2 active numerical columns, skipping.")
        return {'kendall_diff_mean': float('nan'), 'kendall_diff_fro': float('nan')}

    K_ref  = _kendall_mat(ref,  n_cols)
    K_fake = _kendall_mat(fake, n_cols)
    diff   = (K_ref - K_fake).abs()

    mask  = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    upper = diff.values[mask]

    result: dict[str, float] = {
        'kendall_diff_mean': float(upper.mean()),
        'kendall_diff_fro':  float(np.linalg.norm(diff.values)),
    }
    for i, ci in enumerate(n_cols):
        for j, cj in enumerate(n_cols):
            if j > i:
                result[f'kendall_real_{ci}__{cj}'] = float(K_ref.iloc[i, j])
                result[f'kendall_fake_{ci}__{cj}'] = float(K_fake.iloc[i, j])
                result[f'kendall_diff_{ci}__{cj}'] = float(diff.iloc[i, j])
    return result