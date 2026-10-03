from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from evaluation.utils import _corr_diff, _encode_scale, _kendall_mat, _tvd_cat, _tvd_cont, compute_crps, compute_shap_preservation, _score_clf

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



@dataclass
class EvalConfig:
    default_mode: Literal['all', 'minority'] = 'all'
    minority_col: Optional[str] = None
    minority_val: Any = None
    metric_modes: dict[str, str] = field(default_factory=dict)
    # pre_filtered=True 时，default_mode / metric_modes 对 _subset() 不再有意义，
    # 但 minority_col 仍供 _active_cols() 使用
    pre_filtered: bool = False
    def get_mode(self, metric: str) -> str:
        return self.metric_modes.get(metric, self.default_mode)


def _subset(df: pd.DataFrame, mode: str, config: EvalConfig) -> pd.DataFrame:
    if config.pre_filtered:
        return df.copy() # 数据已预过滤，直接返回 
    """Return minority subset or full DataFrame based on mode."""
    if mode == 'minority':
        if config.minority_col is None:
            raise ValueError(
                "EvalConfig.minority_col must be set when using mode='minority'."
            )
        sub = df[df[config.minority_col] == config.minority_val]
        if len(sub) == 0:
            warnings.warn(
                f"Minority subset is empty "
                f"(col='{config.minority_col}', val={config.minority_val!r})."
            )
        return sub.copy()
    return df.copy()


def _active_cols(
    num_cols: list[str],
    cat_cols: list[str],
    mode: str,
    config: EvalConfig,
) -> tuple[list[str], list[str]]:
    """
    In minority mode, minority_col is constant across the subset → exclude it.
    Constant columns corrupt correlation matrices and distance computations.
    """
    should_exclude = (
        config.minority_col is not None
        and (mode == 'minority' or config.pre_filtered)
    )
    if should_exclude:
        n = [c for c in num_cols if c != config.minority_col]
        c = [c for c in cat_cols if c != config.minority_col]
        return n, c
    return list(num_cols), list(cat_cols)


# ==========================================================
# 参照型： （ref_df, fake_df, ...)
# tvd, corr_diff, c2st, kendall, prdc, dep, crps, shap
# ==========================================================

def metric_tvd(
    ref_df:  pd.DataFrame,
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
    ref  = _subset(ref_df, mode, config)
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
    ref_df:  pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    config:   EvalConfig,
    **kwargs,
) -> dict:
    """
    Mixed-type correlation matrix difference.

    Comparison : fake_df vs ref_df
    Mode       : controls full vs minority subset
    Outputs    : corr_diff_mean (avg absolute diff), corr_diff_fro (Frobenius norm)
    Interpretation : lower is better; captures pairwise dependency structure gaps
    """
    mode = config.get_mode('corr_diff')
    ref  = _subset(ref_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)

    if len(n_cols) + len(c_cols) < 2:
        warnings.warn("corr_diff: fewer than 2 active feature columns, skipping.")
        return {'corr_diff_mean': float('nan'), 'corr_diff_fro': float('nan')}

    mean_d, fro_d = _corr_diff(ref, fake, n_cols, c_cols)
    return {'corr_diff_mean': mean_d, 'corr_diff_fro': fro_d}


def metric_c2st(
    ref_df:      pd.DataFrame,
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
    ref  = _subset(ref_df, mode, config)
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


def metric_kendall(
    ref_df:  pd.DataFrame,
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
    ref       = _subset(ref_df, mode, config)
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


def metric_prdc(ref_df, fake_df, num_cols, cat_cols, config, k: int = 5, **kw) -> dict:
    from prdc import compute_prdc
    n, c = _active_cols(num_cols, cat_cols, config.get_mode('prdc'), config)
    xr, xf = _matrix(ref_df, fake_df, n, c)
    names = ('precision', 'recall', 'density', 'coverage')
    if min(len(xr), len(xf)) <= k:
        return {f'prdc_{x}': float('nan') for x in names}
    r = compute_prdc(real_features=xr, fake_features=xf, nearest_k=k)
    return {f'prdc_{x}': float(r[x]) for x in names}


def metric_dep(ref_df, fake_df, num_cols, cat_cols, config,
               threshold: float = 0.5, tolerance: float = 0.1, **kw) -> dict:
    cols, _ = _active_cols(num_cols, cat_cols, config.get_mode('dep'), config)
    iu = np.triu_indices(len(cols), k=1)
    r = ref_df[cols].corr(method='spearman').to_numpy()[iu]
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


def metric_alpha_beta(ref_df, fake_df, num_cols, cat_cols, config,
                      random_state: int = 42, **kw) -> dict:
    from synthcity.metrics.eval_statistical import AlphaPrecision
    from synthcity.plugins.core.dataloader import GenericDataLoader
    n, c = _active_cols(num_cols, cat_cols, config.get_mode('alpha_beta'), config)
    xr, xf = _matrix(ref_df, fake_df, n, c)          # 标准化在下采样之前完成

    m = min(len(xr), len(xf))                        # synthcity 要求两边等长
    rng = np.random.default_rng(random_state)
    if len(xr) > m:
        xr = xr[rng.choice(len(xr), m, replace=False)]
    if len(xf) > m:
        xf = xf[rng.choice(len(xf), m, replace=False)]

    res = AlphaPrecision().evaluate(GenericDataLoader(pd.DataFrame(xr).add_prefix('f')),
                                    GenericDataLoader(pd.DataFrame(xf).add_prefix('f')))
    return {'alpha_precision': float(res['delta_precision_alpha_naive']),
            'beta_recall':     float(res['delta_coverage_beta_naive'])}

_NAN_CRPS = {'crps_F': float('nan'), 'crps_S': float('nan')}
_NAN_SHAP = {'shap_corr': float('nan'), 'shap_cos': float('nan')}


def _y_str(s: pd.Series) -> pd.Series:
    return s.round().astype(int).astype(str) if s.dtype.kind in 'if' else s.astype(str)


def _prep(ref_df, fake_df, num_cols, cat_cols, label_col):
    """全部转为数值；标签放最后一列（correlation_utils 的约定）。
    类别/标签编码以 ref_df 为参照；fake 中未见过的类别 -> -1，未见过的标签行丢弃。"""
    feats = [c for c in num_cols + cat_cols if c != label_col]
    cats = [c for c in cat_cols if c != label_col]
    cmaps = {c: {v: i for i, v in enumerate(sorted(ref_df[c].astype(str).unique()))} for c in cats}
    ymap = {v: i for i, v in enumerate(sorted(_y_str(ref_df[label_col]).unique()))}

    def enc(df):
        X = df[feats].copy().reset_index(drop=True)
        for c in cats:
            X[c] = X[c].astype(str).map(cmaps[c]).fillna(-1)
        y = _y_str(df[label_col]).map(ymap).reset_index(drop=True)
        keep = y.notna()
        out = X[keep].astype(float)
        out[label_col] = y[keep].astype(int)
        return out.reset_index(drop=True)

    return enc(ref_df), enc(fake_df)


def metric_crps(ref_df, fake_df, num_cols, cat_cols, config: EvalConfig,
                top_k: int = 10, **kwargs) -> dict:
    if config.minority_col is None:
        warnings.warn("crps: EvalConfig.minority_col (label) must be set, skipping.")
        return dict(_NAN_CRPS)
    tr, fk = _prep(ref_df, fake_df, num_cols, cat_cols, config.minority_col)
    if tr.iloc[:, -1].nunique() < 2 or tr.shape[1] - 1 < 3:
        warnings.warn("crps: ref_df needs >=2 classes and >=3 features "
                      "(pre_filtered ref_df is single-class), skipping.")
        return dict(_NAN_CRPS)
    F, S = compute_crps(tr, fk, top_k=min(top_k, tr.shape[1] - 1))
    return {'crps_F': float(F), 'crps_S': float(S)}


def metric_shap(ref_df, fake_df, num_cols, cat_cols, config: EvalConfig,
                **kwargs) -> dict:
    if config.minority_col is None:
        warnings.warn("shap: EvalConfig.minority_col (label) must be set, skipping.")
        return dict(_NAN_SHAP)
    tr, fk = _prep(ref_df, fake_df, num_cols, cat_cols, config.minority_col)
    if tr.iloc[:, -1].nunique() < 2 or fk.iloc[:, -1].nunique() < 2:
        warnings.warn("shap: both ref_df and fake_df need >=2 classes "
                      "(not applicable to minority-only fake data), skipping.")
        return dict(_NAN_SHAP)
    c, s = compute_shap_preservation(tr, fk)
    return {'shap_corr': float(c), 'shap_cos': float(s)}

# ==========================================================
# 下游型： （train_df, test_df, fake_df, ...)
# dcr, tstr
# ==========================================================

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
        #result[f'tstr_delta_{k}'] = sv - rv if not (np.isnan(rv) or np.isnan(sv)) else float('nan')
    return result
