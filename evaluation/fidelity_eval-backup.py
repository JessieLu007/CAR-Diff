"""
fidelity_eval.py
================
Synthetic data fidelity evaluator for minority manifold learning research.

Design principles
-----------------
- test_df is the primary comparison baseline (not train_df)
- DCR always compares fake vs train AND fake vs test (its core purpose)
- Global mode ('all' | 'minority') with per-metric override via EvalConfig
- Each metric is an independent function registered in METRICS dict
  → add a metric: write one function + one METRICS entry
  → disable a metric: remove its name from the `metrics` list at call time
- evaluate() returns a flat dict → directly usable as a CSV row
- batch_evaluate() handles multiple models with resume-after-interruption support

Quick start
-----------
    from fidelity_eval import evaluate, batch_evaluate, EvalConfig

    config = EvalConfig(
        default_mode='minority',
        minority_col='label',
        minority_val=1,
        metric_modes={'corr_diff': 'all'},   # override corr_diff to use full data
    )

    # Single model
    result = evaluate(
        train_df=train, test_df=test, fake_df=synthetic,
        num_cols=['age', 'sysBP', ...],
        cat_cols=['male', 'currentSmoker', ...],
        config=config,
        model_name='ctgan_v1',
        metrics=['tvd', 'corr_diff', 'alpha_beta', 'c2st', 'dcr'],
        k_neighbors=5,
    )

    # Multiple models → CSV with resume support
    results_df = batch_evaluate(
        models={'ctgan': fake1, 'tvae': fake2, 'smote': fake3},
        train_df=train, test_df=test,
        num_cols=num_cols, cat_cols=cat_cols,
        config=config,
        output_csv='results/eval.csv',
    )

    print_summary(results_df)

Dependencies
------------
    pip install pandas numpy scipy scikit-learn xgboost
    # or: pip install lightgbm  (fallback if xgboost unavailable)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

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


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalConfig:
    """
    Global evaluation configuration with per-metric mode override.

    Parameters
    ----------
    default_mode : 'all' or 'minority'
        'all'      – compare using full datasets
        'minority' – filter datasets to minority class before comparison

    minority_col : column name used to identify the minority class (e.g. 'label')
    minority_val : value that marks the minority class (e.g. 1, True, 'positive')

    metric_modes : per-metric mode overrides.
        Keys are metric names ('tvd', 'corr_diff', 'alpha_beta', 'c2st', 'dcr').
        Values are 'all' or 'minority'.
        Overrides default_mode for that metric only.

    Example
    -------
        config = EvalConfig(
            default_mode='minority',
            minority_col='diabetes',
            minority_val=1,
            metric_modes={'corr_diff': 'all'},  # corr_diff uses full data
        )
    """
    default_mode: Literal['all', 'minority'] = 'all'
    minority_col: Optional[str] = None
    minority_val: Any = None
    metric_modes: dict[str, str] = field(default_factory=dict)

    def get_mode(self, metric: str) -> str:
        """Return effective mode for a given metric (per-metric override or default)."""
        return self.metric_modes.get(metric, self.default_mode)


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _subset(df: pd.DataFrame, mode: str, config: EvalConfig) -> pd.DataFrame:
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
    if mode == 'minority' and config.minority_col is not None:
        n = [c for c in num_cols if c != config.minority_col]
        c = [c for c in cat_cols if c != config.minority_col]
        return n, c
    return list(num_cols), list(cat_cols)


def _encode_scale(
    ref_df: pd.DataFrame,
    *other_dfs: pd.DataFrame,
    cat_cols: list[str],
) -> list[np.ndarray]:
    """
    Encode and normalize multiple DataFrames consistently.

    Steps:
    1. LabelEncode each categorical column, fitted on ref_df.
    2. Z-score normalize using ref_df mean/std as reference.

    Returns a list of numpy arrays: [ref_encoded, *other_encoded].
    Unknown categories in other_dfs are encoded as -1.
    """
    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder().fit(ref_df[col].astype(str))
        encoders[col] = le

    arrays: list[np.ndarray] = []
    for df in (ref_df, *other_dfs):
        enc = df.copy().reset_index(drop=True)
        for col in cat_cols:
            le = encoders[col]
            enc[col] = df[col].astype(str).map(
                lambda v, _le=le: _le.transform([v])[0] if v in _le.classes_ else -1
            )
        arrays.append(enc.values.astype(float))

    mu  = arrays[0].mean(axis=0)
    sig = arrays[0].std(axis=0) + 1e-8
    return [(a - mu) / sig for a in arrays]


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
    Mixed-type association matrix.
    num–num  : Spearman correlation
    cat–cat  : Cramér's V
    num–cat  : η² (eta-squared)
    """
    all_cols = num_cols + cat_cols
    n = len(all_cols)
    C = np.zeros((n, n))
    num_set = set(num_cols)

    for i, ci in enumerate(all_cols):
        for j, cj in enumerate(all_cols):
            if i == j:
                C[i, j] = 1.0
            elif j < i:
                C[i, j] = C[j, i]
            else:
                ai, aj = df[ci].values, df[cj].values
                if ci in num_set and cj in num_set:
                    C[i, j] = _spearman(ai.astype(float), aj.astype(float))
                elif ci not in num_set and cj not in num_set:
                    C[i, j] = _cramers_v(ai, aj)
                else:
                    num_a = ai.astype(float) if ci in num_set else aj.astype(float)
                    cat_a = aj if ci in num_set else ai
                    C[i, j] = _eta_sq(num_a, cat_a)

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
    α-Precision and β-Recall (k-NN coverage, Alaa et al. 2022).

    Comparison : fake_df vs test_df
    Mode       : controls full vs minority subset

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
    Classifier Two-Sample Test (C2ST).

    Trains a gradient boosting classifier to distinguish real (test_df) from
    fake (fake_df) samples. Reports cross-validated AUC and feature importances.

    Comparison : fake_df vs test_df
    Mode       : controls full vs minority subset
    Outputs    : c2st_auc, c2st_fimp_<col> for each feature column
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
    """
    Distance to Closest Record (DCR) — privacy / memorization check.

    Computes, for each fake sample, the L2 distance to the nearest neighbor
    in train_df and in test_df separately.

    If the model memorized training data, fake samples will be systematically
    closer to train_df than to test_df, producing a significant KS difference.

    Mode    : controls which subset of all three DataFrames to use
    Outputs : dcr_train_mean, dcr_test_mean, dcr_ks_stat, dcr_ks_pval

    Interpretation :
        dcr_train_mean ≈ dcr_test_mean → no memorization signal
        dcr_ks_pval > 0.05             → distributions not significantly different
                                          → no obvious privacy risk
        dcr_ks_pval ≪ 0.05            → training data may be memorized
    """
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


# ══════════════════════════════════════════════════════════════════════════════
# Metric registry
# ══════════════════════════════════════════════════════════════════════════════

METRICS: dict[str, Callable] = {
    'tvd':        metric_tvd,        # Layer 1 – marginal fidelity
    'corr_diff':  metric_corr_diff,  # Layer 1 – dependency structure
    'alpha_beta': metric_alpha_beta, # Layer 2 – minority coverage
    'c2st':       metric_c2st,       # Layer 3 – joint distribution gap
    'dcr':        metric_dcr,        # Layer 3 – privacy / memorization
}

# Default evaluation order (controls column order in output CSV)
DEFAULT_METRICS: list[str] = list(METRICS.keys())


def register_metric(name: str, fn: Callable) -> None:
    """
    Register a custom metric function.

    The function must accept:
        (train_df, test_df, fake_df, num_cols, cat_cols, config, **kwargs)
    and return a flat dict of metric name → float.

    Example
    -------
        def metric_ks(train_df, test_df, fake_df, num_cols, cat_cols, config, **kwargs):
            mode = config.get_mode('ks')
            ref  = _subset(test_df, mode, config)
            fake = _subset(fake_df, mode, config)
            results = {}
            for col in num_cols:
                stat, _ = stats.ks_2samp(ref[col].values, fake[col].values)
                results[f'ks_{col}'] = stat
            results['ks_avg'] = float(np.mean(list(results.values())))
            return results

        register_metric('ks', metric_ks)
    """
    METRICS[name] = fn
    if name not in DEFAULT_METRICS:
        DEFAULT_METRICS.append(name)


# ══════════════════════════════════════════════════════════════════════════════
# Main entry points
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    train_df:   pd.DataFrame,
    test_df:    pd.DataFrame,
    fake_df:    pd.DataFrame,
    num_cols:   list[str],
    cat_cols:   list[str],
    config:     EvalConfig,
    model_name: Optional[str] = None,
    metrics:    Optional[list[str]] = None,
    **metric_kwargs,
) -> dict:
    """
    Evaluate a single synthetic dataset. Returns a flat dict of metric values.

    Parameters
    ----------
    train_df      : original training data
    test_df       : original test / holdout data  ← primary comparison baseline
    fake_df       : synthetic dataset to evaluate
    num_cols      : continuous feature column names
    cat_cols      : categorical feature column names
    config        : EvalConfig (global mode + per-metric overrides)
    model_name    : optional label added to result dict (e.g. 'ctgan_v1')
    metrics       : which metrics to run; None = all DEFAULT_METRICS.
                    pass a subset to skip unwanted metrics, e.g. ['tvd', 'dcr']
    **metric_kwargs : forwarded to metric functions.
                      e.g. k_neighbors=10, n_folds=3, max_samples=1000

    Returns
    -------
    Flat dict suitable for pd.DataFrame([result]) or CSV row.
    """
    to_run = metrics if metrics is not None else DEFAULT_METRICS
    result: dict = {} if model_name is None else {'model_name': model_name}

    for name in to_run:
        if name not in METRICS:
            warnings.warn(f"evaluate: unknown metric '{name}', skipping.")
            continue
        try:
            row = METRICS[name](
                train_df=train_df,
                test_df=test_df,
                fake_df=fake_df,
                num_cols=num_cols,
                cat_cols=cat_cols,
                config=config,
                **metric_kwargs,
            )
            result.update(row)
        except Exception as exc:
            warnings.warn(f"evaluate: metric '{name}' raised an error: {exc}")

    return result


def batch_evaluate(
    models:     dict[str, pd.DataFrame],
    train_df:   pd.DataFrame,
    test_df:    pd.DataFrame,
    num_cols:   list[str],
    cat_cols:   list[str],
    config:     EvalConfig,
    output_csv: str,
    metrics:    Optional[list[str]] = None,
    verbose:    bool = True,
    **metric_kwargs,
) -> pd.DataFrame:
    """
    Evaluate multiple synthetic datasets and save results to CSV.

    Resume support: if output_csv already exists, models already present in its
    'model_name' column are skipped. Each model's result is appended immediately
    after evaluation, so interruptions lose at most one model's work.

    Parameters
    ----------
    models      : {model_name: fake_df}  — one entry per synthetic dataset
    train_df    : original training data
    test_df     : original test / holdout data
    num_cols    : continuous feature column names
    cat_cols    : categorical feature column names
    config      : EvalConfig
    output_csv  : path to output CSV (created or appended)
    metrics     : metrics to run (None = all DEFAULT_METRICS)
    verbose     : print progress per model
    **metric_kwargs : forwarded to evaluate() / individual metrics

    Returns
    -------
    Full results DataFrame (all models, including previously computed ones).

    Example
    -------
        results = batch_evaluate(
            models={'ctgan': fake_ctgan, 'tvae': fake_tvae},
            train_df=train, test_df=test,
            num_cols=num_cols, cat_cols=cat_cols,
            config=config,
            output_csv='results/eval.csv',
            k_neighbors=5,
        )
        print_summary(results)
    """
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # identify already-computed models
    done: set[str] = set()
    if out.exists() and out.stat().st_size > 0:
        try:
            existing = pd.read_csv(out)
            if 'model_name' in existing.columns:
                done = set(existing['model_name'].astype(str))
        except Exception:
            pass  # corrupt CSV → recompute everything

    for name, fake_df in models.items():
        if name in done:
            if verbose:
                print(f"[skip]  {name}")
            continue

        if verbose:
            print(f"[eval]  {name} ...", end=' ', flush=True)

        try:
            row = evaluate(
                train_df=train_df,
                test_df=test_df,
                fake_df=fake_df,
                num_cols=num_cols,
                cat_cols=cat_cols,
                config=config,
                model_name=name,
                metrics=metrics,
                **metric_kwargs,
            )
            row_df = pd.DataFrame([row])
            write_header = not out.exists() or out.stat().st_size == 0
            row_df.to_csv(out, mode='a', header=write_header, index=False)
            done.add(name)
            if verbose:
                print("done")
        except Exception as exc:
            warnings.warn(f"\nbatch_evaluate: model '{name}' failed: {exc}")
            if verbose:
                print(f"FAILED ({exc})")

    return pd.read_csv(out) if (out.exists() and out.stat().st_size > 0) else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# Display utilities
# ══════════════════════════════════════════════════════════════════════════════

# Key summary columns (in preferred display order, excludes per-column details)
_SUMMARY_COLS = [
    'model_name',
    'tvd_avg',
    'corr_diff_mean', 'corr_diff_fro',
    'alpha_precision', 'beta_recall',
    'c2st_auc',
    'dcr_train_mean', 'dcr_test_mean', 'dcr_ks_stat', 'dcr_ks_pval',
]


def print_summary(
    results: 'dict | pd.DataFrame',
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> None:
    """
    Print a concise summary table of key metrics.

    Parameters
    ----------
    results   : output of evaluate() (dict) or batch_evaluate() (DataFrame)
    sort_by   : column name to sort by (e.g. 'tvd_avg', 'c2st_auc')
    ascending : sort direction

    Per-column TVD and C2ST feature importances are excluded from the summary.
    Access them via the full DataFrame or CSV.
    """
    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = results.copy()

    show = [c for c in _SUMMARY_COLS if c in df.columns]
    display = df[show]

    if sort_by and sort_by in display.columns:
        display = display.sort_values(sort_by, ascending=ascending)

    float_cols = [c for c in show if c != 'model_name']
    fmt = {c: lambda v: f'{v:.4f}' if pd.notna(v) else 'nan' for c in float_cols}

    sep = '─' * max(60, len(show) * 12)
    print(sep)
    print('  Fidelity Evaluation Summary')
    print(sep)
    print(display.to_string(index=False, formatters=fmt))
    print(sep)


def summary_df(results: 'dict | pd.DataFrame') -> pd.DataFrame:
    """Return a DataFrame with only the key summary columns (no per-column details)."""
    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = results.copy()
    show = [c for c in _SUMMARY_COLS if c in df.columns]
    return df[show].reset_index(drop=True)
