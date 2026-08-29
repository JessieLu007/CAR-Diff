# evaluation/fidelity_eval.py
from __future__ import annotations
import warnings
from pathlib import Path
from typing import Callable,Optional

import pandas as pd

from evaluation.fidelity_config import EvalConfig
from evaluation.fidelity_metrics import (
    metric_tvd, metric_corr_diff, metric_alpha_beta, metric_c2st,
    metric_dcr, metric_cls, metric_tstr, metric_kendall
)
 

# ══════════════════════════════════════════════════════════════════════════════
# Metric registry
# ══════════════════════════════════════════════════════════════════════════════

METRICS: dict[str, Callable] = {
    'tvd':        metric_tvd,        # Layer 1 – marginal fidelity
    'corr_diff':  metric_corr_diff,  # Layer 1 – dependency structure
    'alpha_beta': metric_alpha_beta, # Layer 2 – minority coverage
    'c2st':       metric_c2st,       # Layer 3 – joint distribution gap
    'dcr':        metric_dcr,        # Layer 3 – privacy / memorization
    'cls':        metric_cls,        # Layer 4 - downstream classification utility
    'tstr':       metric_tstr,
    'kendall':    metric_kendall,
}

# Default evaluation order (controls column order in output CSV)
DEFAULT_METRICS: list[str] = list(METRICS.keys())


def register_metric(name: str, fn: Callable) -> None:

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
    if config.pre_filtered and config.minority_col is not None:
        for tag, df in [('train', train_df), ('test', test_df), ('fake', fake_df)]:
            unique_vals = df[config.minority_col].unique
            if len(unique_vals) > 1:
                warnings.warn(
                    f"evaluate: pre_filtered=True, but '{tag}_df' contains"
                    f"multiple values in '{config.minority_col}': {unique_vals}."
                    f"Did you forget to pre-filter?"
                )
    
    to_run = metrics if metrics is not None else DEFAULT_METRICS
    result: dict = {} if model_name is None else {'model_name': model_name}

    for name in to_run:
        if name not in METRICS:
            warnings.warn(f"evaluate: unknown metric '{name}', skipping.")
            continue
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

    return result


import warnings
from pathlib import Path
import pandas as pd

def batch_evaluate(
    models:     dict[str, pd.DataFrame],
    train_df:   pd.DataFrame,
    test_df:    pd.DataFrame,
    num_cols:   list[str],
    cat_cols:   list[str],
    config,
    output_csv: str,
    metrics:    Optional[list[str]] = None,
    verbose:    bool = True,
    **metric_kwargs,
) -> pd.DataFrame:
    
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---------- 读取已有结果（如果存在且完整）----------
    existing_df = None
    if out.exists() and out.stat().st_size > 0:
        try:
            existing_df = pd.read_csv(out)
            if 'model_name' not in existing_df.columns:
                existing_df = None
        except Exception as e:
            warnings.warn(f"Existing CSV {out} corrupted ({e}), will rebuild all.")
            out.unlink()
            existing_df = None

    # 收集所有行（DataFrame 列表）
    dfs_to_concat = []
    evaluated_models = set()

    if existing_df is not None:
        dfs_to_concat.append(existing_df)
        evaluated_models.update(existing_df['model_name'].astype(str))

    # ---------- 评估新模型 ----------
    for name, fake_df in models.items():
        if name in evaluated_models:
            if verbose:
                print(f"[skip]  {name}")
            continue

        if verbose:
            print(f"[eval]  {name} ...", end=' ', flush=True)


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
        dfs_to_concat.append(pd.DataFrame([row]))
        evaluated_models.add(name)
        if verbose:
            print("done")
    

    # ---------- 合并并一次性写入 ----------
    if not dfs_to_concat:
        return pd.DataFrame()

    final_df = pd.concat(dfs_to_concat, ignore_index=True)
    # 强制所有字段加引号，避免字段内逗号/换行影响
    final_df.to_csv(out, index=False, quoting=1)  # quoting=1 = csv.QUOTE_ALL

    return final_df


def batch_evaluate_old(
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
    
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # identify already-computed models
    done: set[str] = set()
    if out.exists() and out.stat().st_size > 0:
        existing = pd.read_csv(out)
        if 'model_name' in existing.columns:
            done = set(existing['model_name'].astype(str))

    for name, fake_df in models.items():
        if name in done:
            if verbose:
                print(f"[skip]  {name}")
            continue

        if verbose:
            print(f"[eval]  {name} ...", end=' ', flush=True)

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
    # cls: baseline (constant across models — reference point)
    'cls_baseline_acc',     'cls_baseline_recall', 'cls_baseline_f1',
    'cls_baseline_gmean',   'cls_baseline_roc_auc','cls_baseline_pr_auc',
    # cls: augmented (varies per model — primary comparison target)
    'cls_aug_acc',          'cls_aug_recall',      'cls_aug_f1',
    'cls_aug_gmean',        'cls_aug_roc_auc',     'cls_aug_pr_auc',
    # cls: delta = aug − baseline (most actionable signal)
    'cls_delta_acc',        'cls_delta_recall',    'cls_delta_f1',
    'cls_delta_gmean',      'cls_delta_roc_auc',   'cls_delta_pr_auc',
    # kendall: 仅 summary（per-pair 列太多，不入摘要表）
    'kendall_diff_mean', 'kendall_diff_fro',
    # tstr
    'tstr_trtr_acc', 'tstr_trtr_recall', 'tstr_trtr_f1',
    'tstr_trtr_gmean', 'tstr_trtr_roc_auc', 'tstr_trtr_pr_auc',
    'tstr_tstr_acc', 'tstr_tstr_recall', 'tstr_tstr_f1',
    'tstr_tstr_gmean', 'tstr_tstr_roc_auc', 'tstr_tstr_pr_auc',
    'tstr_delta_acc', 'tstr_delta_recall', 'tstr_delta_f1',
    'tstr_delta_gmean', 'tstr_delta_roc_auc', 'tstr_delta_pr_auc',
]

import numpy as np

_CLS_SFXS = ('acc', 'recall', 'f1', 'gmean', 'roc_auc', 'pr_auc')

# 格式: metric_key → (baseline_prefix, aug_prefix, delta_prefix, aug_label, baseline_label)
_CLF_SECTIONS: dict[str, tuple[str, str, str, str, str]] = {
    'cls':  ('cls_baseline', 'cls_aug',  'cls_delta', 'Aug (minority)',  'TRTR baseline'),
    'tstr': ('tstr_trtr',   'tstr_tstr', 'tstr_delta', 'TSTR',          'TRTR baseline'),
}


def _fmt_delta(v: float) -> str:
    if np.isnan(v):  return '    nan'
    return f'  +{v:.4f}' if v > 0 else f'   {v:.4f}'


def _print_clf_section(
    results_df: pd.DataFrame,
    section_key: str,
    baseline_prefix: str,
    aug_prefix: str,
    delta_prefix: str,
    aug_label: str,
    baseline_label: str,
) -> None:
    models = results_df['model_name'].tolist()
    first  = results_df.iloc[0]

    # ── Baseline：只打印一次 ──────────────────────────────────────────────────
    print(f"\n{'━'*70}")
    print(f"  {section_key.upper()}")
    print(f"  {baseline_label}: ", end='')
    parts = [
        f"{s}={first.get(f'{baseline_prefix}_{s}', float('nan')):.4f}"
        for s in _CLS_SFXS
    ]
    print('  '.join(parts))

    # ── Aug / TSTR：metric 为行，model 为列 ───────────────────────────────────
    aug_df = pd.DataFrame(
        {m: {s: results_df.loc[results_df.model_name == m,
                               f'{aug_prefix}_{s}'].iat[0]
             for s in _CLS_SFXS}
         for m in models}
    ).round(4)
    print(f"\n  {aug_label}:")
    print(aug_df.to_string())

    # ── Delta：metric 为行，model 为列，正值显示 + ────────────────────────────
    delta_df = pd.DataFrame(
        {m: {s: results_df.loc[results_df.model_name == m,
                               f'{delta_prefix}_{s}'].iat[0]
             for s in _CLS_SFXS}
         for m in models}
    )
    delta_fmt = delta_df.map(_fmt_delta)
    print(f"\n  Δ  ({aug_label} − {baseline_label}):")
    print(delta_fmt.to_string())


def print_summary(results_df: pd.DataFrame) -> None:
    # ── 常规 metrics：一行一个 model 的宽表 ──────────────────────────────────
    clf_prefixes = {p for (bp, ap, dp, _, __) in _CLF_SECTIONS.values()
                    for p in (bp, ap, dp)}
    std_cols = [
        c for c in _SUMMARY_COLS if c in results_df.columns
        and not any(c.startswith(p + '_') for p in clf_prefixes)
    ]
    if std_cols:
        print("\n=== Evaluation Summary ===")
        print(results_df[std_cols].to_string(index=False))

    # ── 分类 metrics：独立 section，metric 为行，model 为列 ───────────────────
    for key, (bp, ap, dp, al, bl) in _CLF_SECTIONS.items():
        if f'{ap}_acc' in results_df.columns:
            _print_clf_section(results_df, key, bp, ap, dp, al, bl)

    print()


'''def print_summary(
    results: 'dict | pd.DataFrame',
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> None:

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
    print(display.to_string(index=False, formatters=fmt))'''



def summary_df(results: 'dict | pd.DataFrame') -> pd.DataFrame:
    """Return a DataFrame with only the key summary columns (no per-column details)."""
    if isinstance(results, dict):
        df = pd.DataFrame([results])
    else:
        df = results.copy()
    show = [c for c in _SUMMARY_COLS if c in df.columns]
    return df[show].reset_index(drop=True)