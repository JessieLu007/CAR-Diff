"""
CAR-Diff 显著性检验 v2 —— 基于逐数据集长表 metrics_long_train.csv / metrics_long_test.csv
=========================================================================
长表列：ref, dataset, method, metric, mean, std, rank
5 个主排名指标：shape_err, cpt_err_all, c2st_dev, alpha_precision, beta_recall
7 个主排名方法：SMOTE, TabKDE, CTGAN, TVAE, TabDDPM, TabSyn, CAR-Diff
另有 CAR-Diff w/o AR、Real 两个方法不参与主排名，用于 AR 头消融的单独检验。

Part A: Friedman omnibus（每个指标单独一次，k=7 方法，N=6 数据集）
Part B: Wilcoxon signed-rank，CAR-Diff vs 每个基线（每个指标单独一次，6 次配对比较 + Holm 校正）
Part C: Wilcoxon signed-rank，CAR-Diff vs CAR-Diff w/o AR（AR 头消融，5 个指标各一次 + Holm 校正）
Part D: 用重建的 rank 验证与 rank_train.csv/rank_test.csv 的平均排名一致（sanity check）
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare, wilcoxon, chi2

MAIN_METRICS = ["shape_err", "cpt_err_all", "c2st_dev", "alpha_precision", "beta_recall"]
MAIN_METHODS = ["SMOTE", "TabKDE", "CTGAN", "TVAE", "TabDDPM", "TabSyn", "CAR-Diff"]
BASE_DIR = Path(__file__).resolve().parent
DIR = BASE_DIR.parent / "paper_out" / "tables"

def load_long(path):
    df = pd.read_csv(path)
    return df


def pivot_metric(df, metric, methods=MAIN_METHODS):
    sub = df[(df["metric"] == metric) & (df["method"].isin(methods))]
    return sub.pivot(index="dataset", columns="method", values="mean")[methods]


def holm_correct(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank_i, idx in enumerate(order):
        val = min(1.0, pvals[idx] * (m - rank_i))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


# ---------------------------------------------------------------------
# Part A: Friedman omnibus（scipy 直接在原始值上做，内部按数据集分别排名）
# ---------------------------------------------------------------------

def friedman_per_metric(df, metrics=MAIN_METRICS, methods=MAIN_METHODS):
    rows = []
    for metric in metrics:
        wide = pivot_metric(df, metric, methods).dropna()
        stat, p = friedmanchisquare(*[wide[m].values for m in methods])
        rows.append({"metric": metric, "N": len(wide), "k": len(methods),
                      "chi2": round(stat, 3), "p": round(p, 4)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Part B: Wilcoxon, CAR-Diff vs 每个基线（按 raw mean 值配对，每个指标单独一轮）
# ---------------------------------------------------------------------

def wilcoxon_vs_baselines(df, metric, target="CAR-Diff", methods=MAIN_METHODS):
    wide = pivot_metric(df, metric, methods).dropna()
    baselines = [m for m in methods if m != target]
    rows = []
    for b in baselines:
        diff = wide[target] - wide[b]
        if (diff == 0).all():
            rows.append({"baseline": b, "N": len(diff), "stat": np.nan, "p": 1.0})
            continue
        stat, p = wilcoxon(wide[target], wide[b], zero_method="wilcox")
        rows.append({"baseline": b, "N": len(diff), "stat": stat, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correct(out["p"].values)
    out["metric"] = metric
    return out[["metric", "baseline", "N", "stat", "p", "p_holm"]]


# ---------------------------------------------------------------------
# Part C: Wilcoxon, CAR-Diff vs CAR-Diff w/o AR（AR 头消融，5 个指标 + Holm 校正）
# ---------------------------------------------------------------------

def wilcoxon_ar_ablation(df, metrics=MAIN_METRICS):
    rows = []
    for metric in metrics:
        sub = df[(df["metric"] == metric) & (df["method"].isin(["CAR-Diff", "CAR-Diff w/o AR"]))]
        wide = sub.pivot(index="dataset", columns="method", values="mean").dropna()
        diff = wide["CAR-Diff"] - wide["CAR-Diff w/o AR"]
        if (diff == 0).all():
            rows.append({"metric": metric, "N": len(diff), "stat": np.nan, "p": 1.0})
            continue
        stat, p = wilcoxon(wide["CAR-Diff"], wide["CAR-Diff w/o AR"], zero_method="wilcox")
        rows.append({"metric": metric, "N": len(diff), "stat": stat, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_correct(out["p"].values)
    return out


# ---------------------------------------------------------------------
# Part D: 用重建的逐数据集 rank 复核平均排名，和 rank_train.csv / rank_test.csv 对照
# ---------------------------------------------------------------------

HIGHER_IS_BETTER = {"alpha_precision", "beta_recall"}


def rebuild_avg_rank(df, metrics=MAIN_METRICS, methods=MAIN_METHODS):
    out = {}
    for metric in metrics:
        wide = pivot_metric(df, metric, methods)
        ascending = metric not in HIGHER_IS_BETTER
        ranks = wide.rank(axis=1, method="average", ascending=ascending)
        out[metric] = ranks.mean(axis=0)
    rank_df = pd.DataFrame(out)
    rank_df["Overall"] = rank_df.mean(axis=1)
    return rank_df.loc[methods]


# ---------------------------------------------------------------------
# Part E: Nemenyi critical difference（基于修正后的 Overall 平均排名）
# ---------------------------------------------------------------------

NEMENYI_Q_ALPHA_005 = {7: 2.949}  # Demsar (2006) Table 5；写入论文前请对照原文核实


def nemenyi_compare(df, methods=MAIN_METHODS, target="CAR-Diff"):
    k = len(methods)
    N = pivot_metric(df, MAIN_METRICS[0], methods).shape[0]
    cd = NEMENYI_Q_ALPHA_005[k] * np.sqrt(k * (k + 1) / (6 * N))
    overall = rebuild_avg_rank(df, methods=methods)["Overall"]
    rows = []
    for m in methods:
        if m == target:
            continue
        diff = overall[m] - overall[target]  # 正数表示该基线平均排名更差（CAR-Diff 更优）
        rows.append({"baseline": m, "rank_diff": round(diff, 3),
                      "significant_at_CD": abs(diff) > cd})
    out = pd.DataFrame(rows)
    return cd, out


if __name__ == "__main__":
    train_df = load_long(DIR / "metrics_long_train.csv")
    test_df  = load_long(DIR / "metrics_long_test.csv")

    print("########## TRAIN reference ##########")
    print("=== Part A: Friedman omnibus (k=7, N=6) ===")
    print(friedman_per_metric(train_df))
    print()
    print("=== Part B: Wilcoxon, CAR-Diff vs each baseline ===")
    for metric in MAIN_METRICS:
        print(wilcoxon_vs_baselines(train_df, metric))
    print()
    print("=== Part C: Wilcoxon, CAR-Diff vs CAR-Diff w/o AR ===")
    print(wilcoxon_ar_ablation(train_df))
    print()
    print("=== Part D: rebuilt avg rank (sanity check vs rank_train.csv) ===")
    print(rebuild_avg_rank(train_df).round(3))
    print()
    print("=== Part E: Nemenyi CD vs CAR-Diff (Overall rank) ===")
    cd, cmp_df = nemenyi_compare(train_df)
    print(f"CD = {cd:.3f}")
    print(cmp_df)

    print()
    print("########## TEST reference ##########")
    print("=== Part A: Friedman omnibus (k=7, N=6) ===")
    print(friedman_per_metric(test_df))
    print()
    print("=== Part B: Wilcoxon, CAR-Diff vs each baseline ===")
    for metric in MAIN_METRICS:
        print(wilcoxon_vs_baselines(test_df, metric))
    print()
    print("=== Part C: Wilcoxon, CAR-Diff vs CAR-Diff w/o AR ===")
    print(wilcoxon_ar_ablation(test_df))
    print()
    print("=== Part D: rebuilt avg rank (sanity check vs rank_test.csv) ===")
    print(rebuild_avg_rank(test_df).round(3))
    print()
    print("=== Part E: Nemenyi CD vs CAR-Diff (Overall rank) ===")
    cd, cmp_df = nemenyi_compare(test_df)
    print(f"CD = {cd:.3f}")
    print(cmp_df)