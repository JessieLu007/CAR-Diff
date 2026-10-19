# aggregate_main.py   在 ddpm_med 目录下运行
import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import friedmanchisquare

EVAL_DIR = Path("eval")
OUT_DIR = Path("agg_out"); OUT_DIR.mkdir(exist_ok=True)
OURS_MAIN = "ours_h256_t200_3_main"
OURS_D50 = "ours_h256_t200_3_main-target_epochs50"
USE_D50 = False   # 50 epochs 结果出来并确定采用后改 True

# "prdc_coverage", "prdc_density", "prdc_precision", "prdc_recall",
HIGHER = ["alpha_precision", "shap_corr"] # , "shap_corr" "beta_recall",
# "corr_diff_nn", "corr_diff_nc", "corr_diff_cc"
LOWER = ["c2st_dev","corr_diff_mean","kendall_diff_mean"] #  "tvd_avg",  
METRICS = HIGHER + LOWER

# 1. 读取所有折的 summary.csv，只保留 ref=test
df = pd.concat([pd.read_csv(p) for p in sorted(EVAL_DIR.glob("*/summary.csv"))], ignore_index=True)
df = df[df["ref"] == "test"].copy()
df["c2st_dev"] = (df["c2st_auc"] - 0.5).abs()

# 2. 选定 ours 的版本，并统一命名为 ours
is_ours = df["method"].str.startswith("ours_")
target = np.where((df["dataset"] == "diabetic_data") & USE_D50, OURS_D50, OURS_MAIN)
df = df[~is_ours | (df["method"] == target)].copy()
df.loc[df["method"].str.startswith("ours_"), "method"] = "ours"

# 3. 折内对 5 次采样取均值 -> (dataset, fold, method)
g = df.groupby(["dataset", "fold", "method"])[METRICS].mean()

# 4. 各数据集 mean ± std（跨折）
ms = g.groupby(["dataset", "method"]).agg(["mean", "std"])
ms.columns = [f"{m}_{s}" for m, s in ms.columns]
ms.to_csv(OUT_DIR / "per_dataset_mean_std.csv")
dm = g.groupby(["dataset", "method"])[METRICS].mean()   # 数据集级均值

# 5. 数据集内排名 -> 跨数据集平均排名
ranks = pd.DataFrame({c: dm[c].groupby(level="dataset").rank(ascending=(c in LOWER)) for c in METRICS})
ranks.to_csv(OUT_DIR / "ranks_per_dataset.csv")
avg_rank = ranks.groupby(level="method").mean()
avg_rank["overall"] = avg_rank.mean(axis=1)
avg_rank.sort_values("overall").to_csv(OUT_DIR / "avg_rank.csv")

# 6. ours 对各 baseline：数据集级 win/tie/loss、折级胜率、Friedman（仅描述）
wtl, fold_win, fried = [], [], []
for c in METRICS:
    for level, table in (("dataset", dm[c].unstack("method")), ("fold", g[c].unstack("method"))):
        base = table.drop(columns="ours")
        diff = table["ours"].values[:, None] - base.values
        better = diff > 0 if c in HIGHER else diff < 0
        if level == "dataset":
            for j, b in enumerate(base.columns):
                wtl.append(dict(metric=c, baseline=b, win=int(better[:, j].sum()),
                                tie=int((diff[:, j] == 0).sum()),
                                loss=int((~better[:, j] & (diff[:, j] != 0)).sum())))
        else:
            for j, b in enumerate(base.columns):
                fold_win.append(dict(metric=c, baseline=b, fold_win_rate=better[:, j].mean()))
    p = dm[c].unstack("method").dropna()
    stat, pv = friedmanchisquare(*[p[m] for m in p.columns])
    fried.append(dict(metric=c, n_datasets=len(p), chi2=stat, p=pv))

pd.DataFrame(wtl).to_csv(OUT_DIR / "win_tie_loss.csv", index=False)
pd.DataFrame(fold_win).to_csv(OUT_DIR / "fold_win_rate.csv", index=False)
pd.DataFrame(fried).to_csv(OUT_DIR / "friedman_descriptive.csv", index=False)
print(avg_rank.sort_values("overall").round(2))