# split_tvd.py
import json
import pandas as pd
from pathlib import Path

EVAL_DIR = Path("eval")
OUT_DIR = Path("agg_out"); OUT_DIR.mkdir(exist_ok=True)
OURS_MAIN = "ours_h256_t200_3_main"
OURS_D50 = "ours_h256_t200_3_main-target_epochs50"
USE_D50 = False

idx_all = json.load(open("data/datasets_idx.json", encoding="utf-8"))
keep = {"dataset", "fold", "method", "sample", "ref"}

parts = []
for path in sorted(EVAL_DIR.glob("*/detail.csv")):
    d = pd.read_csv(path, usecols=lambda c: c in keep or c.startswith("tvd_"))
    d = d[d["ref"] == "test"].copy()
    ds = d["dataset"].iloc[0]
    header = [h.strip() for h in pd.read_csv(f"data/{ds}.csv", nrows=0).columns]
    info = idx_all[ds]
    num_cols = ["tvd_" + header[i] for i in info["num_col_idx"]]
    cat_cols = ["tvd_" + header[i] for i in info["cat_col_idx"] + info["target_col_idx"]]
    miss = [c for c in num_cols + cat_cols if c not in d.columns]
    if miss:
        print(f"[warn] {path.parent.name} 缺少列: {miss}")
    d["tvd_num"] = d[[c for c in num_cols if c in d.columns]].mean(axis=1)
    d["tvd_cat"] = d[[c for c in cat_cols if c in d.columns]].mean(axis=1)
    parts.append(d[["dataset", "fold", "method", "sample", "tvd_num", "tvd_cat"]])

df = pd.concat(parts, ignore_index=True)

# 选定 ours 版本并统一命名
is_ours = df["method"].str.startswith("ours_")
target = pd.Series(OURS_MAIN, index=df.index)
if USE_D50:
    target[df["dataset"] == "diabetic_data"] = OURS_D50
df = df[~is_ours | (df["method"] == target)].copy()
df.loc[df["method"].str.startswith("ours_"), "method"] = "ours"

# 折内对采样取均值 -> 数据集内对折取均值 -> 数据集内排名（越小越好）
g = df.groupby(["dataset", "fold", "method"])[["tvd_num", "tvd_cat"]].mean()
dm = g.groupby(["dataset", "method"]).mean()
ranks = dm.groupby(level="dataset").rank(ascending=True)

ranks.to_csv(OUT_DIR / "tvd_split_ranks_per_dataset.csv")
avg = ranks.groupby(level="method").mean().sort_values("tvd_num")
avg.to_csv(OUT_DIR / "tvd_split_avg_rank.csv")
print(avg.round(2))
print("\nours 各数据集排名：")
print(ranks.xs("ours", level="method").round(2))