# scripts/aggregate_main.py   在项目根目录运行：python scripts/aggregate_main.py
import numpy as np, pandas as pd
from pathlib import Path

EVAL_DIR = Path("eval")
OUT_ROOT = Path("agg_out"); OUT_ROOT.mkdir(exist_ok=True)
OURS_MAIN = "ours_h256_t200_3_abl_ar" # "ours_h256_t200_3_final"
OURS_BY_DS = {} # {"diabetic_data": "ours_h256_t200_3_ep200"}   # 统一 200 epochs 协议；diabetic 的 final 标签是 ep50，仅作敏感性分析

KEYS = ["dataset", "fold", "method", "sample"]

# 主表：参与排名
HIGHER = ["alpha_precision", "beta_recall", "tstr_roc_auc", "tstr_pr_auc", "tstr_gmean"]
LOWER  = ["shape_err", "cpt_err_all", "c2st_dev"]
# 附录：只输出 mean±std
EXTRA_HIGHER = ["prdc_precision", "prdc_recall", "prdc_density", "prdc_coverage",
                "tstr_f1", "shap_corr", "shap_cos"]            # shap 只在 ref=train 有值
EXTRA_LOWER  = ["cpt_err_nn", "cpt_err_nc", "cpt_err_cc",
                "corr_diff_nn", "corr_diff_nc", "corr_diff_cc"]
# 无单调方向：不排名，只输出 mean±std（dcr_ratio ≈ 1 为宜，明显 < 1 疑似记忆）
NEUTRAL = ["dcr_mean", "dcr_ratio", "dcr_ks", "dcr_exact"]

is_down = lambda c: c.startswith(("tstr_", "dcr_"))        # 来自 ref=both 行

# 1. 读取所有折的 summary.csv
raw = pd.concat([pd.read_csv(p) for p in sorted(EVAL_DIR.glob("*/summary.csv"))], ignore_index=True)
raw["c2st_dev"] = (raw["c2st_auc"] - 0.5).abs()
real = raw[raw["method"] == "real"]                         # TRTR / holdout DCR 基线，不参与排名
df = raw[raw["method"] != "real"].copy()

# 2. 选定 ours 的版本，并统一命名为 ours
is_ours = df["method"].str.startswith("ours_")
target = df["dataset"].map(OURS_BY_DS).fillna(OURS_MAIN)
df = df[~is_ours | (df["method"] == target)].copy()
df.loc[df["method"].str.startswith("ours_"), "method"] = "ours"

ALL = HIGHER + LOWER + EXTRA_HIGHER + EXTRA_LOWER + NEUTRAL
miss = [c for c in ALL if c not in df.columns]
if miss:
    print("⚠ 以下列不存在（未评估），已跳过:", miss)
ALL = [c for c in ALL if c in df.columns]
HIGHER = [c for c in HIGHER if c in ALL]; LOWER = [c for c in LOWER if c in ALL]
METRICS = HIGHER + LOWER


def run(ref):
    out = OUT_ROOT / ref; out.mkdir(exist_ok=True)
    # 保真指标取 ref 行，下游指标（tstr_/dcr_）取 ref=both 行，按 (dataset, fold, method, sample) 对齐
    fid = df[df["ref"] == ref][KEYS + [c for c in ALL if not is_down(c)]]
    down = df[df["ref"] == "both"][KEYS + [c for c in ALL if is_down(c)]]
    d = fid.merge(down, on=KEYS, how="left")

    # 3. 折内对多次采样取均值 -> (dataset, fold, method)
    g = d.groupby(["dataset", "fold", "method"])[ALL].mean()

    # 4. 各数据集 mean ± std（跨折）
    ms = g.groupby(["dataset", "method"]).agg(["mean", "std"])
    ms.columns = [f"{m}_{s}" for m, s in ms.columns]
    ms.to_csv(out / "per_dataset_mean_std.csv")
    dm = g.groupby(["dataset", "method"])[ALL].mean()       # 数据集级均值

    # 5. 数据集内排名；跨数据集平均排名仅作描述性汇总（论文是否使用自行决定）
    ranks = pd.DataFrame({c: dm[c].groupby(level="dataset").rank(ascending=(c in LOWER)) for c in METRICS})
    ranks.to_csv(out / "ranks_per_dataset.csv")
    avg_rank = ranks.groupby(level="method").mean()
    avg_rank["overall"] = avg_rank.mean(axis=1)
    avg_rank.sort_values("overall").to_csv(out / "avg_rank.csv")
    print(f"[ref={ref}]"); print(avg_rank.sort_values("overall").round(2))


for ref in ("train", "test"):          # train 为主表，test 为附录对照
    run(ref)

# 6. real 基线（TRTR 与 holdout→train DCR），与 ref 无关
rcols = [c for c in ALL if is_down(c) and c in real.columns]
rb = real[real["ref"] == "both"].groupby("dataset")[rcols].agg(["mean", "std"])
rb.columns = [f"{m}_{s}" for m, s in rb.columns]
rb.to_csv(OUT_ROOT / "real_baseline.csv")