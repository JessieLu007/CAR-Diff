# scripts/aggregate_variants.py   在项目根目录运行
#   python scripts/aggregate_variants.py ablation | stability | diab_epochs [--ref train]
# 输出 agg_out/<group>/{mean_std.csv, paired_delta.csv}
#   mean_std    : 各数据集、各变体（含参照）的 mean±std（跨折）
#   paired_delta: 逐折配对差值（变体 − 参照）的均值/标准差；gain = 方向调整后的差值，正 = 比参照好
import argparse
import numpy as np, pandas as pd
from pathlib import Path

EVAL_DIR = Path("eval"); OUT_ROOT = Path("agg_out")
KEYS = ["dataset", "fold", "method", "sample"]

HIGHER = ["alpha_precision", "beta_recall", "tstr_roc_auc", "tstr_pr_auc", "tstr_gmean", "tstr_f1",
          "prdc_precision", "prdc_recall", "prdc_density", "prdc_coverage", "shap_corr", "shap_cos"]
LOWER  = ["shape_err", "cpt_err_all", "c2st_dev", "cpt_err_nn", "cpt_err_nc", "cpt_err_cc",
          "corr_diff_nn", "corr_diff_nc", "corr_diff_cc"]
NEUTRAL = ["dcr_mean", "dcr_ratio", "dcr_ks", "dcr_exact"]
SIGN = {**{m: 1 for m in HIGHER}, **{m: -1 for m in LOWER}}
is_down = lambda c: c.startswith(("tstr_", "dcr_"))

# 组: (参照 tag, 变体 tag, 限定数据集)；diabetic_data 的参照统一为 ep200，其 final 标签（ep50）重命名为 ep50
GROUPS = {
    "ablation":    ("final", ["abl_legacy", "abl_nointeract", "abl_sampled", "abl_ar", "abl_nosharedec"], None),
    "stability":   ("final", ["dh128", "dh512", "nt100", "nt500", "nl2", "nl4", "ep100", "ep400"], None),
    "diab_epochs": ("ep200", ["ep12", "ep50"], ["diabetic_data"]),
}
REF_BY_DS = {"diabetic_data": "ep200"}

ap = argparse.ArgumentParser()
ap.add_argument("group", choices=list(GROUPS))
ap.add_argument("--ref", default="train", choices=["train", "test"])
a = ap.parse_args()
ref_tag, variants, only_ds = GROUPS[a.group]

raw = pd.concat([pd.read_csv(p) for p in sorted(EVAL_DIR.glob("*/summary.csv"))], ignore_index=True)
raw["c2st_dev"] = (raw["c2st_auc"] - 0.5).abs()
df = raw[raw["method"].str.startswith("ours_")].copy()
if only_ds:
    df = df[df["dataset"].isin(only_ds)]

known = sorted(set(variants) | {ref_tag, "final", "ep200"}, key=len, reverse=True)
df["tag"] = df["method"].map(lambda m: next((t for t in known if m.endswith("_" + t)), None))
df = df.dropna(subset=["tag"])
df.loc[(df["dataset"] == "diabetic_data") & (df["tag"] == "final"), "tag"] = "ep50"

ALL = [c for c in HIGHER + LOWER + NEUTRAL if c in df.columns]
fid = df[df["ref"] == a.ref][KEYS + ["tag"] + [c for c in ALL if not is_down(c)]]
down = df[df["ref"] == "both"][KEYS + [c for c in ALL if is_down(c)]]
d = fid.merge(down, on=KEYS, how="left")
g = d.groupby(["dataset", "fold", "tag"])[ALL].mean()          # 折内对采样取均值

out = OUT_ROOT / a.group; out.mkdir(parents=True, exist_ok=True)
ms_rows, dl_rows = [], []
for ds, gds in g.groupby(level="dataset"):
    gds = gds.droplevel("dataset")
    ref = REF_BY_DS.get(ds, ref_tag)
    present = set(gds.index.get_level_values("tag"))
    if ref not in present:
        print(f"⚠ {ds}: 缺少参照 {ref}，已跳过"); continue
    r = gds.xs(ref, level="tag")
    for tag in [ref] + [v for v in variants if v in present]:
        v = gds.xs(tag, level="tag")
        for m in ALL:
            ms_rows.append(dict(dataset=ds, variant=tag, metric=m, n_folds=v[m].notna().sum(),
                                mean=v[m].mean(), std=v[m].std(ddof=1)))
        if tag == ref:
            continue
        idx = r.index.intersection(v.index)
        diff = v.loc[idx] - r.loc[idx]
        for m in ALL:
            dm = diff[m].mean()
            dl_rows.append(dict(dataset=ds, variant=tag, metric=m, n_folds=len(idx),
                                delta_mean=dm, delta_std=diff[m].std(ddof=1),
                                gain_mean=SIGN[m] * dm if m in SIGN else np.nan))
pd.DataFrame(ms_rows).to_csv(out / "mean_std.csv", index=False)
pd.DataFrame(dl_rows).to_csv(out / "paired_delta.csv", index=False)
print(f"已写入 {out}")