python - <<'PY'
from pathlib import Path
import pandas as pd
from evaluation.paths import load_fold, load_meta
from evaluation.align import make_aligner
from evaluation.metrics import (
    EvalConfig, metric_tvd, metric_corr_diff,
    metric_c2st, metric_prdc, metric_alpha_beta, metric_dcr
)
from evaluation import metrics as M

root = Path.cwd()
ds, fold = "hepatitis", 0

meta = load_meta(root / "data" / "datasets_info.json", ds)
train, test = load_fold(str(root), ds, fold)
align = make_aligner(train, meta)
cfg = EvalConfig(
    default_mode="all",
    minority_col=meta["label_col"],
    minority_val=meta["minority_val"],
)

dirs = {
    "inputnormal": root / "results/hepatitis/seed_2026_fold_0/"
        "full_data_h256_t200_3_main_indexfix_1017_225103/synthetic_data.csv",
    "catzero": root / "results/hepatitis/seed_2026_fold_0/"
        "full_data_h256_t200_3_main_catzero_1017_225103/synthetic_data.csv",
}

results = {}
for name, path in dirs.items():
    M._REF_CORR.clear()
    fake = align(pd.read_csv(path))
    results[name] = {}

    for ref_name, ref in [("train", train), ("test", test)]:
        kw = dict(
            ref_df=ref, fake_df=fake,
            num_cols=meta["num_cols"], cat_cols=meta["cat_cols"],
            config=cfg,
        )
        row = {}
        for fn in [
            metric_tvd, metric_corr_diff, metric_c2st,
            metric_prdc, metric_alpha_beta,
        ]:
            row.update(fn(**kw))
        results[name][ref_name] = row

    results[name]["both"] = metric_dcr(
        train_df=train, test_df=test, fake_df=fake,
        num_cols=meta["num_cols"], cat_cols=meta["cat_cols"],
        config=cfg,
    )

keys = [
    "tvd_avg", "corr_diff_fro", "corr_diff_nn",
    "corr_diff_nc", "corr_diff_cc", "c2st_auc",
    "alpha_precision", "beta_recall",
    "dcr_train_mean", "dcr_test_mean",
]

for ref in ["train", "test", "both"]:
    print(f"\n===== {ref} =====")
    a, b = results["inputnormal"][ref], results["catzero"][ref]
    for k in keys:
        if k in a and k in b:
            print(f"{k:20s} normal={a[k]:.6f}  zero={b[k]:.6f}"
                  f"  delta={b[k]-a[k]:+.6f}")
PY