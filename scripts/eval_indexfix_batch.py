
import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.paths import load_fold, load_meta
from evaluation.align import make_aligner
from evaluation.metrics import (
    EvalConfig,
    metric_tvd,
    metric_corr_diff,
    metric_c2st,
    metric_prdc,
    metric_alpha_beta,
    metric_dcr,
)
from evaluation import metrics as M


PAIRS = [
    ("hepatitis", 0, "1017_225103"),
    ("hepatitis", 1, "1017_225113"),
    ("chronic_kidney_disease", 0, "1017_221925"),
    ("chronic_kidney_disease", 1, "1017_221949"),
    ("framingham", 0, "1017_222303"),
    ("framingham", 1, "1017_222539"),
]

METRIC_FUNCS = [
    metric_tvd,
    metric_corr_diff,
    metric_c2st,
    metric_prdc,
    metric_alpha_beta,
]

def sample_filename(j):
    return "synthetic_data.csv" if j == 0 else f"synthetic_data_s{j}.csv"


def read_info(path):
    info_path = path.parent / "resample_info.txt"
    if not info_path.exists():
        raise FileNotFoundError(f"缺少采样记录：{info_path}")

    info = {}
    for line in info_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v

    if "base_seed" not in info:
        raise ValueError(f"采样记录没有 base_seed：{info_path}")

    return info


def to_float(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def evaluate_file(path, train, test, meta, align, config):
    M._REF_CORR.clear()
    fake = align(pd.read_csv(path))
    result = {}

    for ref_name, ref_df in [("train", train), ("test", test)]:
        kw = dict(
            ref_df=ref_df,
            fake_df=fake,
            num_cols=meta["num_cols"],
            cat_cols=meta["cat_cols"],
            config=config,
        )

        row = {}
        for fn in METRIC_FUNCS:
            row.update(fn(**kw))
        result[ref_name] = row

    result["both"] = metric_dcr(
        train_df=train,
        test_df=test,
        fake_df=fake,
        num_cols=meta["num_cols"],
        cat_cols=meta["cat_cols"],
        config=config,
    )

    return result


def main():
    # 保存每个 j 的配对结果
    rows = []

    for ds, fold, ts in PAIRS:
        fold_dir = ROOT / "results" / ds / f"seed_2026_fold_{fold}"

        legacy_dir = fold_dir / Path(
            f"full_data_h256_t200_3_main_{ts}"
        )
        fixed_dir = fold_dir / Path(
            f"full_data_h256_t200_3_main_indexfix_{ts}"
        )

        legacy_files = {
            j: legacy_dir / sample_filename(j) for j in range(5)
        }
        fixed_files = {
            j: fixed_dir / sample_filename(j) for j in range(5)
        }

        # 先检查文件齐全，避免跑到一半才发现缺样本
        missing = [
            p for p in list(legacy_files.values()) + list(fixed_files.values())
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "缺少以下采样文件，请先补采样：\n" +
                "\n".join(str(p) for p in missing)
            )

        legacy_info = read_info(legacy_files[0])
        fixed_info = read_info(fixed_files[0])

        for name, info in [
            ("legacy", legacy_info),
            ("indexfix", fixed_info),
        ]:
            if int(info["base_seed"]) != 2026:
                raise ValueError(
                    f"{ds} fold{fold} 的 {name} base_seed="
                    f"{info['base_seed']}，预期为 2026"
                )

        print(f"\n{'=' * 65}")
        print(f"{ds} fold {fold} | timestamp={ts}")

        meta = load_meta(ROOT / "data" / "datasets_info.json", ds)
        train, test = load_fold(str(ROOT), ds, fold)
        align = make_aligner(train, meta)

        config = EvalConfig(
            default_mode="all",
            minority_col=meta["label_col"],
            minority_val=meta["minority_val"],
        )

        for j in range(5):
            old = evaluate_file(
                legacy_files[j], train, test, meta, align, config
            )
            new = evaluate_file(
                fixed_files[j], train, test, meta, align, config
            )

            for ref in ["train", "test", "both"]:
                metrics = sorted(set(old[ref]) | set(new[ref]))

                for metric in metrics:
                    a = to_float(old[ref].get(metric))
                    b = to_float(new[ref].get(metric))

                    rows.append({
                        "dataset": ds,
                        "fold": fold,
                        "j": j,
                        "ref": ref,
                        "metric": metric,
                        "legacy": a,
                        "indexfix": b,
                        "delta": b - a,
                    })

                # C2ST 距离 0.5 越近越好，额外保存便于直接比较的指标
                if "c2st_auc" in old[ref] and "c2st_auc" in new[ref]:
                    a = abs(to_float(old[ref]["c2st_auc"]) - 0.5)
                    b = abs(to_float(new[ref]["c2st_auc"]) - 0.5)
                    rows.append({
                        "dataset": ds, "fold": fold, "j": j,
                        "ref": ref, "metric": "c2st_gap",
                        "legacy": a, "indexfix": b, "delta": b - a,
                    })

            # DCR train/test 比值也按相同 j 配对
            a = to_float(old["both"].get("dcr_train_mean")) / to_float(
                old["both"].get("dcr_test_mean")
            )
            b = to_float(new["both"].get("dcr_train_mean")) / to_float(
                new["both"].get("dcr_test_mean")
            )
            rows.append({
                "dataset": ds, "fold": fold, "j": j,
                "ref": "both", "metric": "dcr_ratio_train_test",
                "legacy": a, "indexfix": b, "delta": b - a,
            })

            print(f"  j={j}: 已完成配对评估")

    detail = pd.DataFrame(rows)

    # 对同一 dataset/fold/ref/metric 下的五个配对 delta 求均值和 SD
    summary = (
        detail.groupby(
            ["dataset", "fold", "ref", "metric"], as_index=False
        )
        .agg(
            n=("delta", "count"),
            legacy_mean=("legacy", "mean"),
            indexfix_mean=("indexfix", "mean"),
            delta_mean=("delta", "mean"),
            delta_sd=("delta", "std"),
        )
    )

    # 保存明细与汇总
    out_dir = ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "indexfix_paired_samples.csv"
    summary_path = out_dir / "indexfix_paired_summary.csv"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    # 终端主要展示 test 指标和 DCR，避免刷屏
    display_metrics = [
        "tvd_avg",
        "corr_diff_fro",
        "corr_diff_nn",
        "corr_diff_nc",
        "corr_diff_cc",
        "c2st_gap",
        "alpha_precision",
        "beta_recall",
        "dcr_ratio_train_test",
    ]

    show = summary[
        summary["metric"].isin(display_metrics)
        & summary["ref"].isin(["test", "both"])
    ].copy()

    print("\n===== 五次配对采样：delta 均值 ± 标准差 =====")
    print("delta = indexfix - legacy")
    print(show[
        ["dataset", "fold", "ref", "metric", "n",
         "delta_mean", "delta_sd"]
    ].round(5).to_string(index=False))

    print(f"\n逐次配对明细：{detail_path}")
    print(f"五次采样汇总：{summary_path}")


if __name__ == "__main__":
    main()
