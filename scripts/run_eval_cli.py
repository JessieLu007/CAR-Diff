#!/usr/bin/env python
# 文件位置: scripts/run_eval_cli.py
"""
命令行版评估对比：读取真实数据 + baseline 生成数据 + meddiff 生成数据，跑 batch_evaluate。
用法：
    python scripts/run_eval_cli.py --dataset framingham --seed 0
    python scripts/run_eval_cli.py --dataset framingham --seed 0 --meddiff-run full_data_fidelity_h512_t200_3_0906_153000
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from evaluation.fidelity_eval import EvalConfig, batch_evaluate, print_summary


def find_latest_run(result_root: Path) -> str:
    candidates = [p for p in result_root.iterdir() if p.is_dir()]
    assert candidates, f"{result_root} 下没有任何实验目录"
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def main():
    parser = argparse.ArgumentParser(description="meddiff vs baseline 保真度评估（CLI 版）")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=0, help="对应训练时 pipeline.random_seed")
    parser.add_argument("--meddiff-run", default=None,
                         help="results/{dataset}/seed_{seed}/ 下的实验目录名，不传则取最近修改的一个")
    parser.add_argument("--data-dir", default="data")
    #parser.add_argument("--split-dir", default="splitted_data")
    parser.add_argument("--baseline-dir", default="baselines/tabsyn_gen")
    parser.add_argument("--metrics", nargs="+",
                         default=["tvd", "corr_diff", "alpha_beta", "c2st", "dcr", "tstr"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    #split_dir = Path(args.split_dir) / "seed_0" / args.dataset  # 与 notebook 一致，划分固定用 seed_0
    baseline_dir = Path(args.baseline_dir) / f"seed_{args.seed}" / "full_data"
    result_root = Path("results") / args.dataset / f"seed_{args.seed}"

    with open(data_dir / "dataset_info.json", "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    num_cols = dataset_info[args.dataset]["numerical"]
    cat_cols = dataset_info[args.dataset]["categorical"]
    target_col = dataset_info[args.dataset]["target"]

    meddiff_run = args.meddiff_run or find_latest_run(result_root)
    result_dir = result_root / meddiff_run
    timestamp_str = "_".join(meddiff_run.split("_")[-2:])
    print(f"评估目录: {result_dir}")

    train = pd.read_csv(result_dir / "real_train_imputed.csv")
    test = pd.read_csv(result_dir / "real_test_imputed.csv")
    fake_tabsyn = pd.read_csv(baseline_dir / f"{args.dataset}.csv")
    fake_meddiff = pd.read_csv(result_dir / "synthetic_data.csv")

    config = EvalConfig(
        default_mode="all",
        minority_col=target_col,   # 用数据集自己的 target 列名，不写死 'label'
        minority_val=1,
        metric_modes={
            "tvd": "all", "alpha_beta": "all", "c2st": "all",
            "dcr": "all", "corr_diff": "all",
        },
    )

    output_path = Path("results") / args.dataset / f"eval_{timestamp_str}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results_df = batch_evaluate(
        models={"tabsyn": fake_tabsyn, "meddiff": fake_meddiff},
        train_df=train, test_df=test,
        num_cols=num_cols, cat_cols=cat_cols,
        config=config, output_csv=str(output_path),
        metrics=args.metrics,
    )
    print_summary(results_df)


if __name__ == "__main__":
    main()