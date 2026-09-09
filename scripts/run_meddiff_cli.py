#!/usr/bin/env python
# 文件位置: scripts/run_meddiff_cli.py
"""
命令行版训练 + 生成入口，等价于 notebook 的 [0]~[4] 单元。
用法：
    python scripts/run_meddiff_cli.py --dataset framingham --denoise v2
    python scripts/run_meddiff_cli.py --dataset pima --denoise v1 --seed 1
"""
import argparse
import gc
import json
import os
import random
import sys

os.environ['PYTHONIOENCODING'] = 'gbk:replace'

import numpy as np
import pandas as pd
import torch

sys.path.append("..")  # 如果脚本不在项目根目录运行，按实际相对路径调整
from experiments.environment import init_environment
from experiments.data_prep import load_dataset_idx
from experiments.run_experiment import run_experiment
from core.denoise_model_v1 import TabMLPDenoise as TabMLPDenoiseV1
from core.denoise_model_v2 import TabMLPDenoise as TabMLPDenoiseV2
from configs.experiment_configs import make_default_experiment_config

from scripts.exp_config_presets import build_exp_cfg


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DENOISE_MAP = {"v1": TabMLPDenoiseV1, "v2": TabMLPDenoiseV2}


def main():
    parser = argparse.ArgumentParser(description="meddiff 训练 + 生成（CLI 版）")
    parser.add_argument("--dataset", required=True, help="需在 dataset_info.json 和 datasets_idx.json 中都有对应条目")
    parser.add_argument("--denoise", choices=["v1", "v2"], default="v2")
    parser.add_argument("--seed", type=int, default=0, help="pipeline.random_seed（数据划分种子）")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    init_environment()
    set_seed(42)

    with open(os.path.join(args.data_dir, "dataset_info.json"), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    df_all = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))

    target_col = dataset_info[args.dataset]["target"]
    if target_col not in df_all.columns:
        df_all[target_col] = 1

    idx_info = load_dataset_idx(os.path.join(args.data_dir, "datasets_idx.json"), args.dataset)

    exp_cfg = build_exp_cfg(random_seed=args.seed)
    denoise_class = DENOISE_MAP[args.denoise]

    set_seed(42)  # 训练前再校正一次，和 notebook 行为一致
    results = run_experiment(
        df_all=df_all,
        target_col=target_col,
        num_col_idx=idx_info["num_col_idx"],
        cat_col_idx=idx_info["cat_col_idx"],
        dataset_name=args.dataset,
        denoise_class=denoise_class,
        exp_cfg=exp_cfg,
    )

    print(f"✅ 完成，结果目录: {results['save_path']}")
    gc.collect()


if __name__ == "__main__":
    main()