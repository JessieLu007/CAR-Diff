#!/usr/bin/env python
# 文件位置: scripts/run_sweep_cli.py
"""
多随机种子批量训练：同一个数据集用不同 seed 重复跑一遍，用于严谨性所需的多种子汇报。
用法：
    python scripts/run_sweep_cli.py --dataset framingham --denoise v2 --seeds 0 1 2
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

sys.path.append("..")
from experiments.environment import init_environment
from experiments.data_prep import load_dataset_idx
from experiments.run_experiment import run_experiment
from core.denoise_model_v1 import TabMLPDenoise as TabMLPDenoiseV1
from core.denoise_model_v2 import TabMLPDenoise as TabMLPDenoiseV2
from scripts.exp_config_presets import build_exp_cfg


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


DENOISE_MAP = {"v1": TabMLPDenoiseV1, "v2": TabMLPDenoiseV2}


def main():
    parser = argparse.ArgumentParser(description="多种子批量训练")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--denoise", choices=["v1", "v2"], default="v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    init_environment()

    with open(os.path.join(args.data_dir, "dataset_info.json"), "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    df_all = pd.read_csv(os.path.join(args.data_dir, f"{args.dataset}.csv"))
    target_col = dataset_info[args.dataset]["target"]
    if target_col not in df_all.columns:
        df_all[target_col] = 1

    idx_info = load_dataset_idx(os.path.join(args.data_dir, "datasets_idx.json"), args.dataset)
    denoise_class = DENOISE_MAP[args.denoise]

    for seed in args.seeds:
        print(f"\n{'='*20} seed={seed} {'='*20}")
        exp_cfg = build_exp_cfg(random_seed=seed)
        set_seed(seed)   # 划分种子和训练种子绑在一起，对应上一轮的讨论
        results = run_experiment(
            df_all=df_all, target_col=target_col,
            num_col_idx=idx_info["num_col_idx"], cat_col_idx=idx_info["cat_col_idx"],
            dataset_name=args.dataset, denoise_class=denoise_class, exp_cfg=exp_cfg,
        )
        print(f"✅ seed={seed} 完成: {results['save_path']}")
        gc.collect()


if __name__ == "__main__":
    main()