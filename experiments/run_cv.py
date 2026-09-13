# 文件位置: experiments/run_cv.py
"""
用法：
    python run_cv.py                    # 跑 DATASET_NAME 的全部 N_FOLDS 折
    python run_cv.py --fold 2           # 只跑第 2 折（0-based），调参时先测一折，快
    python run_cv.py --dataset ckd      # 换数据集，不用改代码

改超参数只需要改下面【常改】那一块，改完直接跑，不用传一堆命令行参数。
--dataset / --fold 是留给"批量跑几个数据集/几折"这种脚本化场景的，平时手动跑可以不传。
"""
import sys
sys.path.append("..")
from experiments.environment import *
init_environment()

import os
os.environ['PYTHONIOENCODING'] = 'gbk:replace'

import gc, json, random, argparse
import numpy as np
import utils.common as common
from experiments.data_prep import load_dataset_idx
from experiments.run_experiment import run_experiment
from core.denoise_model_v1 import TabMLPDenoise as TabMLPDenoiseV1
from core.denoise_model_v2 import TabMLPDenoise as TabMLPDenoiseV2


# ============================ 常改的参数都在这一块 ============================
DATASET_NAME = common.DATASETS[1]
SEED = common.SEEDS[4]
N_FOLDS = 5
DENOISE_CLASS = TabMLPDenoiseV2          # TabMLPDenoiseV1 / TabMLPDenoiseV2

DATA_OVERRIDES = dict(
    num_transform_method="quantile",
    cat_encoding_method="ordinal",
)
MODEL_OVERRIDES = dict(
    dim_hidden=512,
    num_timesteps=200,
    prediction_type="eps",
    focal_mode="off",              # "off" | "fixed" | "adaptive"
    use_marginal_align=False,
    logit_adj_tau=0,
    gamma_base=1,
    focal_alpha_scale=1.1,
    gma_loss_weight=0.00,
    corr_weight=0.05,
    corr_t_ratio=0.5,
    posterior_switch_ratio=1.0,
    cat_temperature_cap=1,
    x0_clamp_min=-4,
    x0_clamp_max=4,
)
TRAINING_OVERRIDES = dict(
    task="full_data",              # "full_data" | "minority_uncond" | "minority_cond"
    objective="fidelity",
    batch_size=64,
    total_steps=20000,             # 不用验证集，靠固定步数 + weight decay/dropout 控过拟合
    lr=1e-3,
    uncond_p=0.1,
)
SAMPLING_OVERRIDES = dict(
    temperature_fidelity=1,
    num_samples=None,
)
# ==============================================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config():
    base_cfg = make_default_experiment_config()
    return base_cfg.with_overrides(
        data=DATA_OVERRIDES,
        pipeline={"random_seed": SEED},
        model=MODEL_OVERRIDES,
        training=TRAINING_OVERRIDES,
        sampling=SAMPLING_OVERRIDES,
    )


def load_data(dataset_name: str):
    with open("data/dataset_info.json", "r", encoding="utf-8") as f:
        dataset_info = json.load(f)

    df_all = pd.read_csv(f"./data/{dataset_name}.csv")
    target_col = dataset_info[dataset_name]["target"]
    if target_col not in df_all.columns:
        df_all[target_col] = 1

    idx_info = load_dataset_idx("data/datasets_idx.json", dataset_name)
    return df_all, target_col, idx_info["num_col_idx"], idx_info["cat_col_idx"]


def run_one_fold(dataset_name, fold_idx, df_all, target_col, num_col_idx, cat_col_idx, exp_cfg):
    set_seed(42)  # 每折训练前都校正一次，保证折与折之间只有数据不同，其余可复现
    print(f"\n========== {dataset_name} | fold {fold_idx + 1}/{N_FOLDS} ==========")
    results = run_experiment(
        df_all=df_all,
        target_col=target_col,
        num_col_idx=num_col_idx,
        cat_col_idx=cat_col_idx,
        dataset_name=dataset_name,
        denoise_class=DENOISE_CLASS,
        exp_cfg=exp_cfg,
        fold_idx=fold_idx,
        n_folds=N_FOLDS,
    )
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME, help="不传则用上面 DATASET_NAME")
    parser.add_argument("--fold", type=int, default=None, help="只跑指定一折(0-based)；不传则跑全部 N_FOLDS 折")
    args = parser.parse_args()

    set_seed(42)
    df_all, target_col, num_col_idx, cat_col_idx = load_data(args.dataset)
    exp_cfg = build_config()

    folds = [args.fold] if args.fold is not None else range(N_FOLDS)
    all_results = []
    for fold_idx in folds:
        all_results.append(
            run_one_fold(args.dataset, fold_idx, df_all, target_col, num_col_idx, cat_col_idx, exp_cfg)
        )
    return all_results


if __name__ == "__main__":
    main()