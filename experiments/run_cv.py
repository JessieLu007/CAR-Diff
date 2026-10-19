# 文件位置: experiments/run_cv.py
import os, sys, platform
sys.path.append("..")

_IS_WINDOWS = sys.platform.startswith("win")
_IS_LINUX   = sys.platform.startswith("linux")
_IS_MAC     = sys.platform.startswith("darwin")

# Windows 上 Inductor 有路径转义 bug，默认禁用 compile
if _IS_WINDOWS:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    print("🖥️  Windows 检测到，已禁用 torch.compile（规避 Inductor 路径 bug）")
else:
    print(f"🖥️  {platform.system()} 检测到，torch.compile 保持启用")

from experiments.environment import *
init_environment()

os.environ['PYTHONIOENCODING'] = 'gbk:replace'

import gc, json, random, argparse
import numpy as np
import utils.common as common
from experiments.data_prep import load_dataset_idx
from experiments.run_experiment import run_experiment
from core.denoise_model_v1 import TabMLPDenoise as TabMLPDenoiseV1
from core.denoise_model_v2 import TabMLPDenoise as TabMLPDenoiseV2


# ============================ 常改的参数都在这一块 ============================
DATASET_NAME = common.DATASETS[0]
SEED = common.SEEDS[0]
N_FOLDS = 5
DENOISE_CLASS = TabMLPDenoiseV2          # TabMLPDenoiseV1 / TabMLPDenoiseV2

DATA_OVERRIDES = dict(
    num_transform_method="quantile",
    cat_encoding_method="ordinal",
)
MODEL_OVERRIDES = dict(
    dim_hidden=256,
    num_timesteps=200,
    prediction_type="eps",
    corr_t_ratio=0.5,  
    use_num_from_cat=False,
    use_cat_from_num=False,
    use_cat_self_attn=False,
    use_cat_tokens=False,
)
TRAINING_OVERRIDES = dict(
    batch_size=64,
    target_epochs=200,             # 不用验证集，靠固定步数 + weight decay/dropout 控过拟合
    enable_compile=True,
    lr=1e-3,
    step_floor=0,
    use_early_stop=False,
)
SAMPLING_OVERRIDES = dict(
    num_samples=None,
)
# ==============================================================================

def parse_kv(s: str) -> dict:
    import ast
    out = {}
    for item in filter(None, (s or "").split(",")):
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_config(tag: str = "", step_floor=None, model_kv=None, train_kv=None):
    from configs.experiment_configs import TrainingConfig
    import dataclasses
    print("TrainingConfig file:", TrainingConfig.__module__)
    base_cfg = make_default_experiment_config()
    training = {**TRAINING_OVERRIDES, **(train_kv or {})}
    if step_floor is not None:
        training["step_floor"] = step_floor
    return base_cfg.with_overrides(
        data=DATA_OVERRIDES,
        pipeline={"random_seed": SEED, "tag": tag},
        model={**MODEL_OVERRIDES, **(model_kv or {})},
        training=training,
        sampling=SAMPLING_OVERRIDES,
    )


def load_data(dataset_name: str):
    with open("data/datasets_info.json", "r", encoding="utf-8") as f:
        datasets_info = json.load(f)

    #df_all = pd.read_csv(f"./data/{dataset_name}.csv")

    df_all = pd.read_csv(f"./data/{dataset_name}.csv", na_values=["?", "", " ", "NA", "nan", "None"],
                         keep_default_na=True)
    # 做额外空格处理，防止后续操作出错
    df_all = df_all.apply(
        lambda s: s.str.replace(r'^[\s\u00A0\u3000]+|[\s\u00A0\u3000]+$', '', regex=True) 
        if s.dtype == object else s
    )

    target_col = datasets_info[dataset_name]["target"]
    minority_class = datasets_info[dataset_name].get("minority_class")
    if target_col not in df_all.columns:
        df_all[target_col] = 1

    idx_info = load_dataset_idx("data/datasets_idx.json", dataset_name)
    return df_all, target_col, idx_info["num_col_idx"], idx_info["cat_col_idx"], minority_class


def run_one_fold(dataset_name, fold_idx, df_all, target_col, num_col_idx, cat_col_idx, exp_cfg, minority_class, seed):
    set_seed(seed)  # 每折训练前都校正一次，保证折与折之间只有数据不同，其余可复现
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
        target_minority_class=minority_class,
        only_splits=False,
    )
    gc.collect()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", default=None, help="数据集名称或索引，不传则用上面 DATASET_NAME")
    parser.add_argument("--fold", type=int, nargs='*', default=None, help="0-based，可传多个：--fold 0 1 2；不传则 0..n_folds-1")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="设置环境变量随机种子")
    parser.add_argument("--all", action="store_true", 
                        help="依次跑 common.DATASETS 中所有数据集的 N_FOLDS 折划分；传了此参数时 --dataset 被忽略")
    parser.add_argument("--step_floor", type=int, default=None, help="步数下限 S_min")
    parser.add_argument("--tag", default="", help="运行标签（仅字母数字下划线），写入结果文件夹名")
    parser.add_argument("--model", default="", help='覆盖 MODEL_OVERRIDES')
    parser.add_argument("--train", default="", help='覆盖 TRAINING_OVERRIDES，如 "target_epochs=200"')
    args = parser.parse_args()

    if args.dataset is None:
        dataset_names = list(common.DATASETS)
    else:
        dataset_names = [common.resolve_dataset(d) for d in args.dataset]
    print(f"▶ datasets = {dataset_names}")

    set_seed(args.seed)

    # if args.all:
    #     dataset_names = list(common.DATASETS)
    #     print(f"▶ --all 模式，共 {len(dataset_names)} 个数据集: {dataset_names}")
    # else:
    #     dataset_names = [common.resolve_dataset(args.dataset)]
    #     print(f"▶ dataset = {dataset_names[0]} (来自 --dataset {args.dataset})")

    all_results = {}

    for dataset_name in dataset_names:
        df_all, target_col, num_col_idx, cat_col_idx, minority_class = load_data(dataset_name)
        exp_cfg = build_config(args.tag, args.step_floor, parse_kv(args.model), parse_kv(args.train))

        folds = args.fold if args.fold else list(range(args.n_folds))
        dataset_results = []
        for fold_idx in folds:
            dataset_results.append(
                run_one_fold(dataset_name, fold_idx, df_all, target_col, num_col_idx, 
                         cat_col_idx, exp_cfg, minority_class, args.seed)
            )
        all_results[dataset_name] = dataset_results
        
    return all_results


if __name__ == "__main__":
    main()