# 用法: python prepare_tabsyn_folds.py --dataset ckd --seed 42 --n_folds 5
"""
把 ddpm_med 里已经存好的 5 折 train/test 接到 tabsyn 的自定义数据集机制上。

关键点：tabsyn 的 process_dataset.py 里，如果 info.json 的 "test_path" 不为空，
它会把 "data_path" 整份当训练集，直接用 "test_path" 当测试集，
【之前那个"循环重试凑齐所有类别值"的逻辑根本不会被触发】——
正好对应我们已经做好的、不做类别覆盖强制的 5 折划分，不需要改 tabsyn 任何代码。

把 5 折当成 5 个不同的数据集名字（{dataset}_fold0 ... {dataset}_fold4），
之后 tabsyn 该怎么跑就怎么跑，结果自然分开存，不会互相覆盖：
    python process_dataset.py --dataname ckd_fold0
    python main.py --dataname ckd_fold0 --method vae --mode train
    python main.py --dataname ckd_fold0 --method tabsyn --mode train
    python main.py --dataname ckd_fold0 --method tabsyn --mode sample --save_path synthetic/ckd_fold0/tabsyn.csv
"""
from pathlib import Path
import os, sys, json, shutil, argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import common
from utils.common import resolve_dataset

# ============================ 常改的参数 ============================
DEFAULT_DATASET = common.DATASETS[0]
DEFAULT_SEED = 2026
DEFAULT_N_FOLDS = 5

#TABSYN_ROOT = os.environ.get('TABSYN_ROOT', str(Path(__file__).resolve().parents[2] / 'tabsyn-main'))
TABSYN_ROOT = os.environ.get('TABSYN_ROOT', str(Path(_ROOT) / 'third_party' / 'tabsyn' / 'tabsyn-main'))
DDPM_MED_SPLIT_DIR = "splits"             # 对应 ddpm_med 里 run_experiment 的 split_dir
TASK = "full_data"                        # 对应 exp_cfg.training.task
TASK_TYPE = "binclass"                    # tabsyn 需要: "binclass" | "regression"
DATASETS_INFO_PATH = "data/datasets_info.json"
DATASETS_IDX_PATH = "data/datasets_idx.json"
# =====================================================================


def prepare_fold(dataset_name, fold_idx, seed, num_col_idx, cat_col_idx, target_col_idx):
    fold_name = f"{dataset_name}_fold{fold_idx}"

    # ---- 源文件：run_experiment 里 get_or_build_imputed_reference_cv 落盘的那份缓存 ----
    src_dir = os.path.join(_ROOT, DDPM_MED_SPLIT_DIR, 
        f"{dataset_name}_seed{seed}_{TASK}_fold{fold_idx}_imputed"
    )
    train_src = os.path.join(src_dir, "real_train_imputed.csv")
    test_src = os.path.join(src_dir, "real_test_imputed.csv")
    assert os.path.exists(train_src) and os.path.exists(test_src), (
        f"{src_dir} 下没找到插补后的 train/test，先跑一遍 run_experiment(fold_idx={fold_idx}) 生成缓存。"
    )

    # ---- 落到 tabsyn 项目的 data/{fold_name}/ 下 ----
    dst_dir = os.path.join(TABSYN_ROOT, "data", fold_name)
    os.makedirs(dst_dir, exist_ok=True)
    train_dst = os.path.join(dst_dir, f"{fold_name}.csv")
    test_dst = os.path.join(dst_dir, f"{fold_name}_test.csv")
    shutil.copy(train_src, train_dst)
    shutil.copy(test_src, test_dst)

    # ---- 写 tabsyn 要求的 info.json，放到 data/Info/ 下 ----
    info = {
        "name": fold_name,
        "task_type": TASK_TYPE,
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"data/{fold_name}/{fold_name}.csv",
        "test_path": f"data/{fold_name}/{fold_name}_test.csv",  # 关键：给了这个，tabsyn 不再自己切分
    }
    info_dir = os.path.join(TABSYN_ROOT, "data", "Info")
    os.makedirs(info_dir, exist_ok=True)
    with open(os.path.join(info_dir, f"{fold_name}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print(f"✅ {fold_name} 准备好了 -> python process_dataset.py --dataname {fold_name}")
    return fold_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help=f"数据集名称或索引（如 ckd / 0）；默认 {DEFAULT_DATASET}",)
    parser.add_argument("--all", action="store_true", help="遍历 common.DATASETS 里的数据集")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"随机种子；默认 {DEFAULT_SEED}")
    parser.add_argument("--n_folds", type=int, default=DEFAULT_N_FOLDS, help=f"折数；默认 {DEFAULT_N_FOLDS}")
    args = parser.parse_args()

    if args.all:
        dataset_names = list(common.DATASETS)
        print(f"▶ 遍历全部 {len(dataset_names)} 个数据集: {dataset_names}")
    else:
        token = args.dataset if args.dataset is not None else str(DEFAULT_DATASET)
        dataset_names = [resolve_dataset(token)]

    #dataset_name = resolve_dataset(args.dataset)
    #print(f"▶ dataset = {dataset_name}  seed = {args.seed}  n_folds = {args.n_folds}")

    with open(os.path.join(_ROOT, DATASETS_IDX_PATH), "r", encoding="utf-8") as f:
        all_idx = json.load(f)
        #idx_info = json.load(f)[dataset_name]
    
    failed = []

    for dataset_name in dataset_names:
        if dataset_name not in all_idx:
            print(f"⚠ 跳过 {dataset_name}: {DATASETS_IDX_PATH} 里没有该条目")
            failed.append(dataset_name)
            continue

        print(f"\n========== {dataset_name} | seed={args.seed} | n_folds={args.n_folds} ==========")
        idx_info       = all_idx[dataset_name]
        num_col_idx    = idx_info["num_col_idx"]
        cat_col_idx    = idx_info["cat_col_idx"]
        target_col_idx = idx_info["target_col_idx"]

        for fold_idx in range(args.n_folds):
            prepare_fold(dataset_name, fold_idx, args.seed, 
                         num_col_idx, cat_col_idx, target_col_idx)

    if failed:
        print(f"\n⚠ 有 {len(failed)} 个数据集没跑成: {failed}")
    else:
        print("\n✅ 全部完成")


if __name__ == "__main__":
    main()