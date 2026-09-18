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
import os, json, shutil, argparse

# ============================ 常改的参数 ============================
TABSYN_ROOT = "E:/projects/tabsyn-main"           # 改成本地 tabsyn 仓库路径
DDPM_MED_SPLIT_DIR = "splits"             # 对应 ddpm_med 里 run_experiment 的 split_dir
TASK = "full_data"                        # 对应 exp_cfg.training.task
TASK_TYPE = "binclass"                    # tabsyn 需要: "binclass" | "regression"
DATASET_INFO_PATH = "data/dataset_info.json"
DATASETS_IDX_PATH = "data/datasets_idx.json"
# =====================================================================


def prepare_fold(dataset_name, fold_idx, seed, num_col_idx, cat_col_idx, target_col_idx):
    fold_name = f"{dataset_name}_fold{fold_idx}"

    # ---- 源文件：run_experiment 里 get_or_build_imputed_reference_cv 落盘的那份缓存 ----
    src_dir = os.path.join(
        DDPM_MED_SPLIT_DIR, f"{dataset_name}_seed{seed}_{TASK}_fold{fold_idx}_imputed"
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
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    with open(DATASETS_IDX_PATH, "r", encoding="utf-8") as f:
        idx_info = json.load(f)[args.dataset]
    num_col_idx = idx_info["num_col_idx"]
    cat_col_idx = idx_info["cat_col_idx"]
    target_col_idx = idx_info["target_col_idx"]  # datasets_idx.json 里本来就有这个字段

    for fold_idx in range(args.n_folds):
        prepare_fold(args.dataset, fold_idx, args.seed, num_col_idx, cat_col_idx, target_col_idx)


if __name__ == "__main__":
    main()