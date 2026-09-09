# experiments/data_prep.py
"""
数据预处理层：数据集划分、task 路由、TabularDataProcessor 构建。
只处理 DataFrame 和最基础的 tensor（标签），不涉及模型。
"""
import os
import json
import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from core.data_processor import TabularDataProcessor

#  数据集划分
def split_dataset(
    dataset_name: str,   # 用于生成划分文件名
    df_all: pd.DataFrame,
    target_col: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_seed: int = 0,
    split_dir: str = "splits", # 划分索引存放目录
    ) -> dict:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "训练、验证和测试集比例之和必须为1"

    split_path = os.path.join(split_dir, f"{dataset_name}_seed{random_seed}.json")

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            split_idx = json.load(f)
        saved_ratios = (split_idx.get("train_ratio"), split_idx.get("val_ratio"), split_idx.get("test_ratio"))
        if saved_ratios != (train_ratio, val_ratio, test_ratio):
            raise ValueError(
                f"{split_path} 已存在， 但划分比例和本次请求不一致"
                f"(存档： {saved_ratios}, 本次： {(train_ratio, val_ratio, test_ratio)})。"
                f"如果是有意更换比例, 请删除该文件或换一个 split_dir。"
            )
    else:
        unique_classes = df_all[target_col].nunique()
        stratify_col = df_all[target_col] if unique_classes > 1 else None

        idx_temp, idx_test = train_test_split(
            df_all.index.to_numpy(),test_size=test_ratio,
            stratify=stratify_col, random_state=random_seed,
        )

        stratify_temp = df_all.loc[idx_temp, target_col] if unique_classes > 1 else None
        val_ratio_in_temp = val_ratio / (train_ratio + val_ratio)
        idx_train, idx_val = train_test_split(
            idx_temp, test_size=val_ratio_in_temp,
            stratify=stratify_temp, random_state=random_seed + 42,
        )
        split_idx = {
            "train_idx": idx_train.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_test.tolist(),
            "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio,
        }
        os.makedirs(split_dir, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump(split_idx, f)

    df_train = df_all.loc[split_idx["train_idx"]].reset_index(drop=True)
    df_val   = df_all.loc[split_idx["val_idx"]].reset_index(drop=True)
    df_test  = df_all.loc[split_idx["test_idx"]].reset_index(drop=True)

    return df_train, df_val, df_test


def get_or_build_imputed_reference(
    processor, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
    dataset_name: str, random_seed: int, task: str, split_dir: str = "splits",
):
    """
    生成/复用一份“缺失值已补全”的参考数据（train/val/test), 存放在稳定路径下，供本项目内部评估、以及外部
    代码库（如tabsyn）共同读取，保证双方看到的“真实数据”完全一致。

    同一 (dataset, seed, task) 第一次调用才真正插补并落盘，后续复用磁盘上
    已有的文件——多次 run_experiment（比如换 denoise_class 对比）不会重复插补。
    """

    ref_dir = os.path.join(split_dir, f"{dataset_name}_seed{random_seed}_{task}_imputed")
    paths = {name: os.path.join(ref_dir, f"real_{name}_imputed.csv") for name in ("train", "val", "test")}

    if all(os.path.exists(p) for p in paths.values()):
        return (pd.read_csv(paths["train"]), pd.read_csv(paths["val"]), pd.read_csv(paths["test"]))

    os.makedirs(ref_dir, exist_ok=True)
    df_train_imputed = processor.transform_readable(df_train)
    df_val_imputed   = processor.transform_readable(df_val)
    df_test_imputed  = processor.transform_readable(df_test)

    df_train_imputed.to_csv(paths["train"], index=False)
    df_val_imputed.to_csv(paths["val"], index=False)
    df_test_imputed.to_csv(paths["test"], index=False)

    with open(os.path.join(ref_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": dataset_name, "random_seed": random_seed, "task": task,
            "n_train": len(df_train_imputed), "n_val": len(df_val_imputed), "n_test": len(df_test_imputed),
        }, f, indent=4, ensure_ascii=False)

    return df_train_imputed, df_val_imputed, df_test_imputed


# 2. Task 路由
def route_task_data(df: pd.DataFrame, target_col: str, task: str, null_class_idx: int):
    """
    full_data       —— target_col 作为普通类别特征保留，y 全为 null
    minority_uncond —— 仅少数类（label==1），target_col 剥离，y 全为 null
    minority_cond   —— 全量数据，target_col 剥离为条件标签
    """
    if task == "full_data":
        df_features = df.copy()
        y = torch.full((len(df),), null_class_idx, dtype=torch.long)

    elif task == "minority_uncond":
        df_minority = df[df[target_col] == 1].copy()
        df_features = df_minority.drop(columns=[target_col])
        y = torch.full((len(df_features),), null_class_idx, dtype=torch.long)

    elif task == "minority_cond":
        y = torch.tensor(df[target_col].values, dtype=torch.long)
        df_features = df.drop(columns=[target_col])

    else:
        raise ValueError(f"未知 task: {task}")

    return df_features.reset_index(drop=True), y


# =========================================================
# 目标类分布统计（对应原 MedSynth._compute_target_dist）
# =========================================================
def compute_target_dist(df: pd.DataFrame, target_col: str, target_class: int = 1) -> dict:
    counts = df[target_col].value_counts()
    total = len(df)
    target_count = int(counts.get(target_class, 0))
    background_count = int(total - target_count)
    ir = background_count / target_count if target_count > 0 else float('inf')

    return {
        "target_class": target_class,
        "target_count": target_count,
        "background_count": background_count,
        "total": total,
        "IR": ir,
    }


# =========================================================
# 构建并拟合 TabularDataProcessor（对应原 MedSynth.fit 中的预处理器构建段）
# =========================================================
import json

def load_dataset_idx(json_path: str, dataset_name: str) -> dict:
    """
    假设 datasets_idx.json 结构为 {dataset_name: {"num_col_idx": [...], "cat_col_idx": [...], "target_col_idx": [...]}}。
    如果你本地实际结构不是这样，告诉我，改这一个函数就行，不影响其他部分。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        all_idx = json.load(f)
    assert dataset_name in all_idx, f"{dataset_name} 不在 {json_path} 中"
    return all_idx[dataset_name]


def resolve_col_idx_after_routing(raw_idx: list, original_columns: list, routed_columns: list) -> list:
    """
    把相对于原始数据集的位置索引，转换成相对于 task 路由后 DataFrame 的位置索引。
    路由会 drop 掉 target_col，导致后面列的位置整体前移，因此按列名重新定位，而不是直接平移下标。
    """
    names = [original_columns[i] for i in raw_idx if original_columns[i] in routed_columns]
    return [routed_columns.index(n) for n in names]


def build_processor(
    df_features: pd.DataFrame,
    original_columns: list,
    num_col_idx: list,
    cat_col_idx: list,
    target_col: str,
    task: str,
    num_transform_method: str,
    cat_encoding_method: str,
    random_state: int,
) -> TabularDataProcessor:
    routed_cols = df_features.columns.tolist()

    resolved_num_idx = resolve_col_idx_after_routing(num_col_idx, original_columns, routed_cols)
    resolved_cat_idx = resolve_col_idx_after_routing(cat_col_idx, original_columns, routed_cols)

    # full_data 模式下 target_col 仍作为普通类别特征保留，datasets_idx.json 里的
    # cat_col_idx 通常不包含 target（它有独立的 target_col_idx），这里显式补上
    if task == "full_data" and target_col in routed_cols:
        t_pos = routed_cols.index(target_col)
        if t_pos not in resolved_cat_idx:
            resolved_cat_idx = sorted(resolved_cat_idx + [t_pos])

    processor = TabularDataProcessor(
        num_col_idx=resolved_num_idx,
        cat_col_idx=resolved_cat_idx,
        num_transform_method=num_transform_method,
        cat_encoding_method=cat_encoding_method,
        random_state=random_state,
    )
    processor.fit(df_features)
    return processor