# experiments/data_prep.py
"""
数据预处理层：数据集划分、task 路由、TabularDataProcessor 构建。
只处理 DataFrame 和最基础的 tensor（标签），不涉及模型。
"""
import os
import json
from pathlib import Path
import torch
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

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
    split_dir: str = "splits",
    ) -> dict:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "训练、验证和测试集比例之和必须为1"

    #ref_dir = Path(split_dir) / f"{dataset_name}_seed{random_seed}"
    #train_path = ref_dir / "real_"

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

from typing import Hashable

def encode_target(
    df:pd.DataFrame, target_col: str, class_order: list | None=None,
) -> tuple[pd.DataFrame, dict]:
    """
    把target_col映射成0,..K-1的整数，返回（新df, 映射表）
    class_order给定按其顺序编码（便于跨fold保持一致），否则按sorted(unique)稳定排序
    """
    df = df.copy()
    if class_order is None:
        class_order = sorted(df[target_col].dropna().unique().tolist())

    mapping = {cls: i for i, cls in enumerate(class_order)}
    unknown = set(df[target_col].dropna().unique()) - set(mapping)
    if unknown:
        raise ValueError(f"{target_col} 出现未在 class_order 中的取值: {unknown}")

    df[target_col] = df[target_col].map(mapping).astype("int64")
    return df, {"class_order": class_order, "mapping": mapping}


def decode_target(y_int, mapping: dict):
    """0..K-1 → 原始类名（生成样本后还原用）"""
    inv = {v: k for k, v in mapping.items()}
    if hasattr(y_int, "tolist"):
        return [inv[int(v)] for v in y_int.tolist()]
    return inv[int(y_int)]


def resolve_minority_class(
    df: pd.DataFrame, target_col: str,
    target_minority_class: Hashable | None = None,
    verbose: bool = True,
) -> Hashable:
        
    counts = df[target_col].value_counts()
    if target_minority_class is not None:
        if target_minority_class not in counts.index:
            str_matches = [c for c in counts.index if str(c) == str(target_minority_class)]
            if len(str_matches) == 1:
                target_minority_class = str_matches[0]
            else:
                raise ValueError(
                    f"指定的少数类{target_minority_class!r}不在 {target_col}的取值里： "
                    f"{counts.index.tolist()}"
                    )
        chosen = target_minority_class
    else:
        chosen = counts.idxmin()
    if verbose:
        print(f"🎯 少数类 = {chosen!r} | 分布 = {counts.to_dict()}")
        if (counts == counts.min()).sum() > 1:
            print("⚠️ 并列最少类，已取 idxmin；建议显式指定")
    return chosen


# =========================================================
# 5折交叉验证划分（不含验证集）—— 新增，和 split_dataset 是并列的两条路，互不影响
# =========================================================
def make_kfold_splits_old(
    dataset_name: str,
    df_all: pd.DataFrame,
    target_col: str,
    n_folds: int = 5,
    random_seed: int = 0,
    split_dir: str = "splits",
) -> list:
    """
    生成/复用 n_folds 折的 (train_idx, test_idx)。没有验证集：不需要验证集就不切，
    过拟合缓解交给权重衰减/dropout等训练期正则化处理。
    和 split_dataset 一样做磁盘缓存，第一次调用才真正划分，之后复用同一份索引文件，
    保证同一 dataset/seed/n_folds 下每次跑到的折完全一致。
    """
    split_path = os.path.join(split_dir, f"{dataset_name}_seed{random_seed}_{n_folds}fold.json")

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            folds_idx = json.load(f)
        if len(folds_idx) != n_folds:
            raise ValueError(
                f"{split_path} 已存在，但折数({len(folds_idx)})和本次请求({n_folds})不一致。"
                f"如果是有意更换折数，请删除该文件或换一个 split_dir。"
            )
        return folds_idx

    unique_classes = df_all[target_col].nunique()
    if unique_classes > 1:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df_all, df_all[target_col])
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        split_iter = splitter.split(df_all)

    folds_idx = [
        {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}
        for train_idx, test_idx in split_iter
    ]

    os.makedirs(split_dir, exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(folds_idx, f)

    return folds_idx


# =========================================================
# 5折交叉验证划分（不含验证集）
# =========================================================

def _build_kfold_candidate(
    df_all: pd.DataFrame,
    target_col: str,
    n_folds: int,
    random_seed: int,
) -> list:
    """按指定 seed 生成一份普通 K-fold / StratifiedKFold 划分。"""
    unique_classes = df_all[target_col].nunique()

    if unique_classes > 1:
        splitter = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        split_iter = splitter.split(
            df_all,
            df_all[target_col],
        )
    else:
        splitter = KFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_seed,
        )
        split_iter = splitter.split(df_all)

    return [
        {
            "train_idx": train_idx.tolist(),
            "test_idx": test_idx.tolist(),
        }
        for train_idx, test_idx in split_iter
    ]


def _collect_test_only_categories(
    df_all: pd.DataFrame,
    folds_idx: list,
    cat_col_idx: list[int],
) -> dict:
    """
    收集所有在至少一个 fold 中只出现在 test、未出现在对应 train
    的 categorical level。

    返回：
        {
            column_name: {problematic_value1, problematic_value2, ...}
        }
    """
    problems = {}

    cat_cols = [
        df_all.columns[i]
        for i in cat_col_idx
        if 0 <= i < len(df_all.columns)
    ]

    for fold in folds_idx:
        train_idx = fold["train_idx"]
        test_idx = fold["test_idx"]

        for col in cat_cols:
            train_values = set(
                df_all.loc[train_idx, col]
                .dropna()
                .unique()
            )
            test_values = set(
                df_all.loc[test_idx, col]
                .dropna()
                .unique()
            )

            unseen = test_values - train_values

            if unseen:
                problems.setdefault(col, set()).update(unseen)

    return problems


def _check_categorical_coverage(
    df_all: pd.DataFrame,
    folds_idx: list,
    cat_col_idx: list[int],
) -> tuple[bool, str]:
    """
    检查每个 fold 的 test categorical levels 是否全部出现在对应 train 中。
    """
    problems = _collect_test_only_categories(
        df_all,
        folds_idx,
        cat_col_idx,
    )

    if not problems:
        return True, ""

    # 只返回第一个问题，便于日志阅读
    col = next(iter(problems))
    values = sorted(problems[col], key=str)

    # 找出具体是哪个 fold 触发了问题
    for fold_idx, fold in enumerate(folds_idx):
        train_values = set(
            df_all.loc[fold["train_idx"], col]
            .dropna()
            .unique()
        )
        test_values = set(
            df_all.loc[fold["test_idx"], col]
            .dropna()
            .unique()
        )
        unseen = test_values - train_values

        if unseen:
            return False, (
                f"fold={fold_idx}, column={col!r}, "
                f"test-only categories={sorted(unseen, key=str)}"
            )

    return False, (
        f"column={col!r}, "
        f"test-only categories={values}"
    )


def _json_safe_value(value):
    """把 numpy scalar 等转换成可直接 json 序列化的 Python scalar。"""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _apply_category_changes(
    df_all: pd.DataFrame,
    category_changes: dict,
) -> pd.DataFrame:
    """
    根据缓存中的 category_changes 重放 fallback。
    """
    df_processed = df_all.copy()

    for col, changes in category_changes.items():
        if col not in df_processed.columns:
            raise ValueError(
                f"缓存中的 categorical column {col!r} "
                f"不在当前数据中。"
            )

        for change in changes:
            source = change["from"]
            target = change["to"]

            df_processed.loc[
                df_processed[col] == source,
                col
            ] = target

    return df_processed


def _collapse_singleton_categories(
    df_all: pd.DataFrame,
    cat_col_idx: list[int],
) -> tuple[pd.DataFrame, dict]:
    """
    第一层 fallback：

    对整个数据集只出现 1 次的 categorical level，
    合并到该列最常见的非 singleton level。

    只处理 singleton，不改变其他类别。
    """
    df_processed = df_all.copy()
    category_changes = {}

    for col_idx in cat_col_idx:
        if not (0 <= col_idx < len(df_processed.columns)):
            continue

        col = df_processed.columns[col_idx]

        counts = df_processed[col].value_counts(dropna=True)

        singleton_values = (
            counts[counts == 1]
            .index
            .tolist()
        )

        if not singleton_values:
            continue

        candidate_counts = counts[
            ~counts.index.isin(singleton_values)
        ]

        if candidate_counts.empty:
            raise RuntimeError(
                f"categorical column {col!r} 的所有非缺失 "
                f"level 都只出现 1 次，无法进行 singleton fallback。"
            )

        fallback_value = candidate_counts.idxmax()

        col_changes = []

        for value in singleton_values:
            if value == fallback_value:
                continue

            df_processed.loc[
                df_processed[col] == value,
                col
            ] = fallback_value

            col_changes.append({
                "from": _json_safe_value(value),
                "to": _json_safe_value(fallback_value),
            })

        category_changes[col] = col_changes

        print(
            f"⚠️ categorical column {col!r}: "
            f"merged {len(col_changes)} singleton levels "
            f"into most frequent level {fallback_value!r}"
        )

    return df_processed, category_changes


def _pool_test_only_categories(
    df_all: pd.DataFrame,
    folds_idx: list,
    cat_col_idx: list[int],
) -> tuple[pd.DataFrame, dict]:
    """
    第二层 fallback：

    对当前 candidate split 中实际出现 test-only 的 categorical levels，
    只把这些有问题的 level 合并到该列一个不会造成 coverage 问题的 level。

    这里不重新划分 fold，因此：
    - train/test 样本数不变
    - fold 索引不变
    - target 的分层关系不变
    - 只改变确实造成 encoder vocabulary 不一致的类别
    """
    df_processed = df_all.copy()

    problems = _collect_test_only_categories(
        df_processed,
        folds_idx,
        cat_col_idx,
    )

    if not problems:
        return df_processed, {}

    category_changes = {}

    for col, problematic_values in problems.items():
        counts = df_processed[col].value_counts(dropna=True)

        # 优先选择一个本身没有出现 test-only 问题的 level
        candidate_counts = counts[
            ~counts.index.isin(problematic_values)
        ]

        if not candidate_counts.empty:
            fallback_value = candidate_counts.idxmax()
        else:
            # 极端情况：该列所有非缺失 level 都在某个 fold 中成为 test-only。
            # 此时直接保留该列全局最常见 level，其他 level 全部并入。
            fallback_value = counts.idxmax()

        col_changes = []

        for value in problematic_values:
            if value == fallback_value:
                continue

            df_processed.loc[
                df_processed[col] == value,
                col
            ] = fallback_value

            col_changes.append({
                "from": _json_safe_value(value),
                "to": _json_safe_value(fallback_value),
            })

        if col_changes:
            category_changes[col] = col_changes

            print(
                f"⚠️ categorical column {col!r}: "
                f"merged {len(col_changes)} test-only levels "
                f"into {fallback_value!r}"
            )

    return df_processed, category_changes


def _save_kfold_cache(
    split_path: str,
    folds_idx: list,
    n_folds: int,
    random_seed: int,
    category_changes: dict | None = None,
):
    """保存新版 split cache。"""
    payload = {
        "version": 2,
        "n_folds": n_folds,
        "random_seed": random_seed,
        "fallback_applied": bool(category_changes),
        "category_changes": category_changes or {},
        "folds": folds_idx,
    }

    os.makedirs(
        os.path.dirname(split_path) or ".",
        exist_ok=True,
    )

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            indent=4,
            ensure_ascii=False,
        )


def make_kfold_splits(
    dataset_name: str,
    df_all: pd.DataFrame,
    target_col: str,
    cat_col_idx: list[int],
    n_folds: int = 5,
    random_seed: int = 0,
    split_dir: str = "splits",
    max_attempts: int = 100,
) -> tuple[list, pd.DataFrame]:
    """
    生成/复用 n_folds 折 train/test。

    处理顺序：

    1. 原始数据上尝试多个随机 seed，优先完整保留原始 categorical levels。
    2. 若始终存在 coverage 问题，只处理 singleton categorical levels，
       然后再次尝试多个随机 seed。
    3. 如果仍存在 test-only category，则针对最终 candidate split，
       只合并实际造成 vocabulary 不一致的类别。
       不改变 fold 索引，只改变有问题的 categorical value。
    4. 保存最终 folds + category fallback 信息，后续直接复用。

    返回：
        folds_idx, 实际用于该 split 的 df_all
    """
    split_path = os.path.join(
        split_dir,
        f"{dataset_name}_seed{random_seed}_{n_folds}fold.json"
    )

    # =====================================================
    # 0. 读取已有 cache
    # =====================================================
    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            cached = json.load(f)

        if isinstance(cached, dict) and "folds" in cached:
            folds_idx = cached["folds"]

            if len(folds_idx) != n_folds:
                raise ValueError(
                    f"{split_path} 已存在，但折数 "
                    f"({len(folds_idx)}) 和本次请求 "
                    f"({n_folds}) 不一致。"
                )

            category_changes = cached.get(
                "category_changes",
                {}
            )

            df_cached = (
                _apply_category_changes(
                    df_all,
                    category_changes,
                )
                if category_changes
                else df_all.copy()
            )

            valid, error = _check_categorical_coverage(
                df_cached,
                folds_idx,
                cat_col_idx,
            )

            if valid:
                print(
                    f"✅ 复用已有 5-fold split: "
                    f"{split_path}"
                )
                if category_changes:
                    print(
                        "   已重放 categorical fallback。"
                    )
                return folds_idx, df_cached

            print(
                f"⚠️ 已有 split cache 的 categorical "
                f"coverage 无效，将重新生成。原因：{error}"
            )

        elif isinstance(cached, list):
            # 兼容你回退前已经生成的旧版 list-format cache
            folds_idx = cached

            if len(folds_idx) != n_folds:
                raise ValueError(
                    f"{split_path} 已存在，但折数 "
                    f"({len(folds_idx)}) 和本次请求 "
                    f"({n_folds}) 不一致。"
                )

            valid, error = _check_categorical_coverage(
                df_all,
                folds_idx,
                cat_col_idx,
            )

            if valid:
                # 升级成新版 cache
                _save_kfold_cache(
                    split_path,
                    folds_idx,
                    n_folds,
                    random_seed,
                )
                print(
                    f"✅ 复用已有 split 并升级 cache: "
                    f"{split_path}"
                )
                return folds_idx, df_all.copy()

            print(
                f"⚠️ 旧 split cache 存在 categorical coverage "
                f"问题，将重新生成。原因：{error}"
            )

    # =====================================================
    # 1. 第一阶段：原始数据，多次随机尝试
    # =====================================================
    last_folds = None
    last_error = ""

    for attempt in range(max_attempts):
        candidate_folds = _build_kfold_candidate(
            df_all,
            target_col,
            n_folds,
            random_seed + attempt,
        )

        valid, error = _check_categorical_coverage(
            df_all,
            candidate_folds,
            cat_col_idx,
        )

        if valid:
            _save_kfold_cache(
                split_path,
                candidate_folds,
                n_folds,
                random_seed,
            )
            print(
                f"✅ categorical coverage valid "
                f"(original data, attempt {attempt + 1}/{max_attempts})"
            )
            return candidate_folds, df_all.copy()

        last_folds = candidate_folds
        last_error = error

    print(
        f"\n⚠️ 原始数据经过 {max_attempts} 次随机划分后，"
        f"仍无法保证 categorical coverage。"
    )
    print(f"   最后一次失败：{last_error}")

    # =====================================================
    # 2. 第二阶段：只处理 singleton categories
    # =====================================================
    df_processed, singleton_changes = (
        _collapse_singleton_categories(
            df_all,
            cat_col_idx,
        )
    )

    # 没有 singleton 就不再浪费一轮完全相同的数据尝试
    if singleton_changes:
        last_folds = None
        last_error = ""

        for attempt in range(max_attempts):
            candidate_folds = _build_kfold_candidate(
                df_processed,
                target_col,
                n_folds,
                random_seed + attempt,
            )

            valid, error = _check_categorical_coverage(
                df_processed,
                candidate_folds,
                cat_col_idx,
            )

            if valid:
                _save_kfold_cache(
                    split_path,
                    candidate_folds,
                    n_folds,
                    random_seed,
                    singleton_changes,
                )
                print(
                    f"✅ singleton fallback 后获得有效 split "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                return candidate_folds, df_processed

            last_folds = candidate_folds
            last_error = error

        print(
            f"⚠️ singleton fallback 后仍存在 "
            f"categorical coverage 问题：{last_error}"
        )

    # =====================================================
    # 3. 第三阶段：针对最后一个 candidate split，
    #    合并实际 test-only categories
    # =====================================================
    if last_folds is None:
        # 理论上不会进入；保险处理
        last_folds = _build_kfold_candidate(
            df_processed,
            target_col,
            n_folds,
            random_seed,
        )

    targeted_changes_df, targeted_changes = (
        _pool_test_only_categories(
            df_processed,
            last_folds,
            cat_col_idx,
        )
    )

    all_category_changes = {}

    # 合并 singleton fallback 的修改记录
    for col, changes in singleton_changes.items():
        all_category_changes.setdefault(col, [])
        all_category_changes[col].extend(changes)

    # 再合并针对 test-only category 的修改记录
    for col, changes in targeted_changes.items():
        all_category_changes.setdefault(col, [])
        all_category_changes[col].extend(changes)

    valid, error = _check_categorical_coverage(
        targeted_changes_df,
        last_folds,
        cat_col_idx,
    )

    if not valid:
        raise RuntimeError(
            "❌ categorical coverage fallback 后仍然无效。\n"
            f"原因：{error}"
        )

    _save_kfold_cache(
        split_path,
        last_folds,
        n_folds,
        random_seed,
        all_category_changes,
    )

    print(
        "✅ targeted categorical fallback 后获得有效 split。"
    )

    return last_folds, targeted_changes_df


def get_fold(
    df_all: pd.DataFrame,
    folds_idx: list,
    fold_idx: int,
):
    """按 fold_idx 取出这一折 train/test DataFrame。"""
    idx = folds_idx[fold_idx]

    df_train = df_all.loc[
        idx["train_idx"]
    ].reset_index(drop=True)

    df_test = df_all.loc[
        idx["test_idx"]
    ].reset_index(drop=True)

    return df_train, df_test


def get_fold(df_all: pd.DataFrame, folds_idx: list, fold_idx: int):
    """按 fold_idx 从 make_kfold_splits 的结果里取出这一折的 train/test DataFrame。"""
    idx = folds_idx[fold_idx]
    df_train = df_all.loc[idx["train_idx"]].reset_index(drop=True)
    df_test = df_all.loc[idx["test_idx"]].reset_index(drop=True)
    return df_train, df_test


def get_or_build_imputed_reference_cv(
    processor, df_train: pd.DataFrame, df_test: pd.DataFrame,
    dataset_name: str, random_seed: int, task: str, fold_idx: int, split_dir: str = "splits",
):
    """和 get_or_build_imputed_reference 是同一套逻辑，去掉了 val，按 fold_idx 单独存档。"""
    ref_dir = os.path.join(split_dir, f"{dataset_name}_seed{random_seed}_{task}_fold{fold_idx}_imputed")
    paths = {name: os.path.join(ref_dir, f"real_{name}_imputed.csv") for name in ("train", "test")}

    if all(os.path.exists(p) for p in paths.values()):
        return pd.read_csv(paths["train"]), pd.read_csv(paths["test"])

    os.makedirs(ref_dir, exist_ok=True)
    df_train_imputed = processor.transform_readable(df_train)
    df_test_imputed = processor.transform_readable(df_test)

    df_train_imputed.to_csv(paths["train"], index=False)
    df_test_imputed.to_csv(paths["test"], index=False)

    with open(os.path.join(ref_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset_name": dataset_name, "random_seed": random_seed, "task": task, "fold_idx": fold_idx,
            "n_train": len(df_train_imputed), "n_test": len(df_test_imputed),
        }, f, indent=4, ensure_ascii=False)

    return df_train_imputed, df_test_imputed


# 2. Task 路由
def route_task_data(df: pd.DataFrame, target_col: str, task: str, 
                    null_class_idx: int, target_minority_class: Hashable | None = None,):
    """
    full_data       —— target_col 作为普通类别特征保留，y 全为 null
    minority_uncond —— 仅少数类（label==1），target_col 剥离，y 全为 null
    minority_cond   —— 全量数据，target_col 剥离为条件标签(y 已是 0..K-1 整数)
    注意：进入本函数前，若 task 需要 label，target_col 应已由 TargetMapper 整数化。
    """
    if task == "full_data":
        df_features = df.copy()
        y = torch.full((len(df),), null_class_idx, dtype=torch.long)

    elif task == "minority_uncond":
        if target_minority_class is None:
            target_minority_class = df[target_col].value_counts().idxmin()
        df_minority = df[df[target_col] == target_minority_class].copy()
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
def compute_target_dist(df: pd.DataFrame, target_col: str, target_class) -> dict:
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