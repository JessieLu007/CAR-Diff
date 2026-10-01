import os
import json
import pickle


# 训练数据集名称
DATASETS = ['chronic_kidney_disease','diabetic_data','framingham', 'hcv_egy_data','hepatitis','risk_factors_cervical_cancer']
SEEDS = [2026, 2027, 2028, 2029, 2030]

# 对比方法名称
BASELINE_MODELS = ['smotenc','ctgan', 'tvae', 'tabddpm', 'tabsyn']


def create_path (path):
    """创建路径，如果已经存在就跳过"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"路径不存在，已创建：{path}")
    else:
        print(f"路径已存在：{path}")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        context = json.load(f)
    return context


def resolve_dataset(token):
    """
    dataset可以使用名称或索引
    """
    try:
        i = int(token)
    except (ValueError, TypeError):
        i = None

    if i is not None:
        if not (0 <= i < len(DATASETS)):
            listing = "\n".join(f"  [{k}] {n}" for k, n in enumerate(DATASETS))
            raise SystemExit(
                f"数据集索引 {i} 越界。common.DATASETS 里有：\n{listing}\n"
                f"（新数据集请直接用名称，例如 --dataset my_new_dataset）"
            )
        return DATASETS[i]

    name = str(token)
    if name in DATASETS:
        print(f"✔ 已知数据集: {name}")
    else:
        print(f"⚠ '{name}' 不在 common.DATASETS 中，将作为新数据集尝试加载"
              f"（需要 data/{name}.csv 和 data/datasets_info.json 里有对应条目）")
    return name        
