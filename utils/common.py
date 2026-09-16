import os
import json
import pickle


# 训练数据集名称
DATASETS = ['framingham', 'hepatitis']
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