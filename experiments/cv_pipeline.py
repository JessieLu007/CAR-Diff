# 文件位置: experiments/cv_pipeline.py
import os
import gc
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, asdict
from sklearn.model_selection import StratifiedKFold

from experiments.base_synthesizer import BaseSynthesizer
from evaluation.cv_runner import CrossValidationEvaluator

# =====================================================================
# 1. 交叉验证管线配置类
# =====================================================================
@dataclass
class CVPipelineConfig:
    n_splits: int = 5
    random_state: int = 42
    save_base_dir: str = "results"
    
# =====================================================================
# 2. 核心大管线：执行 5-Fold，保存结果，一键评估
# =====================================================================
def run_cv_pipeline(
    df_all: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    synthesizer: BaseSynthesizer, 
    cfg: CVPipelineConfig = CVPipelineConfig()
):
    algo_name = synthesizer.__class__.__name__
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    dataset_dir = os.path.join(cfg.save_base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    save_path = os.path.join(dataset_dir, f"CrossValid_{algo_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    print("\n" + "="*60)
    print(f"🚀 开始 CV 流水线 | 算法: {algo_name} | 数据集: {dataset_name}")
    print(f"📁 结果将保存至: {save_path}")
    print("="*60)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    fold_results = []
    
    # 自动获取全局少数类作为增强目标
    global_minority_class = df_all[target_col].value_counts().idxmin()

    for fold, (train_idx, test_idx) in enumerate(skf.split(df_all, df_all[target_col])):
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 正在运行 Fold {fold + 1} / {cfg.n_splits} ...")

        df_real_train = df_all.iloc[train_idx].copy()
        df_real_test = df_all.iloc[test_idx].copy()

        # 1. 拟合算法
        synthesizer.fit(df_real_train)
        
        # 2. 生成假数据
        df_fake = synthesizer.generate(
            target_class=global_minority_class,
        )

        fold_results.append({
            'fold': fold + 1,
            'df_real_train': df_real_train,
            'df_real_test': df_real_test,
            'df_fake': df_fake,
            'df_save_loc': save_path
        })
        gc.collect()

    # ==========================================
    # 结果落盘 (Pkl 与 Json)
    # ==========================================
    print("\n💾 正在保存五折数据集与参数配置...")
    with open(os.path.join(save_path, "fold_results.pkl"), 'wb') as f:
        pickle.dump(fold_results, f)
        
    final_config = {
        "pipeline_config": asdict(cfg),
        "dataset_info": {"name": dataset_name, "total_rows": len(df_all), "target": target_col},
        "algorithm_config": synthesizer.get_config_dict()
    }
    with open(os.path.join(save_path, "experiment_configs.json"), 'w') as f:
        json.dump(final_config, f, indent=4)

    print(f"\n✅ 全部任务完美收官！请前往 {save_path} 查看报告。")
    return fold_results

# =====================================================================
# 3. 简化单次运行：运行单次训练-验证-测试分割的评估流程 
# =====================================================================
from sklearn.model_selection import train_test_split
import pickle
import json

def run_single_split_pipeline(
        df_all: pd.DataFrame,
        target_col: str,
        dataset_name: str,
        synthesizer:BaseSynthesizer,
        cfg: CVPipelineConfig,
        train_ratio: float = 0.7,   # 新增参数，训练集比例
        val_ratio: float = 0.1,     # 新增参数，验证集比例
        test_ratio: float = 0.2,    # 新增参数，测试集比例
):
    #assert target_col in df_all.columns, f"数据集中找不到目标列"
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "训练、验证和测试集的比例之和必须为 1.0"
    
    unique_classes = df_all[target_col].nunique()
    stratify_col = df_all[target_col] if unique_classes > 1 else None

    algo_name = synthesizer.__class__.__name__
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    dataset_dir = os.path.join(cfg.save_base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    save_path = os.path.join(dataset_dir, f"SingleRun_{algo_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    print(f"⚡ 开始单次评估 | 算法: {algo_name} | 训练/验证/测试比例: {train_ratio}:{val_ratio}:{test_ratio}")

    # 1. 两次分层切分以获得训练、验证、测试集
    # 1.1 划分临时训练、验证集和测试集
    df_temp, df_test = train_test_split(
        df_all, test_size=test_ratio, stratify=stratify_col, random_state=cfg.random_state
    )
    # 针对df_temp调整分层列（如果存在）
    stratify_col_temp = df_temp[target_col] if unique_classes > 1 else None

    # 计算 df_temp中验证集占的比例 val_ratio_in_temp = val_ratio / (train_ratio + val_ratio)
    if train_ratio + val_ratio == 0:
        val_ratio_in_temp = 0 # safe code for no validation set
    else:
        val_ratio_in_temp = val_ratio / (train_ratio + val_ratio)
    # 1.2 从临时训练、验证集分出训练集和验证集
    df_train, df_val = train_test_split(
        df_temp, test_size = val_ratio_in_temp, stratify=stratify_col_temp, random_state=cfg.random_state
    )
    print(f"数据划分完成：训练集 {len(df_train)} 条, 验证集 {len(df_val)} 条, 测试集 {len(df_test)} 条")

    # 2. 拟合与生成
    print(f"\n>>> 正在拟合模型与生成数据...")
    synthesizer.fit(df_train, df_val=df_val) # 将验证集传递给synthesizer.fit方法
    synthesizer.save(save_path)


    global_minority_class = df_all[target_col].value_counts().idxmin()
    df_fake = synthesizer.generate(
        target_class=global_minority_class,
    )

    # 3. 包装成K=1的fold_results列表
    fold_results = [{
        'fold': 1,
        'df_real_train': df_train,
        'df_real_val': df_val, #新增验证集
        'df_real_test': df_test,
        'df_fake': df_fake,
        'df_save_loc': save_path
    }]

    # 4. 自动保存
    with open(os.path.join(save_path, "single_run_results.pkl"), 'wb') as f:
        pickle.dump(fold_results, f)

    # 4.5 额外保存csv文件方便查看
    df_train.to_csv(os.path.join(save_path, "real_train.csv"), index=False)
    df_val.to_csv(os.path.join(save_path, "real_val.csv"), index=False) # 保存验证集
    df_test.to_csv(os.path.join(save_path, "real_test.csv"), index=False)
    df_fake.to_csv(os.path.join(save_path, "fake_full.csv"), index=False)
        
    print(f"\n✅ 单次实验完成！报告在: {save_path}")
    return fold_results


# ====================================================
# 新增用于评估训练得到的模型的生成稳定性，即训练1次，生成5次
# ====================================================
from evaluation.tool_stability import StabilityEvaluator

def run_stability_pipeline(
        df_train: pd.DataFrame,
        target_col: str,
        cat_cols: list,
        synthesizer: BaseSynthesizer,
        num_runs: int = 5, # 连续生成的次数
):
    print("\n" + "="*60)
    print(f"🛡️ 开始生成鲁棒性/稳定性测试 (RQ2) | 运行次数: {num_runs}")
    print("="*60)

    # 1. 仅拟合一次模型 (Fit Once)
    print(">>> 1. 正在拟合生成算法 (仅执行一次) ...")
    synthesizer.fit(df_train)
    
    # 自动获取目标类
    target_class = df_train[target_col].value_counts().idxmin()
    
    # 2. 连续采样 N 次 (Sample N times)
    print(f">>> 2. 正在连续进行 {num_runs} 次独立随机采样 ...")
    list_of_fakes = []
    for i in range(num_runs):
        print(f"   - 正在生成第 {i+1} 批数据...")
        df_fake = synthesizer.generate(
            target_class=target_class,
        )
        list_of_fakes.append(df_fake)
        
    # 3. 调用你的稳定性评估器
    print(">>> 3. 进入稳定性评估器...")
    evaluator = StabilityEvaluator(df_train, list_of_fakes, cat_cols)
    report = evaluator.evaluate_categorical_stability()
    
    return list_of_fakes, report