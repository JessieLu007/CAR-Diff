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
from evaluation.evaluator import CrossValidationEvaluator

# =====================================================================
# 1. 交叉验证管线配置类
# =====================================================================
@dataclass
class CVPipelineConfig:
    n_splits: int = 5
    random_state: int = 42
    save_base_dir: str = "results"
    
    # 是否在生成结束后自动调用 Evaluator 计算指标
    auto_evaluate: bool = False

# =====================================================================
# 2. 核心大管线：执行 5-Fold，保存结果，一键评估
# =====================================================================
def run_cv_pipeline(
    df_all: pd.DataFrame,
    target_col: str,
    cat_cols: list,
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

    # ==========================================
    # 自动评估 (可选)
    # ==========================================
    if cfg.auto_evaluate:
        print("\n📊 数据保存完毕，自动启动评估器 (Evaluator)...")
        evaluator = CrossValidationEvaluator(fold_results, target_col, cat_cols)
        stats_agg, ml_agg = evaluator.run()
        evaluator.save_results(stats_agg, ml_agg, save_dir=save_path)

    print(f"\n✅ 全部任务完美收官！请前往 {save_path} 查看报告。")
    return fold_results

# =====================================================================
# 3. 简化单次运行：执行一次运行用于模型调整
# =====================================================================
from sklearn.model_selection import train_test_split

def run_single_split_pipeline(
        df_all: pd.DataFrame,
        target_col: str,
        cat_cols: list,
        dataset_name: str,
        synthesizer:BaseSynthesizer,
        cfg: CVPipelineConfig,
        test_size: float = 0.2
):
    #assert target_col in df_all.columns, f"数据集中找不到目标列"
    unique_classes = df_all[target_col].nunique()
    stratify_col = df_all[target_col] if unique_classes > 1 else None
    """Hold-out Validation，复用CV的评估和落盘系统"""
    algo_name = synthesizer.__class__.__name__
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    dataset_dir = os.path.join(cfg.save_base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    save_path = os.path.join(dataset_dir, f"SingleRun_{algo_name}_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    print(f"⚡ 开始单次评估 | 算法: {algo_name} | 测试集比例: {test_size}")

    # 1. 单次分层切分
    df_train, df_test = train_test_split(
        df_all, test_size = test_size, stratify=stratify_col, random_state=cfg.random_state
    )

    # 2. 拟合与生成
    print(f"\n>>> 正在拟合模型与生成数据...")
    synthesizer.fit(df_train)

    global_minority_class = df_all[target_col].value_counts().idxmin()
    df_fake = synthesizer.generate(
        target_class=global_minority_class,
    )
    #print("DEBUG: df_fake columns in pipeline:", df_fake.columns.tolist())

    # 3. 包装成K=1的fold_results列表
    fold_results = [{
        'fold': 1,
        'df_real_train': df_train,
        'df_real_test': df_test,
        'df_fake': df_fake,
        'df_save_loc': save_path
    }]

    # 4. 自动保存
    with open(os.path.join(save_path, "single_run_results.pkl"), 'wb') as f:
        pickle.dump(fold_results, f)

    # 4.5 额外保存csv文件方便查看
    df_train.to_csv(os.path.join(save_path, "real_train.csv"), index=False)
    df_test.to_csv(os.path.join(save_path, "real_test.csv"), index=False)
    df_fake.to_csv(os.path.join(save_path, "fake_full.csv"), index=False)

    # 5. 复用Evaluator
    if cfg.auto_evaluate:
        print("\n📊 数据生成完毕，正在进行质量评估...")
        evaluator = CrossValidationEvaluator(fold_results, target_col, cat_cols)
        stats_agg, ml_agg = evaluator.run()
        evaluator.save_results(stats_agg, ml_agg, save_dir=save_path)

    # 6. 保存配置文件
    config_dict = synthesizer.get_config_dict()
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
        
    print(f"\n✅ 单次实验完成！报告在: {save_path}")
    return fold_results


# ====================================================
# 新增用于评估训练得到的模型的生成稳定性，即训练1次，生成5次
# ====================================================
from evaluation.evaluator import StabilityEvaluator

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