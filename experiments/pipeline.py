# 文件位置: experiments/pipeline.py

import os
import gc
import json
import time
import pickle
import pandas as pd

from datetime import datetime
from contextlib import redirect_stdout
from dataclasses import asdict
from sklearn.model_selection import train_test_split
from evaluation.core_evaluator import TabularEvaluator
from experiments.synthesizers import MedSynth
from evaluation.tool_stability import StabilityEvaluator


# =========================================================
# 主实验 Pipeline
# =========================================================
def run_pipeline(
        df_all: pd.DataFrame,
        target_col: str,
        cat_cols: list,
        dataset_name: str,
        synthesizer: MedSynth,
        ml_mode: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
):
    start_time = time.time()

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "训练、验证和测试集比例之和必须为1"

    # =====================================================
    # 获取实验配置
    # =====================================================
    exp_cfg = synthesizer.exp_cfg

    task          = exp_cfg.training.task
    objective     = exp_cfg.training.objective
    dim_hidden    = exp_cfg.model.dim_hidden
    num_timesteps = exp_cfg.model.num_timesteps
    num_layers    = exp_cfg.model.num_layers

    # =====================================================
    # 创建保存目录
    # =====================================================
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    dataset_dir = os.path.join(exp_cfg.pipeline.save_base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    exp_name = (
        f"{task}_{objective}_"
        f"h{dim_hidden}_t{num_timesteps}_{num_layers}_"
        f"{timestamp}"
    )
    save_path = os.path.join(dataset_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # =====================================================
    # 配置 log
    # =====================================================
    log_path = os.path.join(save_path, "console_output.txt")

    with open(log_path, 'w', encoding='utf-8', buffering=1) as log_f:
        with redirect_stdout(log_f):

            print("=" * 60)
            print("🚀 开始实验")
            print(f"📁 保存目录: {save_path}")
            print("=" * 60)
            print("\n📌 实验配置:")
            print(exp_cfg)

            # =====================================================
            # 数据划分
            # =====================================================
            unique_classes = df_all[target_col].nunique()
            stratify_col = df_all[target_col] if unique_classes > 1 else None

            df_temp, df_test = train_test_split(
                df_all,
                test_size=test_ratio,
                stratify=stratify_col,
                random_state=exp_cfg.pipeline.random_state,
            )

            stratify_col_temp = df_temp[target_col] if unique_classes > 1 else None
            val_ratio_in_temp = val_ratio / (train_ratio + val_ratio)

            df_train, df_val = train_test_split(
                df_temp,
                test_size=val_ratio_in_temp,
                stratify=stratify_col_temp,
                random_state=exp_cfg.pipeline.random_state,
            )

            print("\n📊 数据划分:")
            print(f"训练集: {len(df_train)}")
            print(f"验证集: {len(df_val)}")
            print(f"测试集: {len(df_test)}")

            # =====================================================
            # 模型训练
            # =====================================================
            print("\n⏳ 开始训练模型...")
            train_start = time.time()

            synthesizer.fit(df_train, df_val=df_val)

            train_time = time.time() - train_start
            print(f"✅ 训练完成 | 用时: {train_time / 60:.2f} min")

            # =====================================================
            # 保存模型
            # =====================================================
            synthesizer.save(save_path)

            # =====================================================
            # 生成插补后可读数据集（与训练预处理完全一致）
            # =====================================================
            print("\n🔧 生成插补后参考数据集...")

            df_train_imputed = synthesizer.transform_readable(df_train)
            df_val_imputed   = synthesizer.transform_readable(df_val)
            df_test_imputed  = synthesizer.transform_readable(df_test)

            # =====================================================
            # 生成合成数据
            # =====================================================
            print("\n🎲 开始生成数据...")
            sample_start = time.time()

            global_minority_class = df_all[target_col].value_counts().idxmin()

            df_fake = synthesizer.generate(target_class=global_minority_class)

            sample_time = time.time() - sample_start
            print(f"✅ 数据生成完成 | 用时: {sample_time:.2f} sec")

            # =====================================================
            # 保存 Loss History
            # =====================================================
            print("\n💾 保存 Loss History...")

            loss_history = synthesizer.loss_history   # 通过属性代理访问，不穿透到 diffusion
            loss_df = pd.DataFrame({
                "train_total": loss_history.train_total,
                "train_num":   loss_history.train_num,
                "train_cat":   loss_history.train_cat,
                "val_total":   loss_history.val_total,
            })
            loss_df.to_csv(os.path.join(save_path, "loss_history.csv"), index=False)

            # =====================================================
            # 保存实验结果（pkl）
            # =====================================================
            results = {
                'df_real_train':         df_train,
                'df_real_val':           df_val,
                'df_real_test':          df_test,
                'df_real_train_imputed': df_train_imputed,
                'df_real_val_imputed':   df_val_imputed,
                'df_real_test_imputed':  df_test_imputed,
                'df_fake':               df_fake,
                'save_path':             save_path,
            }

            with open(os.path.join(save_path, "results.pkl"), 'wb') as f:
                pickle.dump(results, f)

            # =====================================================
            # 保存 CSV
            # =====================================================
            print("\n💾 保存数据文件...")

            df_train.to_csv(os.path.join(save_path, "real_train.csv"),                 index=False)
            df_val.to_csv(  os.path.join(save_path, "real_val.csv"),                   index=False)
            df_test.to_csv( os.path.join(save_path, "real_test.csv"),                  index=False)

            df_train_imputed.to_csv(os.path.join(save_path, "real_train_imputed.csv"), index=False)
            df_val_imputed.to_csv(  os.path.join(save_path, "real_val_imputed.csv"),   index=False)
            df_test_imputed.to_csv( os.path.join(save_path, "real_test_imputed.csv"),  index=False)

            df_fake.to_csv( os.path.join(save_path, "fake_full.csv"),                  index=False)

            # =====================================================
            # 保存实验配置
            # =====================================================
            final_config = {
                "dataset_info": {
                    "name":       dataset_name,
                    "total_rows": len(df_all),
                    "target":     target_col,
                },
                "experiment_config": asdict(exp_cfg),
            }
            with open(
                os.path.join(save_path, "experiment_config.json"),
                'w', encoding='utf-8'
            ) as f:
                json.dump(final_config, f, indent=4, ensure_ascii=False)

            # =====================================================
            # 总时间统计
            # =====================================================
            total_time = time.time() - start_time
            print("\n" + "=" * 60)
            print("✅ 实验完成")
            print(f"🕒 Total Time: {total_time / 60:.2f} min")
            print("=" * 60, flush=True)

            # =====================================================
            # 评估（使用插补后数据作为 real 参照）
            # =====================================================
            evaluator = TabularEvaluator(
                df_real_train=df_train_imputed,
                df_fake=df_fake,
                df_real_test=df_test_imputed,
                target_col=target_col,
                cat_cols=cat_cols,
                ml_mode=ml_mode,
            )

            stat_res    = evaluator.evaluate_statistical_fidelity()
            privacy_res = evaluator.evaluate_privacy()
            cat_res     = evaluator.evaluate_categorical_proportions()

            print(stat_res)
            print(privacy_res)
            print(cat_res)
            print("\n✅ 当前实验完成")

    gc.collect()
    return results


# =========================================================
# 生成稳定性测试
# =========================================================
def run_stability_pipeline(
        df_train: pd.DataFrame,
        target_col: str,
        cat_cols: list,
        synthesizer: MedSynth,
        num_runs: int = 5,
):
    print("\n" + "=" * 60)
    print(f"🛡️ 开始生成稳定性测试 | 次数: {num_runs}")
    print("=" * 60)

    print("\n⏳ 正在训练生成模型...")
    synthesizer.fit(df_train)

    target_class = df_train[target_col].value_counts().idxmin()

    print(f"\n🎲 连续生成 {num_runs} 次数据...")
    list_of_fakes = []
    for i in range(num_runs):
        print(f"   ▶ 第 {i + 1} 次生成")
        df_fake = synthesizer.generate(target_class=target_class)
        list_of_fakes.append(df_fake)

    print("\n📊 开始稳定性评估...")
    evaluator = StabilityEvaluator(df_train, list_of_fakes, cat_cols)
    report = evaluator.evaluate_categorical_stability()

    print("\n✅ 稳定性测试完成")
    return list_of_fakes, report
