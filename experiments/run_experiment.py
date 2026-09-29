# 文件位置: experiments/run_experiment.py
from dataclasses import asdict
import json
import numpy as np
import torch
import os, sys, gc, time, pickle
import pandas as pd
from datetime import datetime
from contextlib import redirect_stdout

from core.target_mapper import TargetMapper
from configs.experiment_configs import ExperimentConfig
from experiments.data_prep import (
    decode_target, encode_target, resolve_minority_class, route_task_data, compute_target_dist, build_processor, 
    make_kfold_splits, get_fold, get_or_build_imputed_reference_cv
)
from experiments.model_runner import build_dataloader, build_meta_info, train_model, generate_samples, save_run


class Tee:
    def __init__(self, *files): self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()


def run_experiment(
    df_all,
    target_col,
    num_col_idx,
    cat_col_idx,
    dataset_name,
    denoise_class,
    exp_cfg: ExperimentConfig,
    fold_idx: int,
    n_folds: int = 5,
    target_minority_class=None, #默认None，自动推断
    target_class_order: list | None=None, # optional: 显示类序，跨fold保持一致
    split_dir: str = "splits",
    evaluator_fn=None,   # 预留：(df_train_imputed, df_test_imputed, df_fake) -> dict
    only_splits=True,
):
    start_time = time.time()
    task = exp_cfg.training.task
    original_columns = df_all.columns.tolist()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    exp_name = (
        f"{task}_{exp_cfg.training.objective}_"
        f"h{exp_cfg.model.dim_hidden}_t{exp_cfg.model.num_timesteps}_{exp_cfg.model.num_layers}_{timestamp}"
    )
    save_path = os.path.join(
        exp_cfg.pipeline.save_base_dir, dataset_name,
        f"seed_{exp_cfg.pipeline.random_seed}_fold_{fold_idx}", exp_name,
    )
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "console_output.txt"), "w", encoding="utf-8", buffering=1) as log_f:
        with redirect_stdout(Tee(sys.stdout, log_f)):
            print("=" * 60); print(f"🚀 开始实验  📁 {save_path}"); print("=" * 60)

            target_mapper = TargetMapper(class_order=target_class_order).fit(df_all[target_col])
            print(f"🏷️ target 类别 {target_mapper.num_classes_} 个: {target_mapper.class_order_}")

            # null_class_idx 必须严格大于所有真实类别编码，按当前数据集类别数动态算
            from dataclasses import replace as dc_replace
            exp_cfg = dc_replace(exp_cfg, model=dc_replace(exp_cfg.model, null_class_idx=target_mapper.num_classes_))

            minority_class_raw = resolve_minority_class(
                df_all, target_col, target_minority_class,
            )
            print(f"🎯 少数类 raw={minority_class_raw!r}")

            # 只有 minority_cond / minority_uncond 才需要把 target 整数化；
            # full_data 保留原始 target（走 processor 的 cat 通道）
            #df_all_enc = df_all.copy()
            if task in ("minority_cond", "minority_uncond"):
                #df_all_enc[target_col] = target_mapper.transform(df_all[target_col])
                minority_class = target_mapper.mapping_[minority_class_raw]
            else:
                minority_class = minority_class_raw
            print(f"🎯 少数类 encoded={minority_class!r}")

            # 落盘映射表（供复核 / 反变换 / 跨 run 对齐）
            with open(os.path.join(save_path, "target_mapping.json"), "w", encoding="utf-8") as f:
                json.dump(target_mapper.to_dict(), f, indent=4, ensure_ascii=False)

            # ---------------- 数据划分 (五折， 不含验证集）----------------
            # folds_idx = make_kfold_splits(
            #     dataset_name, df_all, target_col, n_folds=n_folds,
            #     random_seed=exp_cfg.pipeline.random_seed, split_dir=split_dir,
            # )
            # df_train, df_test = get_fold(df_all, folds_idx, fold_idx)
            # print(f"📊 第{fold_idx+1}/{n_folds}折: train={len(df_train)} test={len(df_test)}")

            folds_idx, df_all = make_kfold_splits(
            dataset_name,
            df_all,
            target_col,
            cat_col_idx=cat_col_idx,
            n_folds=n_folds,
            random_seed=exp_cfg.pipeline.random_seed,
            split_dir=split_dir,
            )

            df_train, df_test = get_fold(df_all, folds_idx, fold_idx,)

            if task in ("minority_cond", "minority_uncond"):
                df_train[target_col] = target_mapper.transform(df_train[target_col])
                df_test[target_col] = target_mapper.transform(df_test[target_col])

            if task in ("minority_uncond", "minority_cond"):
                df_train_ref = df_train[df_train[target_col] == minority_class].copy()
            else:
                df_train_ref = df_train

            target_dist = compute_target_dist(df_train, target_col, minority_class) # target_minority_class

            # ---------------- task 路由 ----------------
            null_idx = exp_cfg.model.null_class_idx
            df_model_train, y_train = route_task_data(df_train_ref, target_col, task, null_idx, minority_class)

            # ---------------- 预处理器（只在本折训练集上 fit，val/test 里没见过的类别由 ordinal encoder 的
            #                   unknown-value 机制兜底，不会报错，也不会泄露） ----------------
            processor = build_processor(
                df_model_train, original_columns, num_col_idx, cat_col_idx, target_col, task,
                exp_cfg.data.num_transform_method, exp_cfg.data.cat_encoding_method,
                exp_cfg.pipeline.random_seed,
            )

            # ---------------- 插补后可读数据集（只依赖 processor，不依赖训练/生成，训练开始前就能算好并落盘） ----------------
            # 数据保存只在此保存一次，缓存在splits/下， results不再重复保存，靠split_info.json里的imputed_cache_dir定位回去
            df_train_imputed, df_test_imputed = get_or_build_imputed_reference_cv(
                processor, df_train, df_test, dataset_name,
                exp_cfg.pipeline.random_seed, task, fold_idx, split_dir,
                )
            imputed_cache_dir = os.path.join(
                split_dir, f"{dataset_name}_seed{exp_cfg.pipeline.random_seed}_{task}_fold{fold_idx}_imputed"
            )

            # ---------------- 划分信息（供后续复核/对比使用；real_train/test 的实际数据在 imputed_cache_dir 下） ----------------            
            split_info = {
                "n_folds": n_folds, "fold_idx": fold_idx,
                "random_seed": exp_cfg.pipeline.random_seed,
                "n_train": len(df_train), "n_test": len(df_test),
                "imputed_cache_dir": imputed_cache_dir,
            }
            with open(os.path.join(save_path, "split_info.json"), "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=4, ensure_ascii=False)

            if only_splits:
                return

            # ---------------- DataLoader / meta_info（不用验证集，val_loader 固定传 None，
            #                   过拟合缓解交给 weight decay / dropout 等训练期正则化） ----------------
            train_loader = build_dataloader(processor, df_model_train, y_train, 
                                            exp_cfg.training.batch_size, shuffle=True, 
                                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                                            last_batch_strategy=getattr(exp_cfg.training, "last_batch_strategy", "drop"))
            #num_cond_classes = target_mapper.num_classes_ if task == "minority_cond" else None
            meta_info = build_meta_info(processor, df_model_train)

            # ---------------- 训练 ----------------
            print("⏳ 训练...")
            t0 = time.time()
            diffusion = train_model(
                denoise_class, exp_cfg.model, exp_cfg.training, exp_cfg.sampling,
                train_loader, None, meta_info,
            )
            print(f"✅ 训练完成 | {(time.time()-t0)/60:.2f} min")

            # ---------------- Loss History（训练一结束立刻落盘+画图，不用等生成/评估跑完）----------------
            diffusion.loss_history.to_csv(os.path.join(save_path, "loss_history.csv"))
            diffusion.loss_history.plot(save_path=os.path.join(save_path, "loss_curve.png"))

            # ---------------- 保存（可重载产物）----------------
            #cat_cols_names = [original_columns[i] for i in cat_col_idx]
            save_run(
                save_path, processor, meta_info, diffusion, target_dist, target_col,
                processor.cat_cols, df_model_train.columns.tolist(), original_columns, exp_cfg, denoise_class,
            )

            # ---------------- 生成 ----------------
            t0 = time.time()
            df_fake = generate_samples(
                diffusion, processor, exp_cfg.sampling, task,
                minority_class, target_dist, target_col, original_columns,
            )        
            print(f"✅ 生成完成 | {time.time()-t0:.2f} sec")

            # ★ target 是标签时，把它还原成原始类名（字符串/数字类名都能还原）
            if task in ("minority_cond", "minority_uncond") and target_col in df_fake.columns:
                df_fake[target_col] = target_mapper.inverse_transform(df_fake[target_col].values)

            # ---------------- 落盘：生成数据（真实/插补数据已经在训练开始前存过了） ----------------
            df_fake.to_csv(os.path.join(save_path, "synthetic_data.csv"), index=False)

            results = {
                "df_real_train": df_train, "df_real_test": df_test,
                "df_real_train_imputed": df_train_imputed,
                "df_real_test_imputed": df_test_imputed, "df_fake": df_fake,
                "save_path": save_path,
            }


            # ---------------- 详细实验配置（含数据集信息） ----------------
            final_config = {
                "dataset_info": {
                    "name": dataset_name,
                    "total_rows": len(df_train) + len(df_test),
                    "target": target_col,
                    "fold_idx": fold_idx,
                    "n_folds": n_folds,
                },
                "experiment_config": asdict(exp_cfg),
            }
            with open(os.path.join(save_path, "experiment_config.json"), "w", encoding="utf-8") as f:
                json.dump(final_config, f, indent=4, ensure_ascii=False)


            # ---------------- 评估（接口预留，等评估代码到位后接上）----------------
            if evaluator_fn is not None:
                results["eval"] = evaluator_fn(df_train_imputed, df_test_imputed, df_fake)
                print(results["eval"])

            with open(os.path.join(save_path, "results.pkl"), "wb") as f:
                pickle.dump(results, f)

            print(f"✅ 实验完成 | 总用时 {(time.time()-start_time)/60:.2f} min")

    gc.collect()
    return results