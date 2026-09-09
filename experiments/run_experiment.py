# 文件位置: experiments/run_experiment.py
from dataclasses import asdict
import json
import os, sys, gc, time, pickle
import pandas as pd
from datetime import datetime
from contextlib import redirect_stdout

from configs.experiment_configs import ExperimentConfig
from experiments.data_prep import split_dataset, route_task_data, compute_target_dist, build_processor, get_or_build_imputed_reference
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
    target_minority_class: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    split_dir: str = "splits",
    evaluator_fn=None,   # 预留：(df_train_imputed, df_val_imputed, df_test_imputed, df_fake) -> dict
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
        f"seed_{exp_cfg.pipeline.random_seed}", exp_name,
    )
    os.makedirs(save_path, exist_ok=True)

    with open(os.path.join(save_path, "console_output.txt"), "w", encoding="utf-8", buffering=1) as log_f:
        with redirect_stdout(Tee(sys.stdout, log_f)):
            print("=" * 60); print(f"🚀 开始实验  📁 {save_path}"); print("=" * 60)

            # ---------------- 数据划分 ----------------
            df_train, df_val, df_test = split_dataset(
                dataset_name, df_all, target_col, train_ratio, val_ratio, test_ratio,
                random_seed=exp_cfg.pipeline.random_seed, split_dir=split_dir,
            )
            print(f"📊 划分: train={len(df_train)} val={len(df_val)} test={len(df_test)}")

            if task in ("minority_uncond", "minority_cond"):
                df_train_ref = df_train[df_train[target_col] == 1].copy()
                df_val_ref = df_val[df_val[target_col] == 1].copy()
            else:
                df_train_ref, df_val_ref = df_train, df_val

            target_dist = compute_target_dist(df_train, target_col, target_minority_class)

            # ---------------- task 路由 ----------------
            null_idx = exp_cfg.model.null_class_idx
            df_model_train, y_train = route_task_data(df_train_ref, target_col, task, null_idx)
            df_model_val, y_val = (
                route_task_data(df_val_ref, target_col, task, null_idx)
                if len(df_val_ref) else (None, None)
            )

            # ---------------- 预处理器 ----------------
            processor = build_processor(
                df_model_train, original_columns, num_col_idx, cat_col_idx, target_col, task,
                exp_cfg.data.num_transform_method, exp_cfg.data.cat_encoding_method,
                exp_cfg.pipeline.random_seed,
            )

            # ---------------- DataLoader / meta_info ----------------
            train_loader = build_dataloader(processor, df_model_train, y_train, exp_cfg.training.batch_size, shuffle=True)
            val_loader = (
                build_dataloader(processor, df_model_val, y_val, exp_cfg.training.batch_size, shuffle=False)
                if df_model_val is not None else None
            )
            meta_info = build_meta_info(processor, df_model_train)

            # ---------------- 训练 ----------------
            print("⏳ 训练...")
            t0 = time.time()
            diffusion = train_model(
                denoise_class, exp_cfg.model, exp_cfg.training, exp_cfg.sampling,
                train_loader, val_loader, meta_info,
            )
            print(f"✅ 训练完成 | {(time.time()-t0)/60:.2f} min")

            # ---------------- 保存（可重载产物）----------------
            cat_cols_names = [original_columns[i] for i in cat_col_idx]
            save_run(
                save_path, processor, meta_info, diffusion, target_dist, target_col,
                processor.cat_cols, df_model_train.columns.tolist(), original_columns, exp_cfg, denoise_class,
            )

            # ---------------- 划分信息（供后续复核/对比使用） ----------------
            split_info = {
                "train_ratio": train_ratio, "val_ratio": val_ratio, "test_ratio": test_ratio,
                "random_seed": exp_cfg.pipeline.random_seed,
                "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
            }
            with open(os.path.join(save_path, "split_info.json"), "w", encoding="utf-8") as f:
                json.dump(split_info, f, indent=4, ensure_ascii=False)

            # ---------------- 插补后可读数据集 ----------------
            #df_train_imputed = processor.transform_readable(df_train)
            #df_val_imputed = processor.transform_readable(df_val)
            #df_test_imputed = processor.transform_readable(df_test)

            df_train_imputed, df_val_imputed, df_test_imputed = get_or_build_imputed_reference(
                processor, df_train, df_val, df_test, dataset_name,
                exp_cfg.pipeline.random_seed, task, split_dir,
                )

            # ---------------- 生成 ----------------
            t0 = time.time()
            df_fake = generate_samples(
                diffusion, processor, exp_cfg.sampling, task,
                target_minority_class, target_dist, target_col, original_columns,
            )
            print(f"✅ 生成完成 | {time.time()-t0:.2f} sec")

            # ---------------- Loss History ----------------
            history = diffusion.loss_history
            n = len(history.train_total)
            loss_df = pd.DataFrame({
                "train_total": history.train_total,
                "train_num":   history.train_num,
                "train_cat":   history.train_cat,
                "val_total":   history.val_total + [float('nan')] * (n - len(history.val_total)),
            })
            loss_df.to_csv(os.path.join(save_path, "loss_history.csv"), index=False)

            # ---------------- 落盘：原始 / 插补 / 生成数据 ----------------
            frames = {
                "real_train": df_train, "real_val": df_val, "real_test": df_test,
                "real_train_imputed": df_train_imputed, "real_val_imputed": df_val_imputed,
                "real_test_imputed": df_test_imputed, "synthetic_data": df_fake,
            }
            for name, d in frames.items():
                d.to_csv(os.path.join(save_path, f"{name}.csv"), index=False)

            results = {f"df_{k}": v for k, v in frames.items()}
            results["df_fake"] = results.pop("df_synthetic_data")  # 沿用旧 key，CSV 文件名不变
            results["save_path"] = save_path


            # ---------------- 详细实验配置（含数据集信息，此前遗漏，现补回） ----------------
            final_config = {
                "dataset_info": {
                    "name": dataset_name,
                    "total_rows": len(df_train) + len(df_val) + len(df_test),
                    "target": target_col,
                },
                "experiment_config": asdict(exp_cfg),
            }
            with open(os.path.join(save_path, "experiment_config.json"), "w", encoding="utf-8") as f:
                json.dump(final_config, f, indent=4, ensure_ascii=False)


            # ---------------- 评估（接口预留，等评估代码到位后接上）----------------
            if evaluator_fn is not None:
                results["eval"] = evaluator_fn(df_train_imputed, df_val_imputed, df_test_imputed, df_fake)
                print(results["eval"])

            with open(os.path.join(save_path, "results.pkl"), "wb") as f:
                pickle.dump(results, f)

            print(f"✅ 实验完成 | 总用时 {(time.time()-start_time)/60:.2f} min")

    gc.collect()
    return results