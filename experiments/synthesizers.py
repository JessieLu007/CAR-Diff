# 文件位置: experiments/synthesizers.py

import os
import json
import pickle
import torch
import numpy as np
import pandas as pd

from dataclasses import asdict, replace, fields
from typing import Optional
from torch.utils.data import DataLoader, TensorDataset

from configs.experiment_configs import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    ExperimentConfig,
)

from core.data_processor import TabularDataProcessor
from core.denoise_model import TabMLPDenoise
from core.diffusion import MedAGDiffusion


# =========================================================
# 主 Synthesizer
# =========================================================
class MedSynth:

    def __init__(
            self,
            target_col: str,
            cat_cols: list[str],
            exp_cfg: ExperimentConfig,
    ):
        self.target_col = target_col
        self.cat_cols = cat_cols

        self.exp_cfg = exp_cfg

        self.d_cfg = exp_cfg.data
        self.m_cfg = exp_cfg.model
        self.t_cfg = exp_cfg.training
        self.s_cfg = exp_cfg.sampling
        self.p_cfg = exp_cfg.pipeline

        self.diffusion: Optional[MedAGDiffusion] = None
        self.processor: Optional[TabularDataProcessor] = None
        self.original_columns: Optional[list] = None

        # 目标类分布（fit 后填充）
        self.target_dist: Optional[dict] = None
        # 实际参与建模的列（fit 后填充，task 路由后确定）
        self.model_feature_cols: Optional[list] = None

        self._label_predictor = None

    # =====================================================
    # 属性代理：让 pipeline 不必穿透到 diffusion 内部
    # =====================================================
    @property
    def loss_history(self):
        assert self.diffusion is not None, "请先 fit()"
        return self.diffusion.loss_history

    # =====================================================
    # 训练
    # =====================================================
    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None,
    ):
        #task = self.t_cfg.task
        self.df_train = df_train
        self.original_columns = df_train.columns.tolist()

        # -------------------------------------------------
        # 推断数值列
        # -------------------------------------------------
        num_cols = [
            c for c in df_train.columns
            if c not in self.cat_cols and c != self.target_col
        ]

        # -------------------------------------------------
        # 统计目标类分布（在路由之前，基于完整训练集）
        # -------------------------------------------------
        self._compute_target_dist(df_train)

        # -------------------------------------------------
        # 数据路由：根据 task 决定送入模型的数据与标签
        # -------------------------------------------------
        df_model_train, y_train = self._route(df_train, num_cols)
        df_model_val, y_val = (
            self._route(df_val, num_cols) if df_val is not None else (None, None)
        )

        # -------------------------------------------------
        # 确定参与建模的特征列（路由后固定）
        # -------------------------------------------------
        self.model_feature_cols = df_model_train.columns.tolist()

        # -------------------------------------------------
        # 预处理器：仅在 model_feature_cols 上 fit
        # -------------------------------------------------
        model_cat_cols = [c for c in self.cat_cols if c in self.model_feature_cols]

        if (self.t_cfg.task == "full_data" and self.target_col not in model_cat_cols
            and self.target_col in self.model_feature_cols):
            model_cat_cols.append(self.target_col)

        model_num_cols = [c for c in num_cols if c in self.model_feature_cols]

        self.processor = TabularDataProcessor(
            num_cols=model_num_cols,
            cat_cols=model_cat_cols,
            num_transform_method=self.d_cfg.num_transform_method,
            cat_encoding_method=self.d_cfg.cat_encoding_method,
            random_state=self.p_cfg.random_state,
        )
        self.processor.fit(df_model_train)

        # -------------------------------------------------
        # 构建 DataLoader
        # -------------------------------------------------
        train_loader = self._build_dataloader(
            df_model_train, y_train, shuffle=True
        )
        val_loader = (
            self._build_dataloader(df_model_val, y_val, shuffle=False)
            if df_model_val is not None else None
        )

        # -------------------------------------------------
        # 构建 meta_info（含 marginals）
        # -------------------------------------------------
        meta_info = self._build_meta_info(df_model_train, model_num_cols, model_cat_cols)

        # -------------------------------------------------
        # 创建并训练 diffusion
        # -------------------------------------------------
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise,
        )

        self.diffusion.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            meta_info=meta_info,
            train_config=self.t_cfg,
        )

    # =====================================================
    # 生成
    # =====================================================
    def generate(self, target_class: int = 1) -> pd.DataFrame:
        assert self.diffusion is not None, "请先 fit()"

        # -------------------------------------------------
        # 决定生成数量
        # -------------------------------------------------
        n_samples = (
            self.s_cfg.num_samples
            if self.s_cfg.num_samples is not None
            else self._decide_n_samples(target_class)
        )

        print(
            f"\n🎲 开始生成:   task={self.t_cfg.task}   objective={self.t_cfg.objective}   target_class={target_class}   n_samples={n_samples}"
        )

        temp_cfg = replace(self.s_cfg, num_samples=n_samples)

        # -------------------------------------------------
        # diffusion 返回原始 tensor
        # -------------------------------------------------
        raw_tensor = self.diffusion.generate(
            sampling_config=temp_cfg,
            y_cond=target_class,
        )

        # -------------------------------------------------
        # 逆变换：tensor → 原始量纲 DataFrame
        # -------------------------------------------------
        df_gen = self.processor.inverse_transform(raw_tensor.numpy())

        # -------------------------------------------------
        # 后处理：按 task 补回 target_col
        # -------------------------------------------------
        df_gen = self._postprocess(df_gen, target_class, n_samples)

        return df_gen

    # =====================================================
    # 插补后可读数据集（供对比实验使用）
    # =====================================================
    def transform_readable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对任意 DataFrame 做与训练一致的 MICE 插补，返回原始量纲可读版本。
        target_col 等不在 processor 管辖内的列会被直接透传。
        需在 fit() 之后调用。
        """
        assert self.processor is not None, "请先 fit()"
        return self.processor.transform_readable(df)

    # =====================================================
    # 私有：目标类分布统计
    # =====================================================
    def _compute_target_dist(self, df_train: pd.DataFrame):
        counts = df_train[self.target_col].value_counts()
        total = len(df_train)
        target_class = 1
        target_count = int(counts.get(target_class, 0))
        background_count = int(total - target_count)
        ir = background_count / target_count if target_count > 0 else float('inf')

        self.target_dist = {
            "target_class": target_class,
            "target_count": target_count,
            "background_count": background_count,
            "total": total,
            "IR": ir,
        }

    # =====================================================
    # 私有：数据路由
    # =====================================================
    def _route(
            self,
            df: pd.DataFrame,
            num_cols: list,
    ) -> tuple[pd.DataFrame, torch.Tensor]:
        """
        根据 task 决定送入模型的特征 DataFrame 和标签 tensor。

        返回：
            df_features  —— 不含 target_col 的特征 DataFrame（minority_uncond 时仅少数类）
            y_tensor     —— 标签 tensor，joint/uncond 时全为 null_class_idx
        """
        task = self.t_cfg.task
        null_idx = self.m_cfg.null_class_idx

        if task == "full_data":
            # target_col 作为普通类别特征一并送入，y 全为 null
            df_features = df.copy()
            y = torch.full((len(df),), null_idx, dtype=torch.long)

        elif task == "minority_uncond":
            # 仅保留少数类，y 全为 null，target_col 不参与特征
            df_minority = df[df[self.target_col] == 1].copy()
            df_features = df_minority.drop(columns=[self.target_col])
            y = torch.full((len(df_features),), null_idx, dtype=torch.long)

        elif task == "minority_cond":
            # 全量数据，target_col 作为条件标签
            y = torch.tensor(df[self.target_col].values, dtype=torch.long)
            df_features = df.drop(columns=[self.target_col])

        else:
            raise ValueError(f"未知 task: {task}")

        return df_features.reset_index(drop=True), y

    # =====================================================
    # 私有：构建 DataLoader
    # =====================================================
    def _build_dataloader(
            self,
            df_features: pd.DataFrame,
            y: torch.Tensor,
            shuffle: bool,
    ) -> DataLoader:
        x_encoded = self.processor.transform(df_features)
        x_tensor = torch.tensor(x_encoded.values, dtype=torch.float32)

        dataset = TensorDataset(x_tensor, y)
        return DataLoader(
            dataset,
            batch_size=self.t_cfg.batch_size,
            shuffle=shuffle,
        )

    # =====================================================
    # 私有：构建 meta_info
    # =====================================================
    def _build_meta_info(
            self,
            df_features: pd.DataFrame,
            model_num_cols: list,
            model_cat_cols: list,
    ) -> dict:
        info = self.processor.get_model_info()

        # 计算各类别列的边际分布（用于采样校正）
        x_encoded = self.processor.transform(df_features)
        num_num = info['num_numerical']
        cat_data = x_encoded.iloc[:, num_num:].values.astype(int)

        true_marginals = []
        for i, col in enumerate(model_cat_cols):
            n_classes = info['num_classes'][i]
            counts = np.bincount(cat_data[:, i], minlength=n_classes)
            true_marginals.append(
                torch.tensor(counts / counts.sum(), dtype=torch.float32)
            )

        info['true_marginals'] = true_marginals
        return info

    # =====================================================
    # 私有：生成后处理
    # =====================================================
    def _postprocess(
            self,
            df_gen: pd.DataFrame,
            target_class: int,
            n_samples: int,
    ) -> pd.DataFrame:
        """
        按 task 补回 target_col：
          - joint：target_col 已在特征中，无需补回
          - cond / uncond：用 target_class 填充
        """
        task = self.t_cfg.task

        if task == "full_data":
            pass
        else:
            df_gen[self.target_col] = target_class

        # 恢复原始列顺序
        if self.original_columns is not None:

            for col in self.original_columns:
                if col not in df_gen.columns:
                    df_gen[col] = np.nan
            df_gen = df_gen[self.original_columns]

        return df_gen

    # =====================================================
    # 私有：决定生成数量
    # =====================================================
    def _decide_n_samples(self, target_class: int) -> int:
        assert self.target_dist is not None, "请先 fit()"

        task = self.t_cfg.task
        objective = self.t_cfg.objective
        t_count = self.target_dist["target_count"]
        b_count = self.target_dist["background_count"]
        total = self.target_dist["total"]
        ir = self.target_dist["IR"]

        if task == "full_data":
            return total

        if task in ("minority_uncond", "minority_cond"):
            if objective == "fidelity":
                return t_count if t_count > 0 else 1
            else:
                if t_count == 0:
                    return 1
                if ir >= 2:
                    return max(b_count - t_count, 1)
                return 2 * t_count

        return total

    # =====================================================
    # 获取完整配置
    # =====================================================
    def get_config_dict(self):
        return asdict(self.exp_cfg)

    # =====================================================
    # 保存
    # =====================================================
    def save(self, path: str):
        assert self.diffusion is not None, "模型尚未训练"
        os.makedirs(path, exist_ok=True)

        # 模型权重
        torch.save(
            self.diffusion.state_dict(),
            os.path.join(path, "model.pt")
        )

        # 预处理器
        with open(os.path.join(path, "processor.pkl"), 'wb') as f:
            pickle.dump(self.processor, f)

        # 配置
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f, indent=4, ensure_ascii=False)

    # =====================================================
    # 加载
    # =====================================================
    def load(self, path: str):

        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)

            def safe_build(cls, d):
                valid_keys = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in d.items() if k in valid_keys})

            self.d_cfg = safe_build(DataConfig,    cfg_dict.get("data", {}))
            self.m_cfg = safe_build(ModelConfig,   cfg_dict.get("model", {}))
            self.t_cfg = safe_build(TrainingConfig, cfg_dict.get("training", {}))
            self.s_cfg = safe_build(SamplingConfig, cfg_dict.get("sampling", {}))

        # 预处理器
        processor_path = os.path.join(path, "processor.pkl")
        if os.path.exists(processor_path):
            with open(processor_path, 'rb') as f:
                self.processor = pickle.load(f)

        # 重建 diffusion 并加载权重
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise,
        )
        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        self.diffusion.load_state_dict(state)