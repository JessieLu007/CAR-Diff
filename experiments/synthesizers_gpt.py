# 文件位置: experiments/synthesizers.py

import os
import json
import torch
import pandas as pd
import numpy as np

from dataclasses import asdict, replace
from typing import Optional, Tuple, List, Dict
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from configs.experiment_configs import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    ExperimentConfig,
)

from core.denoise_model import TabMLPDenoise
from core.diffusion import MedAGDiffusion
from core.data_processor import TabularDataProcessor


class MedSynth:
    """
    医学表格数据生成器 (业务逻辑层)

    职责：
        1. 数据路由：根据 task 决定哪些样本/列送入模型, target_col 如何处理
        2. 预处理：调用 TabularDataProcessor 做编码、插补、数值变换
        3. DataLoader 构建：准备 (x_tensor, y_tensor) 形式的 train/val 批数据
        4. meta_info 准备：num_numerical, num_classes, true_marginals 等
        5. 生成：调用 MedAGDiffusion.generate() 得到 tensor, 再逆变换回 DataFrame
        6. 补回 target_col, 保持列顺序与原始 df 一致

    约定：
        - Pipeline 负责 split：df_train / df_val / df_test
        - MedSynth.fit() 只用 df_train / df_val
        - 任何评估所用的真实数据都直接使用 pipeline split 的 DataFrame
    """

    def __init__(
        self,
        target_col: str,
        cat_cols: List[str],
        num_cols: List[str],
        exp_cfg: ExperimentConfig,
    ):
        self.target_col = target_col
        self.cat_cols = cat_cols          # 原始的分类列（可能包含 target_col）
        self.num_cols = num_cols          # 原始的数值列
        self.exp_cfg = exp_cfg

        # 快捷引用
        self.d_cfg: DataConfig = exp_cfg.data
        self.m_cfg: ModelConfig = exp_cfg.model
        self.t_cfg: TrainingConfig = exp_cfg.training
        self.s_cfg: SamplingConfig = exp_cfg.sampling
        self.p_cfg = exp_cfg.pipeline

        # 模型与预处理组件
        self.diffusion: Optional[MedAGDiffusion] = None
        self.data_processor: Optional[TabularDataProcessor] = None

        # 记录用于 processor 的分类列（可能删掉 target_col，或添加）
        self.actual_cat_cols: Optional[List[str]] = None

        # 原始 DataFrame 的列顺序（用于 inverse_transform 后重排）
        self.original_columns: Optional[List[str]] = None

        # 目标分布统计（类别计数等）
        self.target_dist_info: Dict = {}

        # 训练时路由后用于 processor 的列名（num + cat）
        self.proc_feature_cols: Optional[List[str]] = None

    # =========================================================
    # 内部：根据 task 做数据路由
    # =========================================================
    def _route_data_for_task(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], torch.Tensor]:
        """
        根据 task 决定：
            - 哪些样本使用
            - target_col 是否留在特征中
            - y 条件标签如何构造

        返回：
            df_for_proc     —— 送入 TabularDataProcessor 的 DataFrame
            actual_cat_cols —— 实际的分类列列表（供 processor 使用）
            y_tensor        —— 条件标签 tensor（长度与 df_for_proc 一致）
        """
        task = self.t_cfg.task
        target_col = self.target_col
        actual_cat_cols = deepcopy(self.cat_cols)

        if task == "full_data_joint":
            # target_col 作为普通分类特征一起学
            if target_col not in actual_cat_cols:
                actual_cat_cols.append(target_col)
            df_for_proc = df.copy()
            y_tensor = torch.full(
                (len(df),),
                self.m_cfg.null_class_idx,
                dtype=torch.long,
            )

        elif task == "full_data_cond":
            # target_col 从特征中移除，作为条件标签
            if target_col in actual_cat_cols:
                actual_cat_cols.remove(target_col)
            df_for_proc = df.drop(columns=[target_col])
            y_tensor = torch.tensor(df[target_col].values, dtype=torch.long)

        elif task == "minority_uncond":
            # 只保留少数类=1 的样本，target_col 丢弃，y 全为 null
            df_minority = df[df[target_col] == 1].copy()
            if target_col in actual_cat_cols:
                actual_cat_cols.remove(target_col)
            df_for_proc = df_minority.drop(columns=[target_col])
            y_tensor = torch.full(
                (len(df_minority),),
                self.m_cfg.null_class_idx,
                dtype=torch.long,
            )

        elif task == "minority_cond":
            # 使用全部样本，target_col 作为条件，特征中剥离
            if target_col in actual_cat_cols:
                actual_cat_cols.remove(target_col)
            df_for_proc = df.drop(columns=[target_col])
            y_tensor = torch.tensor(df[target_col].values, dtype=torch.long)

        else:
            raise ValueError(
                f"未知 task='{task}'，合法值: "
                "'full_data_joint' | 'full_data_cond' | 'minority_uncond' | 'minority_cond'"
            )

        return df_for_proc, actual_cat_cols, y_tensor

    # =========================================================
    # 内部：构建 DataProcessor 并拟合
    # =========================================================
    def _build_and_fit_processor(
        self,
        df_train_for_proc: pd.DataFrame,
        actual_cat_cols: List[str],
    ) -> None:
        """
        使用训练数据拟合 TabularDataProcessor。
        num_cols 使用 self.num_cols 与 df_train_for_proc 的交集；
        cat_cols 使用 actual_cat_cols。
        """
        # 记录原始列顺序（基于“路由前”的 df_train）
        # 注意：真正的原始 df 列顺序，应当在 fit() 开头由调用传入的 df_train 决定
        # 这里不修改 self.original_columns

        # 数值列可能在路由后丢掉 target_col 后发生变化，取交集以防万一
        num_cols_for_proc = [c for c in self.num_cols if c in df_train_for_proc.columns]

        self.data_processor = TabularDataProcessor(
            num_cols=num_cols_for_proc,
            cat_cols=actual_cat_cols,
            num_transform_method=self.d_cfg.num_transform_method,
            random_state=self.d_cfg.random_state,
        )
        self.data_processor.fit(df_train_for_proc)

        # 记录 processor 实际使用的列
        self.actual_cat_cols = actual_cat_cols
        self.proc_feature_cols = num_cols_for_proc + actual_cat_cols

    # =========================================================
    # 内部：构建 DataLoader
    # =========================================================
    @staticmethod
    def _build_dataloader(
        df_for_proc_trans: pd.DataFrame,
        y_tensor: torch.Tensor,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        x_tensor = torch.tensor(df_for_proc_trans.values, dtype=torch.float32)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # =========================================================
    # 内部：统计 target 分布（用于 n_samples 决策等）
    # =========================================================
    def _compute_target_dist(self, df_train: pd.DataFrame) -> None:
        counts = df_train[self.target_col].value_counts()
        total = len(df_train)
        majority_class = int(counts.idxmax())
        majority_count = int(counts.max())
        minority_class = int(counts.idxmin())
        minority_count = int(counts.min())
        ir = majority_count / minority_count if minority_count > 0 else float("inf")

        self.target_dist_info = {
            "counts": counts.to_dict(),
            "majority_class": majority_class,
            "majority_count": majority_count,
            "minority_class": minority_class,
            "minority_count": minority_count,
            "total": total,
            "IR": ir,
        }

    # =========================================================
    # 内部：统计每个分类特征的真实边缘分布（true_marginals）
    # =========================================================
    def _compute_true_marginals(
        self,
        df_train_trans: pd.DataFrame,
        y_train_tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        在“路由+预处理后”的空间中统计每个分类特征的边缘分布。
        对于 minority_uncond / minority_cond, 只使用少数类样本做边缘统计；
        其他 task 使用全量样本。
        """
        num_numerical = len(self.data_processor.num_cols)
        num_classes = [self.data_processor.meta[c]["classes"] for c in self.actual_cat_cols]

        # 选择用于边缘统计的子集
        if self.t_cfg.task == "minority_uncond":
            df_for_marginal = df_train_trans  # 已经只包含少数类
        elif self.t_cfg.task == "minority_cond":
            minority_label = self.target_dist_info["minority_class"]
            mask = (y_train_tensor == minority_label).cpu().numpy()
            df_for_marginal = df_train_trans[mask]
            if df_for_marginal.empty:
                print(
                    f"Warning: No samples for minority class (label={minority_label}) "
                    f"found for marginals calculation. Using all data instead."
                )
                df_for_marginal = df_train_trans
        else:
            df_for_marginal = df_train_trans

        true_marginals: List[torch.Tensor] = []
        cat_start_idx = num_numerical

        for i, n_cls in enumerate(num_classes):
            col_data = df_for_marginal.iloc[:, cat_start_idx + i]
            val_counts = col_data.value_counts(normalize=True).sort_index()
            marginal = torch.zeros(n_cls)
            for k, v in val_counts.items():
                marginal[int(k)] = float(v)
            true_marginals.append(torch.clamp(marginal, min=1e-6))

        return true_marginals

    # =========================================================
    # 公开 API：训练
    # =========================================================
    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: Optional[pd.DataFrame] = None,
    ):
        """
        df_train / df_val 由 pipeline 事先划分好，此处不做任何 split。
        整个训练/生成流程中，预处理与路由规则保持一致。
        """
        assert self.target_col in df_train.columns, "df_train 中必须包含 target_col"

        # 记录“路由前”的原始列顺序，用于 inverse_transform 后重排
        self.original_columns = df_train.columns.tolist()

        # 记录 target 分布（始终用全量训练集统计）
        self._compute_target_dist(df_train)

        # ---------- 1. 路由训练数据 ----------
        df_train_for_proc, actual_cat_cols, y_train = self._route_data_for_task(df_train)

        # ---------- 2. 构建并拟合 DataProcessor ----------
        self._build_and_fit_processor(df_train_for_proc, actual_cat_cols)

        # -----------2.5 处理补全缺失值的数据版本 ---------------
        self.df_train_imputed = self.data_processor.transform(
            df_train_for_proc,
        
        )

        # ---------- 3. 训练数据变换 ----------
        df_train_trans = self.data_processor.transform(df_train_for_proc)

        # ---------- 4. 统计 true_marginals ----------
        true_marginals = self._compute_true_marginals(df_train_trans, y_train)

        # ---------- 5. 构建 train_dataloader ----------
        train_loader = self._build_dataloader(
            df_for_proc_trans=df_train_trans,
            y_tensor=y_train,
            batch_size=self.t_cfg.batch_size,
            shuffle=True,
        )

        # ---------- 6. 处理验证数据（可选） ----------
        val_loader = None
        if df_val is not None:
            # 使用同样的 task 路由规则（注意：不重新拟合 processor）
            df_val_for_proc, _, y_val = self._route_data_for_task(df_val)
            df_val_trans = self.data_processor.transform(df_val_for_proc)
            val_loader = self._build_dataloader(
                df_for_proc_trans=df_val_trans,
                y_tensor=y_val,
                batch_size=self.t_cfg.batch_size,
                shuffle=False,
            )

        # ---------- 7. 准备 meta_info 并初始化扩散模型 ----------
        meta_info = self.data_processor.get_model_info()
        # 替换为路由后实际使用的值
        meta_info["num_numerical"] = len(self.data_processor.num_cols)
        meta_info["num_classes"] = [
            self.data_processor.meta[c]["classes"] for c in self.actual_cat_cols
        ]
        meta_info["true_marginals"] = [m.to(self.m_cfg.device) if hasattr(self.m_cfg, "device") else m for m in true_marginals]

        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise,
        )

        # ---------- 8. 训练扩散模型 ----------
        self.diffusion.fit(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            meta_info=meta_info,
            train_config=self.t_cfg,
        )

        return self

    # =========================================================
    # 内部：根据任务自动决定生成数量
    # =========================================================
    def _decide_n_samples(
        self,
        target_class: int,
    ) -> int:
        assert self.target_dist_info, "请先调用 fit() 初始化 target 分布信息"

        task = self.t_cfg.task
        objective = self.t_cfg.objective

        t_count = int(self.target_dist_info["counts"].get(target_class, 0))
        b_count = int(self.target_dist_info["total"] - t_count)
        total = int(self.target_dist_info["total"])
        ir = float(self.target_dist_info["IR"])

        # full joint/cond: 生成与原始训练集规模相同的数据
        if task in ("full_data_joint", "full_data_cond"):
            return total

        # minority 相关任务
        if task in ("minority_uncond", "minority_cond"):
            if objective == "fidelity":
                return t_count if t_count > 0 else 1

            # objective == "balance" 或类似含义
            if t_count == 0:
                return 1

            if ir >= 2.0:
                return max(b_count - t_count, 1)
            return 2 * t_count

        return total

    # =========================================================
    # 内部：生成后补回 target_col + 重排列顺序
    # =========================================================
    def _postprocess_generated_df(
        self,
        df_gen: pd.DataFrame,
        target_class: Optional[int],
    ) -> pd.DataFrame:
        """
        根据 task 补回 target_col，并将列顺序恢复到 original_columns。
        """
        task = self.t_cfg.task
        target_col = self.target_col

        if task == "full_data_joint":
            # target_col 已经在 processor 内还原，无需额外操作
            pass

        elif task == "full_data_cond":
            assert target_class is not None, \
                "task='full_data_cond' 时，generate() 必须指定 target_class"
            df_gen[target_col] = target_class

        elif task == "minority_uncond":
            df_gen[target_col] = 1

        elif task == "minority_cond":
            df_gen[target_col] = target_class if target_class is not None else 1

        else:
            raise ValueError(f"未知 task='{task}'")

        # 如果记录了原始列顺序，则重排
        if self.original_columns is not None:
            # 某些 task 下 target_col 在原特征中被移除/添加，确保列存在再重排
            for col in self.original_columns:
                if col not in df_gen.columns:
                    # 如果预处理中丢了某列，可以按需要填充，当前简单填 NaN
                    df_gen[col] = np.nan
            df_gen = df_gen[self.original_columns]

        return df_gen

    # =========================================================
    # 公开 API：生成
    # =========================================================
    def generate(
        self,
        target_class: int = 1,
    ) -> pd.DataFrame:
        """
        生成一个 DataFrame，与原始空间对齐：
            - 通过 diffusion 采样得到 tensor
            - 使用已拟合的 data_processor.inverse_transform() 还原
            - 按 task 补回 target_col
        """
        assert self.diffusion is not None, "请先调用 fit() 训练模型"
        assert self.data_processor is not None, "data_processor 尚未初始化，请先 fit()"

        # 决定生成数量
        if self.s_cfg.num_samples is not None:
            n_samples = self.s_cfg.num_samples
        else:
            n_samples = self._decide_n_samples(target_class)

        print(
            f"\n🎲 开始生成:"
            f"\n   task={self.t_cfg.task}"
            f"\n   objective={self.t_cfg.objective}"
            f"\n   target_class={target_class}"
            f"\n   n_samples={n_samples}"
        )

        # 构造临时 SamplingConfig
        temp_cfg = replace(self.s_cfg, num_samples=n_samples)

        # 调用 diffusion 生成原始 tensor
        raw_tensor = self.diffusion.generate(
            sampling_config=temp_cfg,
            y_cond=target_class,
            return_latent=False,
        )  # shape: (n_samples, n_features)

        # 逆变换到 DataFrame 空间
        df_gen = self.data_processor.inverse_transform(raw_tensor.numpy())

        # 补回 target_col + 按 original_columns 重排
        df_gen = self._postprocess_generated_df(df_gen, target_class=target_class)

        return df_gen

    # =========================================================
    # 配置导出 / 保存 / 加载
    # =========================================================
    def get_config_dict(self) -> dict:
        return asdict(self.exp_cfg)

    def save(self, path: str):
        """
        只保存 diffusion 权重和实验配置。
        注意：当前未序列化 TabularDataProcessor，如果要完整复现，
        需要你后续根据需求将 processor 的 state 也序列化。
        """
        assert self.diffusion is not None, "模型尚未训练，无法保存"

        os.makedirs(path, exist_ok=True)

        # 保存扩散模型
        torch.save(
            self.diffusion.state_dict(),
            os.path.join(path, "model.pt"),
        )

        # 保存实验配置
        with open(
            os.path.join(path, "config.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.get_config_dict(), f, indent=4, ensure_ascii=False)

    def load(self, path: str):
        """
        加载扩散模型权重和配置。
        注意：当前不恢复 TabularDataProcessor 和 target_dist_info，
        一般用于同一进程中先 fit，再 save/load；或者你在外部重建 processor。
        """
        config_path = os.path.join(path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)

            # 安全重建 ExperimentConfig 及子配置
            def safe_build(cls, d):
                from dataclasses import fields
                valid_keys = {f.name for f in fields(cls)}
                return cls(**{k: v for k, v in d.items() if k in valid_keys})

            self.d_cfg = safe_build(DataConfig, cfg_dict.get("data", {}))
            self.m_cfg = safe_build(ModelConfig, cfg_dict.get("model", {}))
            self.t_cfg = safe_build(TrainingConfig, cfg_dict.get("training", {}))
            self.s_cfg = safe_build(SamplingConfig, cfg_dict.get("sampling", {}))
            # pipeline 配置信息（如果有）
            if "pipeline" in cfg_dict:
                self.p_cfg = cfg_dict["pipeline"]

            self.exp_cfg = ExperimentConfig(
                data=self.d_cfg,
                model=self.m_cfg,
                training=self.t_cfg,
                sampling=self.s_cfg,
                pipeline=self.p_cfg,
            )

        # 重建 diffusion 结构并加载权重
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise,
        )

        state = torch.load(
            os.path.join(path, "model.pt"),
            map_location="cpu",
        )
        self.diffusion.load_state_dict(state)