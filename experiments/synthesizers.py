# 文件位置: experiments/synthesizers.py

import os
import json
import torch
import pandas as pd

from dataclasses import asdict, replace, fields
from typing import Optional

from configs.experiment_configs import (
    DataConfig,
    ModelConfig,
    TrainingConfig,
    SamplingConfig,
    ExperimentConfig,
)

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

        # 完整实验配置
        self.exp_cfg = exp_cfg

        # 配置快捷引用
        self.d_cfg = exp_cfg.data
        self.m_cfg = exp_cfg.model
        self.t_cfg = exp_cfg.training
        self.s_cfg = exp_cfg.sampling
        self.p_cfg = exp_cfg.pipeline

        # diffusion模型
        self.diffusion: Optional[MedAGDiffusion] = None

        # 数据分布统计
        self.target_dist = None

    # =====================================================
    # 训练
    # =====================================================
    def fit(
            self,
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None
    ):

        # -------------------------------------------------
        # 统计类别分布
        # -------------------------------------------------
        counts = df_train[self.target_col].value_counts()

        total = len(df_train)

        target_class = 1

        target_count = int(counts.get(target_class, 0))

        background_count = int(total - target_count)

        ir = (
            background_count / target_count
            if target_count > 0
            else float('inf')
        )

        self.target_dist = {
            "target_class": target_class,
            "target_count": target_count,
            "background_count": background_count,
            "total": total,
            "IR": ir,
        }

        # -------------------------------------------------
        # 自动推断数值列
        # -------------------------------------------------
        num_cols = [
            c for c in df_train.columns
            if c not in self.cat_cols
               and c != self.target_col
        ]

        # -------------------------------------------------
        # 创建 diffusion
        # -------------------------------------------------
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise
        )

        # -------------------------------------------------
        # 开始训练
        # -------------------------------------------------
        self.diffusion.fit(
            df_train=df_train,
            df_val=df_val,
            target_col=self.target_col,
            num_cols=num_cols,
            cat_cols=self.cat_cols,
            train_config=self.t_cfg,
            data_config=self.d_cfg,
        )

    # =====================================================
    # 自动决定生成数量
    # =====================================================
    def _decide_n_samples(
            self,
            target_class: int
    ) -> int:

        assert self.target_dist is not None, \
            "请先 fit()"

        task = self.t_cfg.task

        objective = self.t_cfg.objective

        t_count = self.target_dist["target_count"]

        b_count = self.target_dist["background_count"]

        total = self.target_dist["total"]

        ir = self.target_dist["IR"]

        # -------------------------------------------------
        # full joint/cond
        # -------------------------------------------------
        if task in (
                "full_data_joint",
                "full_data_cond"
        ):
            return total

        # -------------------------------------------------
        # minority
        # -------------------------------------------------
        if task in (
                "minority_uncond",
                "minority_cond"
        ):

            if objective == "fidelity":

                return (
                    t_count
                    if t_count > 0 else 1
                )

            else:

                if t_count == 0:
                    return 1

                if ir >= 2:
                    return max(
                        b_count - t_count,
                        1
                    )

                return 2 * t_count

        return total

    # =====================================================
    # 生成
    # =====================================================
    def generate(
            self,
            target_class: int = 1
    ) -> pd.DataFrame:

        assert self.diffusion is not None, \
            "请先 fit()"

        # -------------------------------------------------
        # 自动决定生成数量
        # -------------------------------------------------
        if self.s_cfg.num_samples is not None:

            n_samples = self.s_cfg.num_samples

        else:

            n_samples = self._decide_n_samples(
                target_class
            )

        print(
            f"\n🎲 开始生成:"
            f"\n   task={self.t_cfg.task}"
            f"\n   objective={self.t_cfg.objective}"
            f"\n   target_class={target_class}"
            f"\n   n_samples={n_samples}"
        )

        # -------------------------------------------------
        # 临时生成配置
        # -------------------------------------------------
        temp_cfg = replace(
            self.s_cfg,
            num_samples=n_samples
        )

        # -------------------------------------------------
        # diffusion generate
        # -------------------------------------------------
        df_gen = self.diffusion.generate(
            sampling_config=temp_cfg,
            y_cond=target_class
        )

        return df_gen

    # =====================================================
    # 获取完整配置
    # =====================================================
    def get_config_dict(self):

        return asdict(self.exp_cfg)

    # =====================================================
    # 保存
    # =====================================================
    def save(self, path: str):

        assert self.diffusion is not None, \
            "模型尚未训练"

        os.makedirs(path, exist_ok=True)

        # -------------------------------------------------
        # 保存模型权重
        # -------------------------------------------------
        torch.save(
            self.diffusion.state_dict(),
            os.path.join(path, "model.pt")
        )

        # -------------------------------------------------
        # 保存配置
        # -------------------------------------------------
        with open(
                os.path.join(path, "config.json"),
                "w",
                encoding="utf-8"
        ) as f:

            json.dump(
                self.get_config_dict(),
                f,
                indent=4,
                ensure_ascii=False
            )

    # =====================================================
    # 加载
    # =====================================================
    def load(self, path: str):

        config_path = os.path.join(
            path,
            "config.json"
        )

        if os.path.exists(config_path):

            with open(
                    config_path,
                    "r",
                    encoding="utf-8"
            ) as f:

                cfg_dict = json.load(f)

            # -------------------------------------------------
            # 安全重建 Config
            # -------------------------------------------------
            def safe_build(cls, d):

                valid_keys = {
                    f.name for f in fields(cls)
                }

                return cls(
                    **{
                        k: v for k, v in d.items()
                        if k in valid_keys
                    }
                )

            self.d_cfg = safe_build(
                DataConfig,
                cfg_dict.get("data", {})
            )

            self.m_cfg = safe_build(
                ModelConfig,
                cfg_dict.get("model", {})
            )

            self.t_cfg = safe_build(
                TrainingConfig,
                cfg_dict.get("training", {})
            )

            self.s_cfg = safe_build(
                SamplingConfig,
                cfg_dict.get("sampling", {})
            )

        # -------------------------------------------------
        # 重建 diffusion
        # -------------------------------------------------
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise
        )

        # -------------------------------------------------
        # 加载权重
        # -------------------------------------------------
        state = torch.load(
            os.path.join(path, "model.pt"),
            map_location="cpu"
        )

        self.diffusion.load_state_dict(state)