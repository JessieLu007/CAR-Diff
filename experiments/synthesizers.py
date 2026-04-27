# 文件位置: experiments/synthesizers.py
import pandas as pd
from dataclasses import asdict

# 导入流水线基类
from experiments.base_synthesizer import BaseSynthesizer

# 导入算法底层组件
from configs.experiment_configs import DataConfig, ModelConfig, TrainingConfig, SamplingConfig, ExperimentConfig
from core.denoise_model import TabMLPDenoise
from core.diffusion import MedAGDiffusion


class OurDiffusionSynthesizer(BaseSynthesizer):
    """
    为 MedAGDiffusion 定制的包裹器：
      - 适配 BaseSynthesizer 接口
      - 内部与 MedAGDiffusion/TabMLPDenoise 高度耦合
    """
    def __init__(self, target_col: str, cat_cols: list[str], exp_cfg: ExperimentConfig):
        super().__init__(
            target_col, 
            cat_cols, 
            training_cfg=exp_cfg.training,
            sampling_cfg = exp_cfg.sampling
            )

        self.d_cfg = exp_cfg.data
        self.m_cfg = exp_cfg.model
        self.t_cfg = exp_cfg.training
        self.s_cfg = exp_cfg.sampling

        self.diffusion: MedAGDiffusion | None = None

    def _fit_algorithm(self, df_train: pd.DataFrame):
        """实现具体算法的训练接口"""
        # 自动推断数值特征列
        num_cols = [c for c in df_train.columns
                    if c not in self.cat_cols and c != self.target_col]

        # 实例化扩散模型
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise
        )

        # 调用扩散模型的 fit
        self.diffusion.fit(
            df=df_train,
            target_col=self.target_col,
            num_cols=num_cols,
            cat_cols=self.cat_cols,
            train_config=self.t_cfg,
            data_config=self.d_cfg,
        )

    def _generate_algorithm(self, n_samples: int, target_class: int) -> pd.DataFrame:
        """实现具体算法的生成接口"""
        assert self.diffusion is not None, "请先调用 fit() 训练模型"

        self.s_cfg.num_samples = n_samples # 把数量写进SamplingConfig

        # TSTR / joint / conditional 场景下如何使用 target_class（label 值）：
        # - joint: 内部可能忽略 y_cond，或仅控制 CFG 分支
        # - conditional: 作为 y_cond 传入
        df_gen = self.diffusion.generate(
            sampling_config=self.s_cfg,
            y_cond=target_class
        )
        return df_gen

    def get_config_dict(self) -> dict:
        """返回完整实验配置，用于保存和复盘"""
        return {
            "algorithm": "MedAGDiffusion",
            "data_config": asdict(self.d_cfg),
            "model_config": asdict(self.m_cfg),
            "train_config": asdict(self.t_cfg),
            "sampling_config": asdict(self.s_cfg),
        }

    # 可选：给 save/load 一个简单实现
    def save(self, path: str):
        import torch, os, json
        assert self.diffusion is not None, "模型尚未训练，无法保存"

        os.makedirs(path, exist_ok=True)
        # 1) 保存权重
        torch.save(self.diffusion.state_dict(), os.path.join(path, "model.pt"))
        # 2) 保存配置
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(self.get_config_dict(), f, indent=2)

    def load(self, path: str):
        import torch, os, json
        # 1) 读配置（如果你想从 checkpoint 恢复配置，可以在这里解析）
        cfg_path = os.path.join(path, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg_dict = json.load(f)
            # TODO: 你可以在这里把 cfg_dict 填回 self.d_cfg / m_cfg / t_cfg / s_cfg

        # 2) 重建 diffusion + 加载权重
        self.diffusion = MedAGDiffusion(
            model_config=self.m_cfg,
            denoise_class=TabMLPDenoise
        )
        state = torch.load(os.path.join(path, "model.pt"), map_location="cpu")
        self.diffusion.load_state_dict(state)