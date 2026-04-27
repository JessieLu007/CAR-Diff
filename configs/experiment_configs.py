# configs/experiment_configs.py
from dataclasses import dataclass, field, replace
from typing import List, Optional, Literal


@dataclass
class DataConfig:
    """数据集处理配置"""
    num_transform_method: Literal['quantile', 'standard'] = 'quantile'
    inject_physics_prior: bool = False
    random_state: int = 42

@dataclass
class ModelConfig:
    """网络架构参数"""
    dim_t: int = 128
    dim_hidden: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    num_timesteps: int = 1000

    # 损失组件 
    use_adaptive_focal: bool = True #是否使用抵抗饥饿的自适应Focal Loss
    # Global Marginal Alignment(GMA)权重
    gma_loss_weight: float = 0.5
    corr_weight: float = 0.0    # 相关性权重损失
    corr_t_ratio: float = 0.2   # 相关性出发的噪声阶段比率

    # CFG与后验参数
    null_class_idx: int = 2
    posterior_switch_ratio: float = 0.3
    cat_temperature_cap: float = 0.6

    x0_clamp_min: float = -5.0
    x0_clamp_max: float = 5.0    
    

@dataclass
class TrainingConfig:
    """训练数据范围与建模范式控制"""
    batch_size: int = 512
    epochs: int = 1000
    lr: float = 1e-4
    uncond_p: float = 0.1   # CFG 空条件丢弃率
    patience: int = 20

    # 训练任务类型
    # full_data: 学整表 P(X, Y)，用于 TSTR / 替代数据
    # minority_uncond: 只用 label=1 学 P(X | Y=1)
    # minority_cond: 用全体样本学 P(X | Y)，生成时只要 Y=1 的 X
    task: Literal['full_data', 'minority_uncond', 'minority_cond'] = 'full_data'

    # 训练/生成的主要目标
    # fidelity: 拟真（分布贴近真实）
    # clf_boost: 帮助分类器（允许牺牲一些拟真）
    objective: Literal['fidelity', 'clf_boost'] = 'fidelity'


@dataclass
class SamplingConfig:
    
    num_samples: Optional[int] = None

    # 下方参数通常由gen_strategy自动覆盖，但也可以在此微调
    cfg_scale_fedility: float = 1.0      #正向引导强度
    cfg_scale_boost: float = 3.0
    temperature_fedility: float = 1.0    #采样温度
    temperature_boost: float = 0.7
    repulse_weight_fedility: float = 0.0 # 负向排斥强度
    repulse_weight_boost: float = 1.0

@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig

    def with_overrides(
            self,
            data: dict | None = None,
            model: dict | None = None,
            training: dict | None = None,
            sampling: dict | None = None,
    ) -> "ExperimentConfig":
        # 基于当前ExperimentConfig，返回一个更改了指定字段的新副本
        new_data = replace(self.data, **(data or {}))
        new_model = replace(self.model, **(model or {}))
        new_training = replace(self.training, **(training or {}))
        new_sampling = replace(self.sampling, **(sampling or {}))

        return ExperimentConfig(
            data = new_data,
            model = new_model,
            training = new_training,
            sampling = new_sampling
        )
    

def make_default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        sampling=SamplingConfig(),
    )