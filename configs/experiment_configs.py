# configs/experiment_configs.py
from dataclasses import dataclass, replace
from typing import Optional, Literal


@dataclass
class DataConfig:
    """数据集处理配置"""
    num_transform_method: Literal['quantile', 'minmax', 'none'] = 'quantile'
    cat_encoding_method: Literal['label', 'ordinal', 'none'] = 'ordinal'
    inject_physics_prior: bool = False
    random_state: int = 42

@dataclass
class PipelineConfig:
    random_seed: int = 0  
    save_base_dir: str = "results"
    tag: str = "" # 运行标签，写入结果文件夹名，用于区分变体/种子

@dataclass
class ModelConfig:
    """网络架构参数"""
    dim_t: int = 256
    dim_hidden: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    num_timesteps: int = 200

    # 控制模型prediction模式
    # 默认eps,可切换为x0,此时直接预测x0而非噪声epsilon，当前模式下eps效果最好，暂时不考虑x0
    prediction_type: str = 'eps' 
    version: str = "v2" # 新增模型版本 "v1" 或者 "v2"

    share_decoder: bool = True
    cat_ar: bool = False   # True：最终类别解码用轻量自回归头（教师强制训练）
    

    # 损失组件
    use_normal_reg: bool = False
    corr_t_ratio: float = 0.5           # 相关性触发的噪声阶段比率

    # CFG 与后验参数
    null_class_idx: int = 5

    x0_clamp_min: float = -4.0
    x0_clamp_max: float = 4.0


@dataclass
class TrainingConfig:
    """训练数据范围与建模范式控制"""
    batch_size: int = 64
    total_steps: int = None
    target_epochs: int = 200
    use_two_phase: bool = False
    lr: float = 1e-3
    uncond_p: float = 0.1   # CFG 空条件丢弃率（仅 *_cond 范式生效）
    patience: int = 10
    eval_every: int = 100
    last_batch_strategy: str = "drop" # "drop" | "pad"

    step_floor: int = 0
    use_early_stop: bool = True # False = 固定预算，用最终权重

    use_label_smoothing: bool = False # 是否对minority_cond的真实标签做随机平滑
    label_smooth_prob: float = 0.05   # 平滑概率，含义同diff_training.smooth_labels
    enable_amp: bool = False
    enable_compile: bool = True
    enable_nan_check:bool = False

    # ------------------------------------------------------------------
    # task：训练范式，调用方始终传入全量数据，内部自动路由
    #
    #   "full_data"
    #       训练数据：全量（target_col 作为普通分类特征保留）
    #       建模范式：Joint 联合分布 P(X, Y)
    #       典型用途：TSTR、替代数据集生成
    #
    #   "minority_uncond"
    #       训练数据：仅 label=1 子集（内部自动筛选）
    #       建模范式：无条件 P(X | Y=1)
    #       典型用途：少数类简单拟真扩增
    #
    #   "minority_cond"
    #       训练数据：全量（target_col 剥离为条件标签）
    #       建模范式：Conditional P(X | Y)，生成时只采样 Y=1
    #       典型用途：CFG 引导的少数类增强（分类器提升场景）
    # ------------------------------------------------------------------
    task: Literal[
        'full_data',
        'minority_uncond',
        'minority_cond',
    ] = 'full_data'


@dataclass
class SamplingConfig:
    num_samples: Optional[int] = None
    cat_temperature_cap: float = 1.0
    cfg_scale_fidelity: float = 1.0
    repulse_weight_fidelity: float = 0.0


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig
    pipeline: PipelineConfig

    def with_overrides(
        self,
        data: dict | None = None,
        model: dict | None = None,
        training: dict | None = None,
        sampling: dict | None = None,
        pipeline: dict | None = None
    ) -> "ExperimentConfig":
        """基于当前配置，返回一个更改了指定字段的新副本"""
        return ExperimentConfig(
            data=replace(self.data, **(data or {})),
            model=replace(self.model, **(model or {})),
            training=replace(self.training, **(training or {})),
            sampling=replace(self.sampling, **(sampling or {})),
            pipeline=replace(self.pipeline, **(pipeline or {}))
        )

def make_default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        sampling=SamplingConfig(),
        pipeline=PipelineConfig()
    )
