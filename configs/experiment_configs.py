# configs/experiment_configs.py
from dataclasses import dataclass, field, replace
from typing import List, Optional, Literal


@dataclass
class DataConfig:
    """数据集处理配置"""
    num_transform_method: Literal['quantile', 'minmax', 'none'] = 'quantile'
    cat_encoding_method: Literal['label', 'ordinal', 'none'] = 'ordinal'
    inject_physics_prior: bool = False
    random_state: int = 42

@dataclass
class PipelineConfig:
    # 可以设置为[0,1,2,3,4]，已保存这几个的补全缺失值数据集用于对比训练
    random_seed: int = 0  
    save_base_dir: str = "results"

@dataclass
class ModelConfig:
    """网络架构参数"""
    dim_t: int = 256
    dim_hidden: int = 512
    num_layers: int = 3
    dropout: float = 0.1
    num_timesteps: int = 1000

    # 控制模型prediction模式
    # 默认eps,可切换为x0,此时直接预测x0而非噪声epsilon，当前模式下eps效果最好，暂时不考虑x0
    prediction_type: str = 'eps' 
    version: str = "v2" # 新增模型版本 "v1" 或者 "v2"

    # 损失组件
    focal_mode: str = "adaptive"  # "off" | "fixed" | "adaptive"
    use_marginal_align: bool = True 
    use_normal_reg: bool = False
    logit_adj_tau: float = 0.3          # LA loss先验修正强度
    gamma_base: float = 1.0             # focal loss中的gamma_base设置
    focal_alpha_scale: float = 1.0
    gma_loss_weight: float = 0.1        # Global Marginal Alignment(GMA) 权重
    corr_weight: float = 0.0            # 相关性权重损失
    corr_t_ratio: float = 0.2           # 相关性触发的噪声阶段比率

    # CFG 与后验参数
    null_class_idx: int = 2
    posterior_switch_ratio: float = 0.3
    cat_temperature_cap: float = 0.6

    x0_clamp_min: float = -5.0
    x0_clamp_max: float = 5.0


@dataclass
class TrainingConfig:
    """训练数据范围与建模范式控制"""
    batch_size: int = 256
    epochs: int = 1000
    use_two_phase: bool = False
    lr: float = 1e-4
    uncond_p: float = 0.1   # CFG 空条件丢弃率（仅 *_cond 范式生效）
    patience: int = 20

    use_label_smoothing: bool = False # 是否对minority_cond的真实标签做随机平滑
    label_smooth_prob: float = 0.05   # 平滑概率，含义同diff_training.smooth_labels

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

    # 训练/生成的主要目标
    # fidelity  : 拟真（分布贴近真实）
    # clf_boost : 帮助分类器（允许牺牲一些拟真）
    objective: Literal['fidelity', 'clf_boost'] = 'fidelity'


@dataclass
class SamplingConfig:
    num_samples: Optional[int] = None

    # 下方参数通常由 gen_strategy 自动覆盖，也可手动微调
    cfg_scale_fidelity: float = 1.0
    cfg_scale_boost: float = 3.0
    temperature_fidelity: float = 1.0
    temperature_boost: float = 0.7
    repulse_weight_fidelity: float = 0.0
    repulse_weight_boost: float = 1.0


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
