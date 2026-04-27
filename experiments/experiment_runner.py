# 文件位置: experiment_runner.py
import os
import json
import pandas as pd
from datetime import datetime
from dataclasses import asdict, replace

from configs.experiment_configs import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig, SamplingConfig
from core.denoise_model import TabMLPDenoise
from core.diffusion import MedAGDiffusion

def run_diffusion_pipeline(
    data_source, # 输入可以是字符串路径，也可以是DataFrame
    target_col: str,
    cat_cols: list,
    dataset_name = None, # 显示实验命名
    exp_cfg: ExperimentConfig = None,
    preprocess_fn=None,
    postprocess_fn=None,
    save_dir: str = "results",
    save_to_disk: bool = True # 是否将结果保存到硬盘
):
    """
    高度封装的通用表格扩散实验管线
    :param data_path: 原始 CSV 数据路径
    :param target_col: 目标/标签列名
    :param cat_cols: 类别特征列名列表
    :param custom_configs: 实验超参字典 (用于消融实验)
    :param preprocess_fn: 预处理回调函数 (如：注入医学先验)
    :param postprocess_fn: 后处理回调函数 (如：剥离医学先验)
    """

    # 1. 如果exp_cfg 为None，则使用全默认配置
    if exp_cfg is None:
        # 创建全默认的ExperimentConfig
        exp_cfg = ExperimentConfig(
            data = DataConfig(),
            model=ModelConfig,
            training=TrainingConfig(),
            sampling=SamplingConfig()
        )
    
    data_cfg = exp_cfg.data
    model_cfg = exp_cfg.model
    train_cfg = exp_cfg.training
    sample_cfg = exp_cfg.sampling

    # 2. 数据加载与回调预处理
    if isinstance(data_source, str):
        df = pd.read_csv(data_source)
        if dataset_name is None:
            dataset_name = os.path.basename(data_source).split('.')[0]
    else:
        df = data_source.copy()
        if dataset_name is None:
            dataset_name = 'custom_df'
    if preprocess_fn:
        df = preprocess_fn(df)

    # 动态推断数值列
    num_cols = [c for c in df.columns if c not in cat_cols and c != target_col]

    # 3. 拟合与训练
    diffusion = MedAGDiffusion(model_cfg, TabMLPDenoise)
    diffusion.fit(df, target_col, num_cols, cat_cols, train_cfg, data_cfg)

    # 4. 生成采样 (Conditional 模式需要指定目标类，这里默认取数据集中原本最小的类)
    target_class = diffusion.target_dist_info['minority_class']
    if hasattr(exp_cfg.sampling, 'target_class') and exp_cfg.sampling.target_class is not None:
        target_class = exp_cfg.sampling.target_class
    #if 'target_class' in exp_cfg.get('sample', {}):
        #target_class = exp_cfg['sample']['target_class']

    df_gen = diffusion.generate(sample_cfg, y_cond=target_class)

    # 5. 回调后处理
    if postprocess_fn:
        df_gen = postprocess_fn(df_gen)

    full_config = {
            "data_config": asdict(data_cfg),
            "model_config": asdict(model_cfg),
            "train_config": asdict(train_cfg),
            "sampling_config": asdict(sample_cfg)
        }

    # 6. 保存结果
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    csv_path = f"{save_dir}/gen_{dataset_name}_{timestamp}.csv"
    json_path = f"{save_dir}/gen_{dataset_name}_{timestamp}_config.json"
    
    if save_to_disk:
        
        df_gen.to_csv(csv_path, index=False)

        with open(json_path, 'w') as f:
            json.dump(full_config, f, indent=4)
        
        print(f"\n✅ 实验完成！")
        print(f"💾 数据已保存至: {csv_path}")
        print(f"⚙️ 参数已保存至: {json_path}")
        print("="*60)

        return df_gen, csv_path
    
    else:
        # 五折交叉验证时，不写入硬盘，直接返回数据和完整的配置字典
        return df_gen, full_config