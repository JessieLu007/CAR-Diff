# evaluation/eva_utils.py


def infer_ml_mode_from_task(task: str) -> str:
    """
    根据训练任务名称自动推断评估模式。

    task → ml_mode 映射关系：
        'full_data'  → 'tstr'         模型学整表 P(X,Y)，生成数据直接替代训练集
        'minority_uncond'  → 'augmentation' 只生成少数类，拼接到真实训练集做增强
        'minority_cond'    → 'augmentation' 同上，CFG 引导版本

    :param task: TrainingConfig.task 字段值
    :return: 'augmentation' 或 'tstr'
    """
    if task == 'full_data':
        return 'tstr'
    elif task in ('minority_uncond', 'minority_cond'):
        return 'augmentation'
    raise ValueError(
        f"无法根据任务 '{task}' 推断评估模式，合法值: "
        "'full_data_joint' | 'full_data_cond' | 'minority_uncond' | 'minority_cond'"
    )
