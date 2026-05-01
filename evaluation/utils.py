# evaluation/eva_utils.py


def infer_ml_mode_from_task(task: str) -> str:
    """
    根据训练任务名称自动推断评估模式。
    :param task: 训练配置中的 task 字段
    :return: 'augmentation' 或 'tstr'
    """
    if task == 'full_data':
        return 'tstr'
    elif task in ('minority_uncond', 'minority_cond'):
        return 'augmentation'
    raise ValueError(f"无法根据任务'{task}'推断评估模式，请手动指定 ml_mode. ")