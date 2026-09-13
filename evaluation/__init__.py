# evaluation/__init__.py
"""
evaluation 包统一入口

快速使用：
    from evaluation import run_full_evaluation_from_config
    stats, ml, cat = run_full_evaluation_from_config(exp_cfg, results, TARGET_COL, CAT_COLS)

模块结构：
    core_evaluator.py  — TabularEvaluator（单折）
    cv_runner.py       — CrossValidationEvaluator（多折聚合）
    tool_stability.py  — StabilityEvaluator（多次生成稳定性）
    utils.py           — 辅助函数（infer_ml_mode_from_task 等）
    tool_compare.py    — 跨实验对比工具
"""
#from evaluation.core_evaluator import TabularEvaluator
#from evaluation.tool_stability import StabilityEvaluator
from evaluation.utils import infer_ml_mode_from_task
#from evaluation.tool_compare import compare_experiments, save_comparison


def run_full_evaluation_from_config(
        exp_cfg,
        fold_results: list,
        target_col: str,
        cat_cols: list,
        run_clf: bool = False,
        run_stability: bool = False,
        **kwargs):
    """
    从训练配置自动推断评估模式，完成多折评估并返回结果。

    :param exp_cfg:       训练配置对象，需含 .training.task 字段
    :param fold_results:  每折数据列表，每项含 df_real_train / df_fake / df_real_test
    :param target_col:    目标列名
    :param cat_cols:      类别特征列名列表
    :param run_clf:       是否运行分类器并打印结果
    :param run_stability: （预留）是否运行稳定性评估
    :param kwargs:        透传至 CrossValidationEvaluator
    :return:              (stats_agg, ml_agg, cat_compare_report)
    """
    '''ml_mode = infer_ml_mode_from_task(exp_cfg.training.task)
    print(f"[评估模式] task={exp_cfg.training.task} → ml_mode={ml_mode}")

    evaluator = CrossValidationEvaluator(
        fold_results=fold_results,
        target_col=target_col,
        cat_cols=cat_cols,
        ml_mode=ml_mode,
        **kwargs
    )

    stats_agg, ml_agg = evaluator.run(run_clf=run_clf)
    return stats_agg, ml_agg, evaluator.cat_compare_report


__all__ = [
    #'TabularEvaluator',
    'CrossValidationEvaluator',
    'StabilityEvaluator',
    'infer_ml_mode_from_task',
    'compare_experiments',
    'save_comparison',
    'run_full_evaluation_from_config',
]'''
