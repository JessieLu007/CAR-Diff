# evaluation/eva_compare.py
"""跨实验对比工具：合并多个实验目录的 metrics_summary.csv 生成透视表"""

import os
import pandas as pd


def compare_experiments(exp_dirs: list, labels: list = None) -> pd.DataFrame:
    """
    合并多个实验目录下的 metrics_summary.csv，生成横向对比透视表。

    :param exp_dirs: 实验目录路径列表
    :param labels:   实验标签列表（与 exp_dirs 等长），默认使用目录名
    :return:         MultiIndex DataFrame（Model × Metric 为行，Exp 为列）
    """
    if labels is None:
        labels = [os.path.basename(d.rstrip('/')) for d in exp_dirs]

    assert len(exp_dirs) == len(labels), "exp_dirs 与 labels 长度必须一致"

    frames = []
    for exp_dir, label in zip(exp_dirs, labels):
        csv_path = os.path.join(exp_dir, "metrics_summary.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️  找不到 {csv_path}，跳过")
            continue
        df = pd.read_csv(csv_path)
        df['Experiment'] = label
        frames.append(df)

    if not frames:
        raise FileNotFoundError("没有找到任何有效的 metrics_summary.csv")

    combined = pd.concat(frames, ignore_index=True)

    # 生成 "Mean ± Std" 格式列
    for col in ['Base', 'Aug', 'Delta']:
        combined[col] = combined.apply(
            lambda r: f"{r[f'{col}_Mean']:.4f} ± {r[f'{col}_Std']:.4f}", axis=1
        )

    pivot = combined.pivot_table(
        index=['Model', 'Metric'],
        columns='Experiment',
        values=['Base', 'Aug', 'Delta'],
        aggfunc='first'
    )
    pivot.columns = ['_'.join(c).strip() for c in pivot.columns]
    return pivot


def save_comparison(pivot: pd.DataFrame, output_path: str = "experiment_comparison.csv"):
    pivot.to_csv(output_path)
    print(f"✅ 对比表已保存至: {output_path}")