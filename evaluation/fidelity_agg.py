# evaluation/fidelity_agg.py
from pathlib import Path
import pandas as pd

def load_all(ddpm_root, datasets, folds=range(5), kind='summary') -> pd.DataFrame:
    """合并所有 (dataset, fold) 的结果为长表，并从 model_name 解析 method / sample。"""
    dfs = []
    for ds in datasets:
        for k in folds:
            p = Path(ddpm_root) / 'eval' / f'{ds}_fold{k}' / f'fidelity_{kind}.csv'
            if p.exists():
                d = pd.read_csv(p)
                d.insert(0, 'fold', k)
                d.insert(0, 'dataset', ds)
                dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    m = df.model_name.str.extract(r'^(.*?)(?:_s(\d+))?$')
    df.insert(2, 'method', m[0])                       # hp 已含在 method 中
    df.insert(3, 'sample', m[1].fillna(0).astype(int)) # 无后缀 = s0
    return df

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """折内先对重复采样取均值，再对折取 mean/std。"""
    keys = ['dataset', 'method']
    ms = [c for c in df.columns if c not in keys + ['fold', 'sample', 'model_name']]
    fold_lvl = df.groupby(keys + ['fold'])[ms].mean()          # 折内：采样平均
    return fold_lvl.groupby(keys).agg(['mean', 'std'])         # 折间：mean ± std

def sampling_std(df: pd.DataFrame) -> pd.DataFrame:
    """采样稳定性：折内各次采样的 std，再对折取平均。"""
    ms = [c for c in df.columns if c not in ['dataset', 'method', 'fold', 'sample', 'model_name']]
    return (df.groupby(['dataset', 'method', 'fold'])[ms].std()
              .groupby(['dataset', 'method']).mean())

# 用法
# df = load_all(r'e:\projects\ddpm_med', ['chronic_kidney_disease', 'hepatitis'])
# agg = aggregate(df)            # 论文表格：mean ± std（折间）
# stab = sampling_std(df)        # 采样稳定性
# df.to_csv('all_summary_long.csv', index=False)
# # 逐属性明细同理：load_all(..., kind='detail')，再 melt 成 (属性, 值) 画图

# 统计口径：

# 折间 std 反映数据划分带来的方差。
# 采样 std 反映生成随机性带来的方差。
# 方法间做显著性检验（Wilcoxon / Friedman 等）时，应使用折级均值，配对到 (dataset, fold)。
# 不要把 5×J 个采样当作独立样本。