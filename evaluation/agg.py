# evaluation/agg.py
from pathlib import Path
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

INDEX = ['dataset', 'fold', 'method', 'sample', 'ref']


def load_all(ddpm_root, kind='summary') -> pd.DataFrame:
    ps = sorted((Path(ddpm_root) / 'eval').glob(f'*/{kind}.csv'))
    return pd.concat([pd.read_csv(p) for p in ps], ignore_index=True)


def _ms(df):
    return [c for c in df.columns if c not in INDEX and pd.api.types.is_numeric_dtype(df[c])]


def fold_level(df):                    # 采样 -> 折
    return df.groupby(['dataset', 'method', 'ref', 'fold'])[_ms(df)].mean()


def sampling_std(df):                  # 生成随机性
    return (df.groupby(['dataset', 'method', 'ref', 'fold'])[_ms(df)].std()
              .groupby(['dataset', 'method', 'ref']).mean())


def dataset_level(df):                 # 折 -> 数据集：mean / std
    return fold_level(df).groupby(['dataset', 'method', 'ref']).agg(['mean', 'std'])


def train_test_gap(df, metrics):
    """逐折计算 (vs test) − (vs train)，再取折间 mean / std。"""
    F = fold_level(df)[metrics]
    d = F.xs('test', level='ref') - F.xs('train', level='ref')
    return d.groupby(['dataset', 'method']).agg(['mean', 'std'])


def rank_table(df, metric, ref, methods=None, lower_better=True):
    """数据集 × 方法 矩阵、逐数据集排名、平均排名。ref: 'train' | 'test' | 'both'（dcr/cls/tstr 用 both）。"""
    M = (fold_level(df)[metric].xs(ref, level='ref')
         .groupby(['dataset', 'method']).mean().unstack('method'))
    if methods:
        M = M[methods]
    M = M.dropna()
    R = M.rank(axis=1, ascending=lower_better)
    return M, R, R.mean().sort_values()


def friedman(M):
    return friedmanchisquare(*[M[c] for c in M.columns])


def paired_fold_test(df, metric, ref, a, b, dataset):
    F = fold_level(df)[metric].xs(ref, level='ref').unstack('method').loc[dataset]
    return wilcoxon(F[a], F[b])


def mean_std_table(df, ds, fold, metrics, ref):
    """单折初步分析：指标为行、方法为列，格式 mean ± std（5 次采样之间）。ref: train | test | both"""
    d = df[(df.dataset == ds) & (df.fold == fold) & (df.ref == ref)]
    m = d.groupby('method')[metrics].mean()
    s = d.groupby('method')[metrics].std()
    return (m.round(4).astype(str) + ' ± ' + s.round(4).astype(str)).T

# 用法
# df = load_all(r'e:\projects\ddpm_med')
# dataset_level(df).loc[(slice(None), slice(None), 'test'), 'tvd_avg']   # 以 test 为参照
# train_test_gap(df, ['tvd_avg', 'corr_diff_mean'])
# M, R, avg = rank_table(df, 'cls_aug_f1', 'both', lower_better=False)