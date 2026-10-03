# evaluation/gen_baselines.py
"""在当前数据划分上，用各对比方法在 full train（含标签列）上训练一次，重复采样 n 次，按 paths.py 的约定保存。

    python -m evaluation.gen_baselines --dataset chronic_kidney_disease hepatitis \
        --fold 0 1 2 3 4 --methods smote ctgan tvae tabddpm tabkde --n-samples 5

输出（n=5 即 s0..s4）：
    <synth-root>/synthetic/<ds>_fold<k>/<method>.csv          s0
    <synth-root>/synthetic/<ds>_fold<k>/<method>_s<j>.csv     j = 1..n-1

说明：
  * 与旧版 baselines.py 的区别：旧版只在少数类（X_train_pos）上训练；这里是 full_data 评估，
    在含标签列的完整 train 上训练，每次生成 len(train) 行。
  * 每个 (数据集, 折, 方法) 只训练一次，之后用 seed + j 采样第 j 次。已存在的文件跳过（--force 覆盖）。
  * 某个方法报错只打印并跳过，不影响其他方法。依赖在函数内延迟导入，可以只装需要的方法。
  * SMOTE：每个类别扩到原来的 2 倍并只取新增样本，故共 len(train) 行且类别比例不变。
"""
from __future__ import annotations
import argparse
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation.paths import fake_path, save_fake, load_fold, load_meta
from utils import common


def _seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except ImportError:
        pass


# ── 各方法：统一签名 (train, test, meta, name, seed, js) -> [DataFrame]，与 js 一一对应 ─────────

def gen_smote(train, test, meta, name, seed, js):
    from imblearn.over_sampling import SMOTE, SMOTENC
    label = meta['label_col']
    feats = [c for c in meta['num_cols'] + meta['cat_cols'] if c != label]
    cats = [c for c in meta['cat_cols'] if c != label]
    X, y = train[feats], train[label]
    cnt = y.value_counts()
    strategy = {c: 2 * n for c, n in cnt.items()}
    k = max(1, min(5, int(cnt.min()) - 1))            # 类内样本数需大于 k_neighbors
    out = []
    for j in js:
        kw = dict(sampling_strategy=strategy, k_neighbors=k, random_state=seed + j)
        sm = SMOTENC(categorical_features=cats, **kw) if cats else SMOTE(**kw)
        Xr, yr = sm.fit_resample(X, y)
        df = Xr.iloc[len(X):].copy()                   # imblearn 把新增样本追加在原样本之后
        df[label] = yr.iloc[len(X):].to_numpy()
        out.append(df)
    return out


def gen_ctgan(train, test, meta, name, seed, js):
    from ctgan import CTGAN
    _seed_all(seed)
    m = CTGAN()
    m.fit(train, discrete_columns=meta['cat_cols'])
    out = []
    for j in js:
        _seed_all(seed + j)
        out.append(m.sample(len(train)))
    return out


def _synthcity(plugin, train, seed, js, **kw):
    from synthcity.plugins import Plugins
    _seed_all(seed)
    m = Plugins().get(plugin, **kw)
    m.fit(train)
    out = []
    for j in js:
        _seed_all(seed + j)
        try:                                           # 部分版本的 generate 接受 random_state
            df = m.generate(count=len(train), random_state=seed + j).dataframe()
        except TypeError:
            df = m.generate(count=len(train)).dataframe()
        out.append(df)
    return out


def gen_tvae(train, test, meta, name, seed, js):
    return _synthcity('tvae', train, seed, js, n_iter=1000)


def gen_tabddpm(train, test, meta, name, seed, js):
    return _synthcity('ddpm', train, seed, js)


def gen_tabkde(train, test, meta, name, seed, js):
    from tabkde.process_dataset import process_data
    from tabkde.copula_encoding.main import main as train_main
    from tabkde.sample import main as sample_main

    process_data(name, train, test)                    # name = <ds>_fold<k>，每折一份预处理结果

    def args(n):
        return SimpleNamespace(
            dataname=name, device='cpu', method='TabKDE', mode='train',
            decorrelation=False, cat_encoding_ordinal='ours', copula_encoder=1,
            latent_encoding='vae', rank_encoding=1, steps=50, num_samples=n,
            save_path=Path('synthetic') / name)        # 仅为形式，实现中已不保存

    _seed_all(seed)
    train_main(args(len(train)))
    out = []
    for j in js:
        _seed_all(seed + j)
        out.append(sample_main(args(len(train))))
    return out


GEN = {'smote': gen_smote, 'ctgan': gen_ctgan, 'tvae': gen_tvae,
       'tabddpm': gen_tabddpm, 'tabkde': gen_tabkde}


# ── 统一的后处理：列顺序、类别取值、整数列，对所有方法一视同仁 ─────────────────────────────

# def _canon(fake: pd.DataFrame, train: pd.DataFrame, meta: dict) -> pd.DataFrame:
#     fake = fake[list(train.columns)].copy().reset_index(drop=True)   # 缺列/改名会在此报 KeyError
#     for c in meta['cat_cols']:                                       # 数值型类别列：吸附到真实取值
#         if pd.api.types.is_numeric_dtype(train[c]):
#             vals = np.sort(train[c].unique()).astype(float)
#             v = fake[c].to_numpy(dtype=float)
#             fake[c] = vals[np.abs(v[:, None] - vals[None, :]).argmin(1)]
#     for c in train.columns:                                          # 真实数据为整数的列：取整并还原 dtype
#         if pd.api.types.is_integer_dtype(train[c]):
#             fake[c] = fake[c].round().astype(train[c].dtype)
#     return fake


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="名称或索引；不传则遍历 common.DATASETS")
    ap.add_argument("--fold", type=int, default=None, help="0-based；不传则 0..n_folds-1")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument('--methods', nargs='+', default=list(GEN), choices=list(GEN))
    ap.add_argument('--n-samples', type=int, default=5, help='每折采样总次数，生成 s0..s(n-1)')
    ap.add_argument('--seed', type=int, default=common.SEEDS[0], help='基础种子；第 j 次采样使用 seed + j')
    ap.add_argument('--ddpm-root', default=r'e:\projects\ddpm_med')
    ap.add_argument('--synth-root', default=None, help='默认等于 --ddpm-root')
    ap.add_argument('--info', default=None)
    ap.add_argument('--split-seed', type=int, default=common.SEEDS[0])
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    root = a.synth_root or a.ddpm_root

    datasets = list(common.DATASETS) if a.dataset is None else [common.resolve_dataset(a.dataset)]
    folds = [a.fold] if a.fold is not None else range(a.n_folds)

    for ds in datasets:
        meta = load_meta(a.info or Path(a.ddpm_root) / 'data' / 'datasets_info.json', ds)
        for k in folds:
            train, test = load_fold(a.ddpm_root, ds, k, a.split_seed)
            name = f'{ds}_fold{k}'
            for m in a.methods:
                js = [j for j in range(a.n_samples)
                      if a.force or not fake_path(root, m, ds, k, j).exists()]
                if not js:
                    print(f'[skip] {name} {m}')
                    continue
                print(f'[gen] {name} {m} samples={js}', flush=True)
                t0 = time.time()
                try:
                    frames = GEN[m](train, test, meta, name, a.seed, js)
                    # frames = [_canon(f, train, meta) for f in frames]
                except Exception:
                    print(f'[fail] {name} {m}'); traceback.print_exc()
                    continue
                for j, df in zip(js, frames):
                    save_fake(df, root, m, ds, k, j)
                print(f'[done] {name} {m}  rows={len(frames[0])}  {time.time() - t0:.0f}s')

                # 检查各次采样是否真的不同
                base = fake_path(root, m, ds, k, 0)
                if base.exists():
                    b = pd.read_csv(base)
                    for j in range(1, a.n_samples):
                        p = fake_path(root, m, ds, k, j)
                        if p.exists() and pd.read_csv(p).equals(b):
                            print(f'[warn] {p.name} 与 s0 完全相同：随机种子可能未生效')


if __name__ == '__main__':
    main()