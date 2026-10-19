# # evaluation/gen_baselines.py
# """在当前数据划分上，用各对比方法在 full train（含标签列）上训练一次，重复采样 n 次，按 paths.py 的约定保存。

#     python -m evaluation.gen_baselines --dataset chronic_kidney_disease hepatitis \
#         --fold 0 1 2 3 4 --methods smote ctgan tvae tabddpm tabkde --n-samples 5

# 输出（n=5 即 s0..s4）：
#     <synth-root>/synthetic/<ds>_fold<k>/<method>.csv          s0
#     <synth-root>/synthetic/<ds>_fold<k>/<method>_s<j>.csv     j = 1..n-1

# 说明：
#   * 与旧版 baselines.py 的区别：旧版只在少数类（X_train_pos）上训练；这里是 full_data 评估，
#     在含标签列的完整 train 上训练，每次生成 len(train) 行。
#   * 每个 (数据集, 折, 方法) 只训练一次，之后用 seed + j 采样第 j 次。已存在的文件跳过（--force 覆盖）。
#   * 某个方法报错只打印并跳过，不影响其他方法。依赖在函数内延迟导入，可以只装需要的方法。
#   * SMOTE：每个类别扩到原来的 2 倍并只取新增样本，故共 len(train) 行且类别比例不变。
# """
# from __future__ import annotations
# import shutil
# import torch
# from evaluation.align import make_aligner
# import argparse
# import random
# import time
# import traceback
# from pathlib import Path
# from types import SimpleNamespace

# import numpy as np
# import pandas as pd

# from evaluation.paths import fake_path, save_fake, load_fold, load_meta
# from utils import common


# def _seed_all(s: int) -> None:
#     random.seed(s)
#     np.random.seed(s)
#     try:
#         import torch
#         torch.manual_seed(s)
#     except ImportError:
#         pass


# def realign_existing(root, ds, k, align):
#     """对已保存的文件重新对齐：首次先把原文件备份到 raw/，之后总是从备份对齐（可重复执行）。"""
#     d = Path(root) / 'synthetic' / f'{ds}_fold{k}'
#     raw = Path(root) / 'raw' / 'synthetic' / f'{ds}_fold{k}'
#     raw.mkdir(parents=True, exist_ok=True)
#     for p in sorted(d.glob('*.csv')):
#         if not (raw / p.name).exists():
#             shutil.copy2(p, raw / p.name)
#         align(pd.read_csv(raw / p.name)).to_csv(p, index=False)
#         print(f'[realign] {ds}_fold{k}/{p.name}')


# # ── 各方法：统一签名 (train, test, meta, name, seed, js) -> [DataFrame]，与 js 一一对应 ─────────

# def gen_smote(train, test, meta, name, seed, js, on_each=None):
#     from imblearn.over_sampling import SMOTE, SMOTENC
#     label = meta['label_col']
#     feats = [c for c in meta['num_cols'] + meta['cat_cols'] if c != label]
#     cats = [c for c in meta['cat_cols'] if c != label]
#     X, y = train[feats], train[label]
#     cnt = y.value_counts()
#     strategy = {c: 2 * n for c, n in cnt.items()}
#     k = max(1, min(5, int(cnt.min()) - 1))
#     out = []
#     for j in js:
#         kw = dict(sampling_strategy=strategy, k_neighbors=k, random_state=seed + j)
#         sm = SMOTENC(categorical_features=cats, **kw) if cats else SMOTE(**kw)
#         Xr, yr = sm.fit_resample(X, y)
#         df = Xr.iloc[len(X):].copy()
#         df[label] = yr.iloc[len(X):].to_numpy()
#         out.append(df)
#         if on_each is not None:
#             on_each(j, df)
#     return out


# def gen_ctgan(train, test, meta, name, seed, js, batch_size=500):
#     import torch
#     from ctgan import CTGAN
#     _seed_all(seed)
#     m = CTGAN(batch_size=batch_size, verbose=True,
#               cuda=torch.cuda.is_available())
    
#     print(f'[ctgan] cuda={torch.cuda.is_available()} rows={len(train)}', flush=True)
#     m.fit(train, discrete_columns=meta['cat_cols'])
#     out = []
#     for j in js:
#         _seed_all(seed + j)
#         out.append(m.sample(len(train)))
#     return out


# def _synthcity(plugin, train, seed, js, **kw):
#     from synthcity.plugins import Plugins
#     _seed_all(seed)
#     m = Plugins().get(plugin, **kw)
#     m.fit(train)
#     out = []
#     for j in js:
#         _seed_all(seed + j)
#         try:                                           # 部分版本的 generate 接受 random_state
#             df = m.generate(count=len(train), random_state=seed + j).dataframe()
#         except TypeError:
#             df = m.generate(count=len(train)).dataframe()
#         out.append(df)
#     return out


# def gen_tvae(train, test, meta, name, seed, js):
#     bs = 2000 if len(train) > 20000 else 200
#     return _synthcity('tvae', train, seed, js, n_iter=1000, batch_size=bs)


# def gen_tabddpm_basic(train, test, meta, name, seed, js):
#     return _synthcity('ddpm', train, seed, js)

# def gen_tabddpm(train, test, meta, name, seed, js):
#     const = {c: train[c].iloc[0] for c in train.columns if train[c].nunique() <= 1}
#     tr = train.drop(columns=list(const))
#     frames = _synthcity('ddpm', tr, seed, js)
#     for df in frames:
#         for c, v in const.items():
#             df[c] = v
#     return frames


# def gen_tabkde(train, test, meta, name, seed, js):
#     from tabkde.process_dataset import process_data
#     from tabkde.copula_encoding.main import main as train_main
#     from tabkde.sample import main as sample_main

#     process_data(name, train, test)                    # name = <ds>_fold<k>，每折一份预处理结果

#     def args(n):
#         return SimpleNamespace(
#             dataname=name, device='cuda' if torch.cuda.is_available() else 'cpu', method='TabKDE', mode='train',
#             decorrelation=False, cat_encoding_ordinal='ours', copula_encoder=1,
#             latent_encoding='vae', rank_encoding=1, steps=50, num_samples=n,
#             save_path=Path('synthetic') / name)        # 仅为形式，实现中已不保存

#     _seed_all(seed)
#     train_main(args(len(train)))
#     out = []
#     for j in js:
#         _seed_all(seed + j)
#         out.append(sample_main(args(len(train))))
#     return out


# GEN = {'smote': gen_smote, 'ctgan': gen_ctgan, 'tvae': gen_tvae,
#        'tabddpm': gen_tabddpm, 'tabkde': gen_tabkde}


# # ── 统一的后处理：列顺序、类别取值、整数列，对所有方法一视同仁 

# ROOT = Path(__file__).resolve().parents[1]

# def main() -> None:
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--dataset", default=None, help="名称或索引；不传则遍历 common.DATASETS")
#     ap.add_argument("--fold", type=int, default=None, help="0-based；不传则 0..n_folds-1")
#     ap.add_argument("--n_folds", type=int, default=5)
#     ap.add_argument('--methods', nargs='+', default=list(GEN), choices=list(GEN))
#     ap.add_argument('--n-samples', type=int, default=5, help='每折采样总次数，生成 s0..s(n-1)')
#     ap.add_argument('--seed', type=int, default=common.SEEDS[0], help='基础种子；第 j 次采样使用 seed + j')
#     ap.add_argument('--ddpm-root', default=str(ROOT))
#     ap.add_argument('--synth-root', default=None, help='默认等于 --ddpm-root')
#     ap.add_argument('--info', default=None)
#     ap.add_argument('--realign', action='store_true', help='对已保存的文件就地重新对齐（不训练、不采样）')
#     ap.add_argument('--split-seed', type=int, default=common.SEEDS[0])
#     ap.add_argument('--force', action='store_true')
#     ap.add_argument('--ctgan-bs', type=int, default=500)   # 必须是 10 的倍数（pac=10）
#     a = ap.parse_args()
#     root = a.synth_root or a.ddpm_root

#     datasets = list(common.DATASETS) if a.dataset is None else [common.resolve_dataset(a.dataset)]
#     folds = [a.fold] if a.fold is not None else range(a.n_folds)

#     for ds in datasets:
#         meta = load_meta(a.info or Path(a.ddpm_root) / 'data' / 'datasets_info.json', ds)
#         for k in folds:
#             train, test = load_fold(a.ddpm_root, ds, k, a.split_seed)
#             align = make_aligner(train, meta)
#             if a.realign:
#                 realign_existing(root, ds, k, align)
#                 continue
#             name = f'{ds}_fold{k}'
#             for m in a.methods:
#                 js = [j for j in range(a.n_samples)
#                       if a.force or not fake_path(root, m, ds, k, j).exists()]
#                 if not js:
#                     print(f'[skip] {name} {m}')
#                     continue
#                 print(f'[gen] {name} {m} samples={js}', flush=True)
#                 t0 = time.time()
#                 try:
#                     if m == 'ctgan':
#                         frames = gen_ctgan(train, test, meta, name, a.seed, js, a.ctgan_bs)
#                     elif m == 'smote':
#                         frames = gen_smote(train, test, meta, name, a.seed, js,
#                                         on_each=lambda j, df, m=m: save_fake(align(df), root, m, ds, k, j))
#                     else:
#                         frames = GEN[m](train, test, meta, name, a.seed, js)
#                     # for j, df in zip(js, frames):                        # 先保存原始输出
#                     #     save_fake(df, Path(root) / 'raw', m, ds, k, j)
#                     frames = [align(f) for f in frames]                  # 再对齐
#                 except Exception:
#                     print(f'[fail] {name} {m}'); traceback.print_exc()
#                     continue
                
#                 for j, df in zip(js, frames):
#                     save_fake(df, root, m, ds, k, j)
#                 print(f'[done] {name} {m}  rows={len(frames[0])}  {time.time() - t0:.0f}s')

#                 # 检查各次采样是否真的不同
#                 base = fake_path(root, m, ds, k, 0)
#                 if base.exists():
#                     b = pd.read_csv(base)
#                     for j in range(1, a.n_samples):
#                         p = fake_path(root, m, ds, k, j)
#                         if p.exists() and pd.read_csv(p).equals(b):
#                             print(f'[warn] {p.name} 与 s0 完全相同：随机种子可能未生效')


# if __name__ == '__main__':
#     main()

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
import shutil

import torch
from evaluation.align import make_aligner
import argparse
import random
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import synthcity.plugins.core.models.tabular_ddpm as td_mod

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


def realign_existing(root, ds, k, align):
    """对已保存的文件重新对齐：首次先把原文件备份到 raw/，之后总是从备份对齐（可重复执行）。"""
    d = Path(root) / 'synthetic' / f'{ds}_fold{k}'
    raw = Path(root) / 'raw' / 'synthetic' / f'{ds}_fold{k}'
    raw.mkdir(parents=True, exist_ok=True)
    for p in sorted(d.glob('*.csv')):
        if not (raw / p.name).exists():
            shutil.copy2(p, raw / p.name)
        align(pd.read_csv(raw / p.name)).to_csv(p, index=False)
        print(f'[realign] {ds}_fold{k}/{p.name}')


# ── 各方法：统一签名 (train, test, meta, name, seed, js) -> [DataFrame]，与 js 一一对应 ─────────

def gen_smote(train, test, meta, name, seed, js, on_each=None):
    from imblearn.over_sampling import SMOTE, SMOTENC
    label = meta['label_col']
    feats = [c for c in meta['num_cols'] + meta['cat_cols'] if c != label]
    cats = [c for c in meta['cat_cols'] if c != label]
    X, y = train[feats], train[label]
    cnt = y.value_counts()
    strategy = {c: 2 * n for c, n in cnt.items()}
    k = max(1, min(5, int(cnt.min()) - 1))
    out = []
    for j in js:
        kw = dict(sampling_strategy=strategy, k_neighbors=k, random_state=seed + j)
        sm = SMOTENC(categorical_features=cats, **kw) if cats else SMOTE(**kw)
        Xr, yr = sm.fit_resample(X, y)
        df = Xr.iloc[len(X):].copy()
        df[label] = yr.iloc[len(X):].to_numpy()
        out.append(df)
        if on_each is not None:
            on_each(j, df)
    return out


def gen_ctgan(train, test, meta, name, seed, js, batch_size=500):
    import torch
    from ctgan import CTGAN
    _seed_all(seed)
    m = CTGAN(batch_size=batch_size, verbose=True,
              cuda=torch.cuda.is_available())
    
    print(f'[ctgan] cuda={torch.cuda.is_available()} rows={len(train)}', flush=True)
    m.fit(train, discrete_columns=meta['cat_cols'])
    out = []
    for j in js:
        _seed_all(seed + j)
        out.append(m.sample(len(train)))
    return out


def _synthcity(plugin, train, meta, seed, js, **kw):
    from synthcity.plugins import Plugins
    _seed_all(seed)
    m = Plugins().get(plugin, **kw)


    # import numpy as np, torch
    # from synthcity.plugins.core.models.tabular_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion as G
    # _orig = G.mixed_loss
    # _bad = []
    # def _dbg(self, x, *a, **k):
    #     self._n = getattr(self, '_n', 0) + 1
    #     lm, lg = _orig(self, x, *a, **k)
    #     if not (torch.isfinite(lm) and torch.isfinite(lg)):
    #         xc = x[:, self.num_numerics:]
    #         _bad.append(self._n)
    #         if len(_bad) == 1:
    #             nc = self.num_classes
    #             print(f'[NaN] batch#{self._n} multi={lm.item()} gauss={lg.item()}')
    #             for i, k_ in enumerate(nc):
    #                 col = xc[:, i].long()
    #                 if col.max() >= k_ or col.min() < 0:
    #                     print('越界列', i, 'k=', k_, col.min().item(), col.max().item())
    #             print('num_classes 最大的列:', sorted(enumerate(nc.tolist()), key=lambda t: -t[1])[:5])
    #         print('inf batches so far:', _bad, flush=True)
    #         if len(_bad) >= 10:
    #             raise SystemExit
    #         return lm.new_tensor(0.0, requires_grad=True), lg.detach() * 0   # 仅为诊断：跳过坏 batch
    #     return lm, lg
    # G.mixed_loss = _dbg
    
    #cats = [c for c in meta['cat_cols'] if c != meta['label_col']]
    cats = [c for c in meta['cat_cols'] if c in train.columns]

    if cats is not None:
        orig_fit = m.encoder.fit
        m.encoder.fit = lambda X, discrete_columns=None, **k: orig_fit(X, discrete_columns=list(cats), **k)
        old_dc = td_mod.discrete_columns
        td_mod.discrete_columns = lambda X, max_classes=10, return_counts=False: [
            (c, int(X[c].nunique())) if return_counts else c for c in cats if c in X.columns]
    try:
        # import numpy as np
        # from synthcity.plugins.core.dataloader import GenericDataLoader
        # enc = m.encoder.fit_transform(GenericDataLoader(train).dataframe(), discrete_columns=list(cats))
        # v = enc.values.astype('float64')
        # print('nan/inf:', np.isnan(v).sum(), np.isinf(v).sum())
        # mm = pd.DataFrame({'min': enc.min(), 'max': enc.max(), 'nu': enc.nunique()})
        # print(mm[~enc.columns.isin(cats)])          # 数值列：应在约 [-5.2, 5.2]
        # print(mm[enc.columns.isin(cats)].assign(k=lambda d: d.nu - d['max'] - 1).query('min != 0 or k != 0'))  # 分类列：应为空

        # cm = mm[enc.columns.isin(cats)]
        # print(cm[(cm['min'] != 0) | (cm['nu'] != cm['max'] + 1)])   # 应为空

        m.fit(train)
        out = []
        for j in js:
            _seed_all(seed + j)
            try:                                           # 部分版本的 generate 接受 random_state
                df = m.generate(count=len(train), random_state=seed + j).dataframe()
            except TypeError:
                df = m.generate(count=len(train)).dataframe()
            out.append(df)
    finally:
        if cats is not None:
            td_mod.discrete_columns = old_dc

    return out


def gen_tvae(train, test, meta, name, seed, js):
    bs = 2000 if len(train) > 20000 else 200
    return _synthcity('tvae', train, meta, seed, js, n_iter=1000, batch_size=bs)


def _merge_rare(df, cols, min_count=20):
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(str)                    # 整型 ID 列也统一成字符串，避免混型
        vc = df[c].value_counts()
        rare = vc[vc < min_count].index
        df.loc[df[c].isin(rare), c] = 'other_rare'
    return df

def gen_tabddpm(train, test, meta, name, seed, js):
    #return _synthcity('ddpm', train, meta, seed, js)
    const = {c: train[c].iloc[0] for c in train.columns if train[c].nunique() <= 1}
    tr = train.drop(columns=list(const))
    cats = [c for c in meta['cat_cols'] if c in tr.columns]
    cmap = {}
    for c in cats:
        cat = pd.Categorical(tr[c])
        cmap[c] = cat.categories
        tr[c] = cat.codes
    frames = _synthcity('ddpm', tr, meta, seed, js, lr=1e-5)
    for df in frames:
        for c in cats:
            k = len(cmap[c])
            df[c] = cmap[c][df[c].round().astype(int).clip(0, k - 1).to_numpy()]
        for c, v in const.items():
            df[c] = v
    return frames


def gen_tabkde(train, test, meta, name, seed, js):
    from tabkde.process_dataset import process_data
    from tabkde.copula_encoding.main import main as train_main
    from tabkde.sample import main as sample_main

    process_data(name, train, test)                    # name = <ds>_fold<k>，每折一份预处理结果

    def args(n):
        return SimpleNamespace(
            dataname=name, device='cuda' if torch.cuda.is_available() else 'cpu', method='TabKDE', mode='train',
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


# ── 统一的后处理：列顺序、类别取值、整数列，对所有方法一视同仁 

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="名称或索引；不传则遍历 common.DATASETS")
    ap.add_argument("--fold", type=int, default=None, help="0-based；不传则 0..n_folds-1")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument('--methods', nargs='+', default=list(GEN), choices=list(GEN))
    ap.add_argument('--n-samples', type=int, default=5, help='每折采样总次数，生成 s0..s(n-1)')
    ap.add_argument('--seed', type=int, default=common.SEEDS[0], help='基础种子；第 j 次采样使用 seed + j')
    ap.add_argument('--ddpm-root', default=str(ROOT))
    ap.add_argument('--synth-root', default=None, help='默认等于 --ddpm-root')
    ap.add_argument('--info', default=None)
    ap.add_argument('--realign', action='store_true', help='对已保存的文件就地重新对齐（不训练、不采样）')
    ap.add_argument('--split-seed', type=int, default=common.SEEDS[0])
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--ctgan-bs', type=int, default=500)   # 必须是 10 的倍数（pac=10）
    a = ap.parse_args()
    root = a.synth_root or a.ddpm_root

    datasets = list(common.DATASETS) if a.dataset is None else [common.resolve_dataset(a.dataset)]
    folds = [a.fold] if a.fold is not None else range(a.n_folds)

    for ds in datasets:
        meta = load_meta(a.info or Path(a.ddpm_root) / 'data' / 'datasets_info.json', ds)
        for k in folds:
            train, test = load_fold(a.ddpm_root, ds, k, a.split_seed)
            align = make_aligner(train, meta)
            if a.realign:
                realign_existing(root, ds, k, align)
                continue
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
                    if m == 'ctgan':
                        frames = gen_ctgan(train, test, meta, name, a.seed, js, a.ctgan_bs)
                    elif m == 'smote':
                        frames = gen_smote(train, test, meta, name, a.seed, js,
                                        on_each=lambda j, df, m=m: save_fake(align(df), root, m, ds, k, j))
                    else:
                        frames = GEN[m](train, test, meta, name, a.seed, js)
                    # for j, df in zip(js, frames):                        # 先保存原始输出
                    #     save_fake(df, Path(root) / 'raw', m, ds, k, j)
                    frames = [align(f) for f in frames]                  # 再对齐
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