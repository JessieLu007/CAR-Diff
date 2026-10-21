# evaluation/run.py
"""python -m evaluation.run --dataset chronic_kidney_disease hepatitis [--fold 0 1 2 3 4]
结果：<ddpm_root>/eval/<ds>_fold<k>/summary.csv 与 detail.csv
每行带 dataset / fold / method / sample / ref 五列：
  ref=train|test : 与真实训练集/测试集比较的保真指标（shap 只在 train 上算）
  ref=both       : 下游指标 dcr、tstr
  method=real    : 每折一行（sample=0, ref=both）；tstr_* 为 TRTR，dcr_mean/dcr_exact 为 holdout→train 基线

只补算部分指标（覆盖这些列，其余列不动，已评估的行也会处理）：
  python -m evaluation.run --only cpt dcr --hp h256_t200_3_final
  --only 可选：shape cpt corr c2st prdc ab shap dcr tstr
"""
from __future__ import annotations
import os
import argparse, re
from itertools import chain
from pathlib import Path
import pandas as pd

from evaluation.metrics import (
    EvalConfig, metric_shape, metric_cpt, metric_corr_diff, metric_c2st, metric_prdc,
    metric_alpha_beta, metric_shap, metric_dcr, metric_tstr, metric_real, holdout_dcr)

from evaluation.paths import load_fold, load_meta, flat_fakes, ddpm_fakes, ddpm_runs
from evaluation.align import make_aligner
from utils import common


BASELINES = ['smote', 'ctgan', 'tvae', 'tabddpm', 'tabkde']
INDEX = ['dataset', 'fold', 'method', 'sample', 'ref']  # ref: train | test | both
_DETAIL = re.compile(r'^(colshape_|cpt_pair_|c2st_fimp_)')

FID = {'shape': metric_shape, 'cpt': metric_cpt, 'corr': metric_corr_diff, 'c2st': metric_c2st,
       'prdc': metric_prdc, 'ab': metric_alpha_beta, 'shap': metric_shap}   # ref=train / test
TRAIN_ONLY = {'shap'}
DOWN = ('dcr', 'tstr')                                                       # ref=both

import time
def _t(f, **kw):
    t = time.time(); r = f(**kw)
    print(f'  {f.__name__}: {time.time()-t:.0f}s', flush=True)
    return r

def _default_tabsyn_root() -> str:
    if env := os.environ.get('TABSYN_ROOT'):
        return env
    return str(Path(__file__).resolve().parents[2] / 'tabsyn-main')

def _default_ddpm_root() -> str:
    if env := os.environ.get("DDPM_ROOT"):
        return env
    return str(Path(__file__).resolve().parents[1])


def evaluate_one(train, test, fake, meta, config, only=None, dcr_real=None) -> dict[str, dict]:
    want = lambda n: only is None or n in only
    kw = dict(num_cols=meta['num_cols'], cat_cols=meta['cat_cols'], config=config)
    out = {}
    for name, ref in (('train', train), ('test', test)):
        row = {}
        for n, f in FID.items():
            if want(n) and not (name == 'test' and n in TRAIN_ONLY):
                row.update(_t(f, ref_df=ref, fake_df=fake, **kw))
        if row:
            out[name] = row
    row = {}
    if want('dcr'):
        row.update(_t(metric_dcr, train_df=train, test_df=test, fake_df=fake, dcr_real=dcr_real, **kw))
    if want('tstr'):
        row.update(_t(metric_tstr, train_df=train, test_df=test, fake_df=fake, **kw))
    if row:
        out['both'] = row
    return out


def _merge(path: Path, row: dict) -> None:
    """把一行并入 csv：同 (dataset,fold,method,sample,ref) 的新值覆盖旧值，旧的其他列保留。"""
    new = pd.DataFrame([row]).set_index(INDEX)
    if path.exists():
        new = new.combine_first(pd.read_csv(path).set_index(INDEX))
    new.reset_index().to_csv(path, index=False)


def _overwrite(path: Path, row: dict) -> None:
    """只覆盖 row 中给出的列（包括新值为 NaN 的情况）；其他列、其他行不动。"""
    new = pd.DataFrame([row]).set_index(INDEX)
    if not path.exists():
        new.reset_index().to_csv(path, index=False)
        return
    old = pd.read_csv(path).set_index(INDEX)
    for c in new.columns:
        if c not in old.columns:
            old[c] = float('nan')
    if new.index[0] not in old.index:
        old = pd.concat([old, new])
    else:
        old.loc[new.index[0], new.columns] = new.iloc[0].values
    old.reset_index().to_csv(path, index=False)


def run_fold(ds, fold, tabsyn_root, ddpm_root, info=None, hp=None, latest=True,
             extra=None, real='test', mode='all', redo=False,
             baselines=(), baseline_root=None, only=None):
    from evaluation import metrics as _m
    _m._REF_CORR.clear()
    meta = load_meta(info or Path(ddpm_root) / 'data' / 'datasets_info.json', ds)
    train, test = load_fold(ddpm_root, ds, fold)
    config = EvalConfig(default_mode=mode, minority_col=meta['label_col'],
                        minority_val=meta['minority_val'])
    nc = dict(num_cols=meta['num_cols'], cat_cols=meta['cat_cols'], config=config)

    out = Path(ddpm_root) / 'eval' / f'{ds}_fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    S, D = out / 'summary.csv', out / 'detail.csv'
    write = _overwrite if only else _merge          # --only：只覆盖指定指标的列
    want = lambda n: only is None or n in only
    done = set()
    if S.exists() and not redo and not only:
        old = pd.read_csv(S)
        done = set(zip(old['method'], old['sample']))

    print(f'[eval] {ds} fold{fold} 找到的 ours 运行:',
          [(h, ts) for ts, h, _ in ddpm_runs(ddpm_root, ds, fold, hp=hp, latest=latest)], flush=True)

    # method=real 行：holdout→train 的 DCR 基线 + TRTR（每折一次）
    dcr_real = None
    if (want('dcr') or want('tstr')) and ('real', 0) not in done:
        print(f'[eval] {ds} fold{fold} real baseline', flush=True)
        dcr_real = holdout_dcr(train, test, **nc)
        key = dict(dataset=ds, fold=fold, method='real', sample=0, ref='both')
        write(S, {**key, **metric_real(train, test, dcr_real=dcr_real, **nc)})

    n_seen = n_skip = 0
    fakes = chain(flat_fakes(tabsyn_root, 'tabsyn', ds, fold),
                  ddpm_fakes(ddpm_root, ds, fold, hp=hp, latest=latest),
                  *(flat_fakes(baseline_root or ddpm_root, m, ds, fold) for m in baselines),
                  *(flat_fakes(r, m, ds, fold) for m, r in (extra or {}).items()))

    align = make_aligner(train, meta)

    for method, sample, path in fakes:
        n_seen += 1
        if (method, sample) in done:
            n_skip += 1
            continue
        print(f'[eval] {ds} fold{fold} {method} s{sample}', flush=True)
        fake = align(pd.read_csv(path))
        if want('dcr') and dcr_real is None:
            dcr_real = holdout_dcr(train, test, **nc)

        for ref, row in evaluate_one(train, test, fake, meta, config, only, dcr_real).items():
            key = dict(dataset=ds, fold=fold, method=method, sample=sample, ref=ref)
            summ = {k: v for k, v in row.items() if not _DETAIL.match(k)}
            det  = {k: v for k, v in row.items() if _DETAIL.match(k)}
            write(S, {**key, **summ})
            if det:
                write(D, {**key, **det})
    print(f'[eval] {ds} fold{fold}: 共 {n_seen} 个合成文件，已评估跳过 {n_skip}，本次评估 {n_seen - n_skip}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=None, help="名称或索引；不传则遍历 common.DATASETS")
    ap.add_argument("--fold", type=int, nargs='*', default=None, help="0-based，可传多个：--fold 0 1 2；不传则 0..n_folds-1")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument('--tabsyn-root', default=_default_tabsyn_root())
    ap.add_argument('--ddpm-root', default=_default_ddpm_root())
    ap.add_argument('--info', default=None)
    ap.add_argument('--extra', action='append', default=[], metavar='METHOD=ROOT')
    ap.add_argument('--hp', default=None)
    ap.add_argument('--all-runs', action='store_true')
    ap.add_argument('--real', choices=['test', 'train'], default='test')
    ap.add_argument('--mode', choices=['all', 'minority'], default='all')
    ap.add_argument('--redo', action='store_true', help='重算已有行（全部指标）')
    ap.add_argument('--only', nargs='+', choices=list(FID) + list(DOWN), default=None,
                    help='只算（并覆盖）这些指标，其余列不动；对已评估的行同样生效')
    ap.add_argument('--baselines', nargs='*', default=BASELINES, choices=BASELINES,
                    help='与 ddpm_root 同目录的对比方法；传空 (--baselines) 则不评估')
    ap.add_argument('--baseline-root', default=None, help='默认等于 --ddpm-root')
    a = ap.parse_args()
    extra = dict(x.split('=', 1) for x in a.extra)

    if a.dataset is None:
        datasets = list(common.DATASETS)
    else:
        datasets = [common.resolve_dataset(d) for d in a.dataset]
    print(f"▶ datasets = {datasets}")

    folds = a.fold if a.fold else list(range(a.n_folds))

    for ds in datasets:
        for k in folds:
            run_fold(ds, k, a.tabsyn_root, a.ddpm_root, a.info, a.hp, not a.all_runs,
                     extra, a.real, a.mode, a.redo,
                     baselines=a.baselines, baseline_root=a.baseline_root, only=a.only)


if __name__ == '__main__':
    main()

# 全量评估（先删除旧 eval/）：python -m evaluation.run --hp h256_t200_3_final
# 只补新指标：python -m evaluation.run --only cpt shape --hp h256_t200_3_finals
