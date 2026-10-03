# evaluation/run.py
"""python -m evaluation.run --dataset chronic_kidney_disease hepatitis [--fold 0 1 2 3 4]
结果：<ddpm_root>/eval/<ds>_fold<k>/summary.csv 与 detail.csv
每行带 dataset / fold / method / sample 四列，无需再解析名字。
"""
from __future__ import annotations
import os
import argparse, re
from itertools import chain
from pathlib import Path
import pandas as pd

from evaluation.metrics import (
    EvalConfig, metric_alpha_beta, metric_tvd, metric_corr_diff, metric_c2st, metric_kendall, metric_prdc, metric_dep, 
    metric_crps, metric_shap, metric_dcr, metric_tstr)

from evaluation.paths import load_fold, load_meta, flat_fakes, ddpm_fakes
from evaluation.align import make_aligner


INDEX = ['dataset', 'fold', 'method', 'sample', 'ref']  # ref: train | test | both
_DETAIL = re.compile(r'^(tvd_(?!avg$)|c2st_fimp_|kendall_(real|fake|diff)_.+__)')

def _default_tabsyn_root() -> str:
    if env := os.environ.get('TABSYN_ROOT'):
        return env
    return str(Path(__file__).resolve().parents[2] / 'tabsyn-main')

def _default_ddpm_root() -> str:
    if env := os.environ.get("DDPM_ROOT"):
        return env
    return str(Path(__file__).resolve().parents[1])

def vs_ref(ref, fake, meta, config) -> dict:
    kw = dict(ref_df=ref, fake_df=fake, num_cols=meta['num_cols'],
              cat_cols=meta['cat_cols'], config=config)
    row = {}
    row.update(metric_tvd(**kw))
    row.update(metric_corr_diff(**kw))
    row.update(metric_c2st(**kw))
    row.update(metric_kendall(**kw))
    row.update(metric_crps(**kw))
    row.update(metric_shap(**kw))
    row.update(metric_prdc(**kw))
    row.update(metric_alpha_beta(**kw))
    row.update(metric_dep(**kw))
    return row


def downstream(train, test, fake, meta, config) -> dict:
    kw = dict(train_df=train, test_df=test, fake_df=fake, num_cols=meta['num_cols'],
              cat_cols=meta['cat_cols'], config=config)
    row = {}
    row.update(metric_dcr(**kw))
    row.update(metric_tstr(**kw))
    return row


def evaluate_one(train, test, fake, meta, config) -> dict[str, dict]:
    return {'train': vs_ref(train, fake, meta, config),
            'test':  vs_ref(test,  fake, meta, config),
            'both':  downstream(train, test, fake, meta, config)}


def _merge(path: Path, row: dict) -> None:
    """把一行并入 csv：同 (dataset,fold,method,sample) 的新值覆盖旧值，旧的其他列保留。"""
    new = pd.DataFrame([row]).set_index(INDEX)
    if path.exists():
        new = new.combine_first(pd.read_csv(path).set_index(INDEX))
    new.reset_index().to_csv(path, index=False)


def run_fold(ds, fold, tabsyn_root, ddpm_root, info=None, hp=None, latest=True,
             extra=None, real='test', mode='all', redo=False):
    meta = load_meta(info or Path(ddpm_root) / 'data' / 'datasets_info.json', ds)
    train, test = load_fold(ddpm_root, ds, fold)
    #real_df = test if real == 'test' else train
    config = EvalConfig(default_mode=mode, minority_col=meta['label_col'],
                        minority_val=meta['minority_val'])

    out = Path(ddpm_root) / 'eval' / f'{ds}_fold{fold}'
    out.mkdir(parents=True, exist_ok=True)
    S, D = out / 'summary.csv', out / 'detail.csv'
    done = set()
    if S.exists() and not redo:
        old = pd.read_csv(S)
        done = set(zip(old['method'], old['sample']))

    fakes = chain(flat_fakes(tabsyn_root, 'tabsyn', ds, fold),
                  ddpm_fakes(ddpm_root, ds, fold, hp=hp, latest=latest),
                  *(flat_fakes(r, m, ds, fold) for m, r in (extra or {}).items()))

    align = make_aligner(train, meta)

    for method, sample, path in fakes:
        if (method, sample) in done:
            continue
        print(f'[eval] {ds} fold{fold} {method} s{sample}', flush=True)
        fake = align(pd.read_csv(path))
        for ref, row in evaluate_one(train, test, fake, meta, config).items():
            key = dict(dataset=ds, fold=fold, method=method, sample=sample, ref=ref)
            summ = {k: v for k, v in row.items() if not _DETAIL.match(k)}
            det  = {k: v for k, v in row.items() if _DETAIL.match(k)}
            _merge(S, {**key, **summ})
            if det:
                _merge(D, {**key, **det})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+', required=True)
    ap.add_argument('--fold', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--tabsyn-root', default=_default_tabsyn_root())
    ap.add_argument('--ddpm-root', default=_default_ddpm_root())
    ap.add_argument('--info', default=None)
    ap.add_argument('--extra', action='append', default=[], metavar='METHOD=ROOT')
    ap.add_argument('--hp', default=None)
    ap.add_argument('--all-runs', action='store_true')
    ap.add_argument('--real', choices=['test', 'train'], default='test')
    ap.add_argument('--mode', choices=['all', 'minority'], default='all')
    ap.add_argument('--redo', action='store_true', help='重算已有行（可配合注释掉不需要的指标，仅补算新指标）')
    a = ap.parse_args()
    extra = dict(x.split('=', 1) for x in a.extra)
    for ds in a.dataset:
        for k in a.fold:
            run_fold(ds, k, a.tabsyn_root, a.ddpm_root, a.info, a.hp, not a.all_runs,
                     extra, a.real, a.mode, a.redo)


if __name__ == '__main__':
    main()

# 用法 powershell
# python -m evaluation.fidelity_run --dataset chronic_kidney_disease hepatitis

# 补算新指标：
# 在 evaluate_one 里只保留新指标那几行，加 --redo 运行。
# _merge 会把新列并进旧行，原有指标不丢。跑完再把注释恢复。