# evaluation/fidelity_run.py
"""简洁调用入口：run_eval / load_results / show，以及终端 CLI。

保存：每次实验两个 csv
    {out}_summary.csv   每模型一行，仅汇总指标（tvd_avg、c2st_auc、cls_* ...）
    {out}_detail.csv    每模型一行，仅逐属性明细（tvd_<col>、c2st_fimp_<col>、kendall_*_<ci>__<cj>）

终端（在项目根目录）：
    python -m evaluation.fidelity_run run  --real te.csv \
        --fake ddpm=ddpm.csv --fake vae=vae.csv --meta meta.json --out results/exp1
    python -m evaluation.fidelity_run show --out results/exp1 [--detail [前缀]]
    python -m evaluation.fidelity_run fold --dataset chronic_kidney_disease --fold 0 [--all-runs] [--real test|train] [--hp h256_t200_3]
meta.json: {"num_cols": [...], "cat_cols": [...], "label_col": "y", "minority_val": 1}
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from evaluation.fidelity_utils import EvalConfig
from evaluation.fidelity_eval import evaluate, print_summary

# 逐属性明细的键：tvd_<col>（不含 tvd_avg）、c2st_fimp_<col>、kendall_{real,fake,diff}_<ci>__<cj>
_DETAIL_RE = re.compile(r'^(tvd_(?!avg$)|c2st_fimp_|kendall_(real|fake|diff)_.+__)')


def split_result(row: dict) -> tuple[dict, dict]:
    summ = {k: v for k, v in row.items() if not _DETAIL_RE.match(k)}
    det = {k: v for k, v in row.items() if k == 'model_name' or _DETAIL_RE.match(k)}
    return summ, det


def _p(out: str, kind: str) -> Path:
    return Path(f'{out}_{kind}.csv')


def _read(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else None


def _save(out: str, summ: pd.DataFrame, det: pd.DataFrame) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    summ.to_csv(_p(out, 'summary'), index=False)
    if det.shape[1] > 1:
        det.to_csv(_p(out, 'detail'), index=False)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def run_eval(
    models:       dict[str, pd.DataFrame],
    real_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    label_col:    str,
    minority_val,
    out:          str,
    metrics:      Optional[list[str]] = None,
    train_df:     Optional[pd.DataFrame] = None,   # 仅 dcr 等需要训练集的指标使用
    pre_filtered: bool = False,
    mode:         str = 'all',
    force:        bool = False,
    verbose:      bool = True,
    **metric_kwargs,
) -> pd.DataFrame:
    """评估 models 中所有生成结果，按 model_name 断点续评，返回汇总表。
    force=True 时重算 models 中已存在的模型。新增指标需 force=True 才会补算。"""
    config = EvalConfig(default_mode=mode, minority_col=label_col,
                        minority_val=minority_val, pre_filtered=pre_filtered)
    summ_old, det_old = _read(_p(out, 'summary')), _read(_p(out, 'detail'))
    if force and summ_old is not None:
        summ_old = summ_old[~summ_old.model_name.isin(models)]
        if det_old is not None:
            det_old = det_old[~det_old.model_name.isin(models)]
    done = set(summ_old.model_name) if summ_old is not None else set()

    S = [summ_old] if summ_old is not None and len(summ_old) else []
    D = [det_old] if det_old is not None and len(det_old) else []
    for name, fake_df in models.items():
        if name in done:
            verbose and print(f'[skip] {name}')
            continue
        verbose and print(f'[eval] {name} ...', flush=True)
        row = evaluate(real_df, fake_df, num_cols, cat_cols, config,
                       train_df=train_df, model_name=name, metrics=metrics, **metric_kwargs)
        s, d = split_result(row)
        S.append(pd.DataFrame([s]))
        D.append(pd.DataFrame([d]))
        _save(out, pd.concat(S, ignore_index=True), pd.concat(D, ignore_index=True))  # 每个模型后落盘
    return pd.concat(S, ignore_index=True) if S else pd.DataFrame()


# ── 读取 / 展示 ───────────────────────────────────────────────────────────────

def load_results(out: str, detail: bool = False) -> pd.DataFrame:
    df = pd.read_csv(_p(out, 'summary'))
    if detail:
        d = _read(_p(out, 'detail'))
        if d is not None:
            df = df.merge(d, on='model_name', how='left')
    return df


def show(out: str, detail: Optional[str] = None, models: Optional[list[str]] = None) -> None:
    """detail=None 只看汇总；'all' 看全部明细；其他字符串按前缀筛选（如 'tvd'、'kendall_diff'）。"""
    df = load_results(out)
    if models:
        df = df[df.model_name.isin(models)]
    print_summary(df)
    if detail:
        d = _read(_p(out, 'detail'))
        if d is None:
            print('(no detail file)')
            return
        d = d[d.model_name.isin(df.model_name)]
        cols = [c for c in d.columns if c == 'model_name' or detail == 'all' or c.startswith(detail)]
        print(f'=== Detail ({detail}) ===')
        print(d[cols].set_index('model_name').T.to_string())   # 属性为行，模型为列


# ── CLI ──────────────────────────────────────────────────────────────────────

def load_df(path: str) -> pd.DataFrame:
    """唯一的读数据入口；后续补目录/命名规则解析时只需替换这里。"""
    p = Path(path)
    return pd.read_csv(p) if p.suffix == '.csv' else pd.read_csv(p)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='fidelity_run')
    sub = ap.add_subparsers(dest='cmd', required=True)

    r = sub.add_parser('run')
    r.add_argument('--real', required=True, help='用于对比的真实数据 csv（通常为验证集）')
    r.add_argument('--train', default=None, help='训练集 csv，dcr 需要')
    r.add_argument('--fake', action='append', required=True, metavar='NAME=PATH')
    r.add_argument('--meta', required=True, help='json: num_cols/cat_cols/label_col/minority_val')
    r.add_argument('--out', required=True, help='结果前缀，如 results/exp1')
    r.add_argument('--metrics', nargs='+', default=None)
    r.add_argument('--mode', choices=['all', 'minority'], default='all')
    r.add_argument('--pre-filtered', action='store_true')
    r.add_argument('--force', action='store_true')

    f = sub.add_parser('fold', help='按目录规则自动收集某数据集某折的所有方法并评估')
    f.add_argument('--dataset', required=True)
    f.add_argument('--fold', type=int, required=True)
    f.add_argument('--info', default=None, help='datasets_info.json，默认 <ddpm-root>/data/datasets_info.json')
    f.add_argument('--real', choices=['test', 'train'], default='test', help='与生成数据对比的真实数据')
    f.add_argument('--tabsyn-root', default=r'e:\projects\tabsyn-main')
    f.add_argument('--ddpm-root', default=r'e:\projects\ddpm_med')
    f.add_argument('--extra', action='append', default=[], metavar='METHOD=ROOT')
    f.add_argument('--hp', default=None, help='如 h256_t200_3；不指定则不过滤超参')
    f.add_argument('--all-runs', action='store_true', help='遍历所有有效时间戳（默认每个超参只取最新）')
    f.add_argument('--out', default=None, help='默认 <ddpm-root>/eval/<ds>_fold<k>/fidelity')
    f.add_argument('--metrics', nargs='+', default=None)
    f.add_argument('--mode', choices=['all', 'minority'], default='all')
    f.add_argument('--force', action='store_true')

    s = sub.add_parser('show')
    s.add_argument('--out', required=True)
    s.add_argument('--detail', nargs='?', const='all', default=None)
    s.add_argument('--models', nargs='+', default=None)

    a = ap.parse_args(argv)
    if a.cmd == 'run':
        meta = json.loads(Path(a.meta).read_text())
        models = {n: load_df(p) for n, p in (x.split('=', 1) for x in a.fake)}
        run_eval(models, load_df(a.real),
                 meta['num_cols'], meta['cat_cols'], meta['label_col'], meta['minority_val'],
                 out=a.out, metrics=a.metrics, pre_filtered=a.pre_filtered,
                 mode=a.mode, force=a.force,
                 train_df=load_df(a.train) if a.train else None)
        show(a.out)
    elif a.cmd == 'fold':
        from evaluation.fidelity_paths import collect_fold, load_meta
        meta = load_meta(a.info or Path(a.ddpm_root) / 'data' / 'datasets_info.json', a.dataset)
        real_df, train_df, models = collect_fold(
            a.dataset, a.fold, a.tabsyn_root, a.ddpm_root, hp=a.hp, latest=not a.all_runs,
            extra=dict(x.split('=', 1) for x in a.extra), real=a.real)
        out = a.out or str(Path(a.ddpm_root) / 'eval' / f'{a.dataset}_fold{a.fold}' / 'fidelity')
        print(f'found {len(models)} fake sets: {list(models)}')
        run_eval(models, real_df, meta['num_cols'], meta['cat_cols'], meta['label_col'],
                 meta['minority_val'], out=out, metrics=a.metrics, mode=a.mode, force=a.force,
                 train_df=train_df)
        show(out)
    else:
        show(a.out, detail=a.detail, models=a.models)


if __name__ == '__main__':
    main()