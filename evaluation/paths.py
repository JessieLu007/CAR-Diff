# evaluation/paths.py
"""按 (dataset, fold) 找文件。产出 (method, sample, csv路径)，sample 为整数，无后缀 = 0。
平铺型: <root>/synthetic/<ds>_fold<k>/<method>.csv, <method>_s<j>.csv
ddpm:   <root>/results/<ds>/seed_<seed>_fold_<k>/full_data_fidelity_<hp>_<MMDD_HHMMSS>/synthetic_data[_s<j>].csv
"""
from __future__ import annotations
import json, re
from pathlib import Path
import pandas as pd
from utils.common import SEEDS

PREFIX = 'full_data_fidelity_'
_TS = re.compile(r'(\d{4}_\d{6})$')
SEED = SEEDS[0]


def load_fold(ddpm_root, ds, fold, split_seed=SEED):
    d = Path(ddpm_root) / 'splits' / f'{ds}_seed{split_seed}_full_data_fold{fold}_imputed'
    return pd.read_csv(d / 'real_train_imputed.csv'), pd.read_csv(d / 'real_test_imputed.csv')


def fake_path(root, method, ds, fold, sample=0) -> Path:
    """平铺型方法的文件路径；sample=0 无后缀。与 flat_fakes 的读取规则一致。"""
    d = Path(root) / 'synthetic' / f'{ds}_fold{fold}'
    return d / (f'{method}.csv' if sample == 0 else f'{method}_s{sample}.csv')


def save_fake(df, root, method, ds, fold, sample=0):
    p = fake_path(root, method, ds, fold, sample)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def _sample_id(p: Path) -> int:
    m = re.search(r'_s(\d+)$', p.stem)
    return int(m.group(1)) if m else 0


def flat_fakes(root, method, ds, fold):
    d = Path(root) / 'synthetic' / f'{ds}_fold{fold}'
    for p in sorted(d.glob(f'{method}*.csv')):
        if re.fullmatch(rf'{re.escape(method)}(_s\d+)?', p.stem):
            yield method, _sample_id(p), p


def ddpm_runs(ddpm_root, ds, fold, seed=SEED, hp=None, latest=True,
              require=('synthetic_data.csv',)):
    """返回 [(ts, hp, 运行目录)]；目录内须含 require 中的全部文件。latest=True 时每个 hp 只取最新。"""
    base = Path(ddpm_root) / 'results' / ds / f'seed_{seed}_fold_{fold}'
    runs = []
    for d in base.glob(PREFIX + '*'):
        m = _TS.search(d.name)
        if not (m and all((d / f).exists() for f in require)):
            continue
        h = d.name[len(PREFIX):m.start() - 1]
        if hp is None or h == hp:
            runs.append((m.group(1), h, d))
    runs.sort()                                   # 时间戳不含年份，跨年需改用 mtime
    if latest:
        runs = list({h: (ts, h, d) for ts, h, d in runs}.values())
    return runs


def ddpm_fakes(ddpm_root, ds, fold, seed=SEED, hp=None, latest=True, name='ours'):
    for ts, h, d in ddpm_runs(ddpm_root, ds, fold, seed, hp, latest):
        method = f'{name}_{h}' + ('' if latest else f'_{ts}')
        yield method, 0, d / 'synthetic_data.csv'
        for p in sorted(d.glob('synthetic_data_s*.csv')):
            yield method, _sample_id(p), p


def load_meta(info_path, ds):
    info = json.loads(Path(info_path).read_text(encoding='utf-8'))
    m = info[ds] if ds in info else info
    label = m['target']
    num, cat = list(m['numerical']), list(m['categorical'])
    if label not in num and label not in cat:
        cat.append(label)
    return dict(num_cols=num, cat_cols=cat, label_col=label, minority_val=m['minority_class'])