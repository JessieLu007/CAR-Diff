# evaluation/fidelity_paths.py
"""目录解析：按 (dataset, fold) 收集 real 数据与各方法生成结果，输出可直接喂给 run_eval 的 models 字典。

约定
  真实数据（所有方法共用，预处理后的第一手文件）:
      <ddpm_root>/splits/<ds>_seed<split_seed>_full_data_fold<k>_imputed/real_train_imputed.csv 与 real_test_imputed.csv
  平铺型方法（tabsyn、tabddpm ...）: <root>/synthetic/<ds>_fold<k>/<method>.csv          -> 视为 s0
                                       <root>/synthetic/<ds>_fold<k>/<method>_s<j>.csv    -> 第 j 次重复采样
  ddpm_med: <root>/results/<ds>/seed_<seed>_fold_<k>/full_data_fidelity_<hp>_<MMDD_HHMMSS>/synthetic_data.csv    -> s0
            同目录下 synthetic_data_s<j>.csv                                                                      -> 第 j 次重复采样
            <hp> 形如 h256_t200_3；不含 synthetic_data.csv 的文件夹（中断运行）自动忽略
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

PREFIX = 'full_data_fidelity_'
_TS = re.compile(r'(\d{4}_\d{6})$')          # MMDD_HHMMSS


def fold_tag(ds: str, fold: int) -> str:
    return f'{ds}_fold{fold}'


def load_fold(ddpm_root, ds: str, fold: int, split_seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = Path(ddpm_root) / 'splits' / f'{ds}_seed{split_seed}_full_data_fold{fold}_imputed'
    return pd.read_csv(d / 'real_train_imputed.csv'), pd.read_csv(d / 'real_test_imputed.csv')


def flat_fake(root, method: str, ds: str, fold: int) -> dict[str, pd.DataFrame]:
    """tabsyn / tabddpm 等平铺布局；返回 {'tabsyn': df, 'tabsyn_s1': df, ...}"""
    d = Path(root) / 'synthetic' / fold_tag(ds, fold)
    out = {}
    p0 = d / f'{method}.csv'
    if p0.exists():
        out[method] = pd.read_csv(p0)
    for p in sorted(d.glob(f'{method}_s*.csv')):
        out[p.stem] = pd.read_csv(p)
    return out


def ddpm_runs(ddpm_root, ds: str, fold: int, seed: int = 2026,
              hp: Optional[str] = None, prefix: str = PREFIX) -> list[tuple[str, str, Path]]:
    """返回 [(timestamp, hp, run_dir)]，按时间戳升序；只保留含 synthetic_data.csv 的文件夹。
    hp 指定时只保留该超参组合（如 'h256_t200_3'）。"""
    base = Path(ddpm_root) / 'results' / ds / f'seed_{seed}_fold_{fold}'
    runs = []
    for d in base.glob(prefix + '*'):
        m = _TS.search(d.name)
        if not (d.is_dir() and m and (d / 'synthetic_data.csv').exists()):
            continue
        h = d.name[len(prefix):m.start() - 1]
        if hp is None or h == hp:
            runs.append((m.group(1), h, d))
    return sorted(runs, key=lambda r: r[0])   # 注意：时间戳不含年份，跨年时需改用 mtime


def ddpm_fake(ddpm_root, ds: str, fold: int, seed: int = 2026, hp: Optional[str] = None,
              latest: bool = True, name: str = 'ours') -> dict[str, pd.DataFrame]:
    """latest=True：每个超参组合只取最新一次运行；False：取全部运行。
    键名：{name}_{hp}[_{ts}][_s{j}]"""
    runs = ddpm_runs(ddpm_root, ds, fold, seed, hp)
    if latest:
        runs = list({h: (ts, h, d) for ts, h, d in runs}.values())   # 升序，后者覆盖前者
    out = {}
    for ts, h, d in runs:
        key = f'{name}_{h}' + ('' if latest else f'_{ts}')
        out[key] = pd.read_csv(d / 'synthetic_data.csv')
        for p in sorted(d.glob('synthetic_data_s*.csv')):
            out[f"{key}_{p.stem.split('_')[-1]}"] = pd.read_csv(p)     # ..._s1
    return out


def load_meta(info_path, ds: str) -> dict:
    """读 datasets_info.json，返回 run_eval 需要的 num_cols / cat_cols / label_col / minority_val。
    兼容两种结构：顶层按数据集名分组，或文件本身就是单个数据集的条目。
    target 若不在 numerical / categorical 中，则并入 cat_cols（评估代码默认标签列在列集合内）。"""
    info = json.loads(Path(info_path).read_text(encoding='utf-8'))
    if ds in info:
        m = info[ds]
    elif 'target' in info:
        m = info
    else:
        raise KeyError(f"'{ds}' not in {info_path}; keys: {list(info)[:10]}")
    label = m['target']
    num, cat = list(m['numerical']), list(m['categorical'])
    if label not in num and label not in cat:
        cat.append(label)
    return dict(num_cols=num, cat_cols=cat, label_col=label, minority_val=m['minority_class'])


def collect_fold(ds: str, fold: int, tabsyn_root, ddpm_root, hp: Optional[str] = None,
                 latest: bool = True, extra: Optional[dict[str, str]] = None,
                 seed: int = 2026, split_seed: int = 2026, real: str = 'test'):
    """返回 real_df, train_df, models（train_df 仅供 dcr 等需要训练集的指标使用）。real='test'|'train' 决定与生成数据对比的真实数据。
    extra: {'tabddpm': root} 等平铺型方法。
    latest=False 且 hp=None：遍历该划分下 ddpm_med 所有超参、所有时间戳的有效运行。"""
    train, test = load_fold(ddpm_root, ds, fold, split_seed)
    real_df = test if real == 'test' else train
    models = {}
    models.update(flat_fake(tabsyn_root, 'tabsyn', ds, fold))
    models.update(ddpm_fake(ddpm_root, ds, fold, seed, hp, latest))
    for m, root in (extra or {}).items():
        models.update(flat_fake(root, m, ds, fold))
    return real_df, train, models