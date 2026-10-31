# resample_tabsyn.py
"""用已训练好的 TabSyn（或本仓库中同一 CLI 的其他基线）重复采样。

    python resample_tabsyn.py --dataset chronic_kidney_disease hepatitis --fold 0 1 2 3 4 --n 4

输出（与 main.py 默认保存位置一致，s0 已存在则跳过，不会重采）：
    ./synthetic/<ds>_fold<k>/tabsyn.csv          s0（初次采样结果）
    ./synthetic/<ds>_fold<k>/tabsyn_s<j>.csv     j = 1..n

说明：
  * 通过调用本目录的 main.py（--mode sample）采样，不改动其内部代码。
  * 每次采样前设定 python / numpy / torch 随机种子（seed + j），保证可复现且各次不同。
  * 采样后检查各次结果是否与 s0 完全相同；相同说明 main.py 内部固定了种子，需要去掉。
  * 其他基线同理：--method tabddpm / smote / stasy ...（需已训练好）。
"""
from __future__ import annotations
import argparse
import os, json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def _load_datasets() -> list[str]:
    if env := os.environ.get('DDPM_DATASETS'):
        return env.split(',')
    cfg = ROOT.parents[2] / 'data' / 'datasets_info.json'   # <repo>/third_party/tabsyn/tabsyn-main → <repo>
    if not cfg.exists():
        return []
    data = json.loads(cfg.read_text(encoding='utf-8'))
    return [k for k in data if not k.startswith('_')]

DEFAULT_DATASETS = _load_datasets()

# 先设种子，再以 __main__ 方式运行 main.py；第一个参数是种子，取出后其余参数原样交给 main.py
_RUN = (
    "import sys, runpy, random; import numpy as np, torch; "
    "s = int(sys.argv.pop(1)); random.seed(s); np.random.seed(s); torch.manual_seed(s); "
    "sys.argv[0] = 'main.py'; runpy.run_path('main.py', run_name='__main__')"
)


def fake_path(method: str, name: str, sample: int = 0) -> Path:
    """须与 ddpm_med/evaluation/paths.py 的 fake_path 保持一致。name = <ds>_fold<k>"""
    d = ROOT / 'synthetic' / name
    return d / (f'{method}.csv' if sample == 0 else f'{method}_s{sample}.csv')


def sample_once(name: str, method: str, out: Path, seed: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-c', _RUN, str(seed),
           '--dataname', name, '--method', method, '--mode', 'sample', '--save_path', str(out)]
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', nargs='+', default=None, help='一个或多个数据集名；与 --all 二选一')
    ap.add_argument('--all', action='store_true',
                    help=f'使用默认数据集列表：{", ".join(DEFAULT_DATASETS)}')
    ap.add_argument('--fold', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument('--n', type=int, default=4, help='重复采样次数，生成 s1..sn')
    ap.add_argument('--method', default='tabsyn')
    ap.add_argument('--name-fmt', default='{ds}_fold{k}', help='tabsyn 中该折对应的 dataname')
    ap.add_argument('--seed', type=int, default=2026, help='基础种子；第 j 次采样使用 seed + j')
    ap.add_argument('--force', action='store_true', help='覆盖已存在的文件')
    a = ap.parse_args()

    if a.all:
        datasets = DEFAULT_DATASETS
    elif a.dataset:
        datasets = a.dataset
    else:
        ap.error('必须指定 --dataset 或 --all 之一')

    if a.all and a.dataset:
        print('[warn] 同时指定了 --all 和 --dataset，--all 优先生效')

    for ds in datasets:
        for k in a.fold:
            name = a.name_fmt.format(ds=ds, k=k)
            for j in range(a.n + 1):
                out = fake_path(a.method, name, j)
                if out.exists() and not a.force:
                    print(f'[skip] {name}/{out.name}')
                    continue
                print(f'[sample] {name} {a.method} s{j} -> {out}', flush=True)
                sample_once(name, a.method, out, a.seed + j)

            files = [p for p in (fake_path(a.method, name, j) for j in range(a.n + 1)) if p.exists()]
            if len(files) > 1:
                base = pd.read_csv(files[0])
                for p in files[1:]:
                    if pd.read_csv(p).equals(base):
                        print(f'[warn] {p.name} 与 {files[0].name} 完全相同：main.py 可能固定了随机种子')


if __name__ == '__main__':
    main()