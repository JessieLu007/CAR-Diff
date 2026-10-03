"""python -m evaluation.make_tabkde_info
由 datasets_idx.json 生成 data/Info/{ds}_fold{k}.json（数据集 × 折）。
JSON 中的 data_path / test_path 为相对 --root 的路径，使用时需以 --root 为工作目录。"""
import sys
import json
import argparse
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]      # evaluation/ 的上一级，即 ddpm_med 根目录
sys.path.insert(0, str(ROOT))
from utils.common import SEEDS

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=str(ROOT))
ap.add_argument('--idx', default=None)
ap.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
ap.add_argument('--ext', default='csv')
a = ap.parse_args()

root = Path(a.root).resolve()
idx = json.load(open(a.idx or root / 'data' / 'datasets_idx.json', encoding='utf-8'))
out_dir = root / 'data' / 'Info'
out_dir.mkdir(parents=True, exist_ok=True)

for ds, ix in idx.items():
    for k in a.folds:
        d = Path('splits') / f'{ds}_seed{SEEDS[0]}_full_data_fold{k}_imputed'   # 相对路径
        tr, te = d / f'real_train_imputed.{a.ext}', d / f'real_test_imputed.{a.ext}'
        task = 'binclass'
        if (root / tr).exists():
            n = pd.read_csv(root / tr).iloc[:, ix['target_col_idx'][0]].nunique()
            task = 'binclass' if n == 2 else 'multiclass'
        cat_idx = ix['cat_col_idx'] + ix['target_col_idx'] # 标签列作为最优一个类别列
        info = {
            'name': ds,
            'task_type': task,
            'header': 0,
            'column_names': None,
            'num_col_idx': ix['num_col_idx'],
            'cat_col_idx': cat_idx,
            'target_col_idx': ix['target_col_idx'],
            'file_type': 'csv',
            'data_path': tr.as_posix(),
            'test_path': te.as_posix(),
        }
        with open(out_dir / f'{ds}_fold{k}.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, ensure_ascii=False)
        print('[ok]', f'{ds}_fold{k}.json', task)