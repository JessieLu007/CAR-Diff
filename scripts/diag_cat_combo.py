# 放在项目根目录运行：python diag_cat_combo.py
import numpy as np, pandas as pd
from evaluation.paths import load_fold, ddpm_fakes, load_meta

ROOT  = '.'
INFO  = 'data/datasets_info.json'
DS    = ['hepatitis', 'chronic_kidney_disease', 'framingham']
FOLDS = [0, 1]
TAGS  = ['slim', 'ar']
HP    = 'h256_t200_3'

def keys(df, cols):
    out = []
    for c in cols:
        n = pd.to_numeric(df[c], errors='coerce')
        out.append(n.astype(float).astype(str) if n.notna().all() else df[c].astype(str))
    return list(zip(*out))

rows = []
for ds in DS:
    cat = load_meta(INFO, ds)['cat_cols']          # 含 target 列（full_data 下作为类别特征）
    for k in FOLDS:
        tr, te = load_fold(ROOT, ds, k)
        seen = set(keys(tr, cat))
        rows.append((ds, k, 'test', np.mean([t in seen for t in keys(te, cat)])))
        for tag in TAGS:
            for _, _, p in ddpm_fakes(ROOT, ds, k, hp=f'{HP}_{tag}'):
                fk = pd.read_csv(p)
                fk.columns = fk.columns.str.strip()
                rows.append((ds, k, tag, np.mean([t in seen for t in keys(fk, cat)])))

res = pd.DataFrame(rows, columns=['dataset', 'fold', 'src', 'in_train'])
print(res.groupby(['dataset', 'src']).in_train.mean().unstack().round(3))