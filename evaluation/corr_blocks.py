# evaluation/corr_blocks.py
"""python -m evaluation.corr_blocks --datasets hepatitis chronic_kidney_disease framingham --folds 0 1 \
       --ours_hp h256_t200_3_target_epochs=200_catzero [--ref test|train]
关联矩阵分块误差 |C_real - C_fake| 的均值（各 sample 取均值）：
nn=数值-数值，cc=类别-类别，nc=数值-类别，cc/nn=类别块相对数值块的误差比。对比 ours、tabddpm、tabsyn。"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from evaluation.metrics import _mixed_corr
from evaluation.paths import load_fold, load_meta, ddpm_fakes, flat_fakes
from evaluation.align import make_aligner
from evaluation.run import _default_ddpm_root, _default_tabsyn_root
from utils import common


def _blocks(c_real, c_fake, p):
    d = (c_real - c_fake).abs().to_numpy()
    tri = lambda m: m[np.triu_indices(m.shape[0], k=1)]
    f = lambda a: float(a.mean()) if a.size else float('nan')
    return dict(nn=f(tri(d[:p, :p])), cc=f(tri(d[p:, p:])), nc=f(d[:p, p:].ravel()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', required=True)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--ours_hp', required=True)
    ap.add_argument('--ref', choices=['test', 'train'], default='test')
    ap.add_argument('--ddpm-root', default=_default_ddpm_root())
    ap.add_argument('--tabsyn-root', default=_default_tabsyn_root())
    ap.add_argument('--baseline-root', default=None, help='tabddpm 所在根目录，默认等于 --ddpm-root')
    a = ap.parse_args()

    rows = []
    for ds in a.datasets:
        ds = common.resolve_dataset(ds)
        meta = load_meta(Path(a.ddpm_root) / 'data' / 'datasets_info.json', ds)
        num, cat = meta['num_cols'], meta['cat_cols']
        for k in a.folds:
            train, test = load_fold(a.ddpm_root, ds, k)
            ref = test if a.ref == 'test' else train
            c_real = _mixed_corr(ref, num, cat)
            align = make_aligner(train, meta)
            sources = {
                'ours': ddpm_fakes(a.ddpm_root, ds, k, hp=a.ours_hp),
                'tabddpm': flat_fakes(a.baseline_root or a.ddpm_root, 'tabddpm', ds, k),
                'tabsyn': flat_fakes(a.tabsyn_root, 'tabsyn', ds, k),
            }
            for name, it in sources.items():
                for _, sample, path in it:
                    fake = align(pd.read_csv(path))
                    rows.append(dict(dataset=ds, fold=k, method=name, sample=sample,
                                     **_blocks(c_real, _mixed_corr(fake, num, cat), len(num))))

    g = pd.DataFrame(rows).groupby(['dataset', 'fold', 'method'])
    out = g[['nn', 'cc', 'nc']].mean()
    out['cc_sd'] = g['cc'].std(ddof=1)
    out['cc/nn'] = out['cc'] / out['nn']
    print(out.round(4))


if __name__ == '__main__':
    main()


# # evaluation/corr_blocks.py
# """python -m evaluation.corr_blocks --datasets hepatitis chronic_kidney_disease --folds 0 1 \
#        --base_hp h256_t200_3_target_epochs=200_catprob --cand_hp h256_t200_3_target_epochs=200_catzero
# 关联矩阵分块的 |C_real - C_fake| 均值（ref=test）：nn=数值-数值，cc=类别-类别，nc=数值-类别。
# 输出候选减基线的配对差（按 sample 配对），负值=候选更接近真实。"""
# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd

# from evaluation.metrics import _mixed_corr
# from evaluation.paths import load_fold, load_meta, ddpm_fakes
# from evaluation.align import make_aligner
# from evaluation.run import _default_ddpm_root
# from utils import common


# def _blocks(c_real, c_fake, p):
#     d = (c_real - c_fake).abs().to_numpy()
#     iu = lambda m: m[np.triu_indices(m.shape[0], k=1)]
#     f = lambda a: float(a.mean()) if a.size else float('nan')
#     return dict(nn=f(iu(d[:p, :p])), cc=f(iu(d[p:, p:])), nc=f(d[:p, p:].ravel()))


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--datasets', nargs='+', required=True)
#     ap.add_argument('--folds', type=int, nargs='+', default=[0, 1])
#     ap.add_argument('--base_hp', required=True)
#     ap.add_argument('--cand_hp', required=True)
#     ap.add_argument('--ddpm-root', default=_default_ddpm_root())
#     a = ap.parse_args()

#     rows = []
#     for ds in a.datasets:
#         ds = common.resolve_dataset(ds)
#         meta = load_meta(Path(a.ddpm_root) / 'data' / 'datasets_info.json', ds)
#         num, cat = meta['num_cols'], meta['cat_cols']
#         for k in a.folds:
#             train, test = load_fold(a.ddpm_root, ds, k)
#             c_real = _mixed_corr(test, num, cat)
#             align = make_aligner(train, meta)
#             for hp in (a.base_hp, a.cand_hp):
#                 for _, sample, path in ddpm_fakes(a.ddpm_root, ds, k, hp=hp):
#                     fake = align(pd.read_csv(path))
#                     rows.append(dict(dataset=ds, fold=k, hp=hp, sample=sample,
#                                      **_blocks(c_real, _mixed_corr(fake, num, cat), len(num))))

#     piv = pd.DataFrame(rows).pivot_table(index=['dataset', 'fold', 'sample'], columns='hp',
#                                          values=['nn', 'cc', 'nc'], dropna=False)
#     out = []
#     for b in ['nn', 'cc', 'nc']:
#         delta = (piv[(b, a.cand_hp)] - piv[(b, a.base_hp)]).groupby(['dataset', 'fold'])
#         base = piv[(b, a.base_hp)].groupby(['dataset', 'fold']).mean()
#         out.append(pd.DataFrame({f'{b}_base': base, f'{b}_dmean': delta.mean(), f'{b}_dsd': delta.std(ddof=1)}))
#     print(pd.concat(out, axis=1).round(4))


# if __name__ == '__main__':
#     main()