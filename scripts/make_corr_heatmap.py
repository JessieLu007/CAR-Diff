# scripts/make_corr_heatmap.py      python scripts/make_corr_heatmap.py [--redo]
import sys, os, pickle, argparse; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.getcwd())
from scripts.paper_style import apply, COL_W, PAGE_W
apply()
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from evaluation.paths import load_fold, load_meta, flat_fakes, ddpm_fakes
from evaluation.utils import _mixed_corr

DDPM_ROOT, INFO_PATH = ".", "data/datasets_info.json"
TABSYN_ROOT = "../tabsyn-main"                 # ← 按服务器实际位置修改
OURS_HP = "h256_t200_3_abl_ar"
OUT = Path("paper_out"); (OUT / "figs").mkdir(parents=True, exist_ok=True); (OUT / "tables").mkdir(exist_ok=True)
CACHE = OUT / "corr_cache.pkl"
DS = ["hepatitis", "chronic_kidney_disease", "risk_factors_cervical_cancer", "hcv_egy_data", "framingham", "diabetic_data"]
DS_NAME = {"hepatitis": "Hepatitis", "chronic_kidney_disease": "CKD", "risk_factors_cervical_cancer": "Cervical",
           "hcv_egy_data": "HCV", "framingham": "Framingham", "diabetic_data": "Diabetic"}
METHODS = [("SMOTE", "smote", "base"), ("CTGAN", "ctgan", "base"), ("TVAE", "tvae", "base"),
           ("TabDDPM", "tabddpm", "base"), ("TabKDE", "tabkde", "base"), ("TabSyn", "tabsyn", "tabsyn"),
           ("CAR-Diff", None, "ours")]
ap = argparse.ArgumentParser(); ap.add_argument("--redo", action="store_true"); a = ap.parse_args()

def fakes(kind, name, ds, k):
    if kind == "ours": return [(s, p) for _, s, p in ddpm_fakes(DDPM_ROOT, ds, k, hp=OURS_HP)]
    return [(s, p) for _, s, p in flat_fakes(TABSYN_ROOT if kind == "tabsyn" else DDPM_ROOT, name, ds, k)]

if CACHE.exists() and not a.redo:
    M = pickle.load(open(CACHE, "rb"))
else:
    M = {}                                   # (ds, 显示名) -> (矩阵 [折均值再取均值], p=数值列数)
    for ds in DS:
        mt = load_meta(INFO_PATH, ds); num, cat = mt["num_cols"], mt["cat_cols"]; cols = num + cat
        acc = {lab: [] for lab, _, _ in METHODS}
        for k in range(5):
            tr, _ = load_fold(DDPM_ROOT, ds, k); C_ref = _mixed_corr(tr, num, cat)
            for lab, name, kind in METHODS:
                fs = fakes(kind, name, ds, k)
                if not fs: print(f"⚠ 缺少 {ds} fold{k} {lab}"); continue
                ds_ = []
                for s, p in fs:
                    fk = pd.read_csv(p); fk.columns = fk.columns.str.strip()
                    ds_.append((C_ref - _mixed_corr(fk[cols], num, cat)).abs().values)
                if len(fs) != 5: print(f"⚠ {ds} fold{k} {lab}: 采样数 {len(fs)}")
                acc[lab].append(np.mean(ds_, axis=0))       # 折内先对采样取均值
        for lab, _, _ in METHODS:
            if acc[lab]: M[(ds, lab)] = (np.mean(acc[lab], axis=0), len(num))   # 再对折取均值
        print("done", ds, flush=True)
    pickle.dump(M, open(CACHE, "wb"))

# 与 corr_diff_{nn,nc,cc} 对齐的块均值，供与表核对
rows = []
for (ds, lab), (D, p) in M.items():
    up = lambda m: m[np.triu_indices(m.shape[0], k=1)]
    rows.append(dict(dataset=ds, method=lab, nn=np.mean(up(D[:p, :p])) if p > 1 else np.nan,
                     nc=D[:p, p:].mean() if 0 < p < D.shape[0] else np.nan,
                     cc=np.mean(up(D[p:, p:])) if D.shape[0] - p > 1 else np.nan))
pd.DataFrame(rows).to_csv(OUT / "tables" / "corr_block_means.csv", index=False)

nD, nM = len(DS), len(METHODS)
allv = np.concatenate([M[k][0].ravel() for k in M])
vmax = max(np.percentile(allv, 95), 1e-3)          # 全局统一上限；想更浅就把 95 改成 99，想更深就改小

CMAP = ["Blues", "GnBu", "YlGnBu", "viridis_r", "mako_r"]
CELL, GAP, VSCALE = 1.0, 0.06, 0.70  # 格子边长（英寸）与间距比例, VSCALE<1压缩纵向
CELL = PAGE_W / ((nM + 0.3) * (1 + GAP))
fig = plt.figure(figsize=(PAGE_W, CELL * VSCALE * nD * (1 + GAP) + 0.3))
gs = GridSpec(nD, nM + 1, width_ratios=[1] * nM + [0.08], wspace=GAP, hspace=GAP)


im = None
for i, ds in enumerate(DS):
    for j, (lab, _, _) in enumerate(METHODS):
        ax = fig.add_subplot(gs[i, j])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False); #s.set_linewidth(0.5); s.set_color("0.4")
        if (ds, lab) in M:
            D, p = M[(ds, lab)]
            im = ax.imshow(D, cmap=CMAP[1], vmin=0, vmax=vmax, interpolation="nearest", aspect="auto")
            #ax.axhline(p - 0.5, color="0.4", lw=0.3); ax.axvline(p - 0.5, color="0.4", lw=0.3)
        if i == 0: ax.set_title(lab, fontsize=8)
        if j == 0: ax.set_ylabel(DS_NAME[ds], fontsize=8)
cax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=6, length=2)
fig.savefig(OUT / "figs" / "fig_corr_heatmap.pdf", bbox_inches='tight', pad_inches=0.01)
fig.savefig(OUT / "figs" / "fig_corr_heatmap.svg", bbox_inches='tight', pad_inches=0.01)
print("done")