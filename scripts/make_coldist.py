# scripts/make_coldist.py      python scripts/make_coldist.py
import sys, os, textwrap; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.getcwd())
from scripts.paper_style import apply, COL_W, PAGE_W
apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import ks_2samp, gaussian_kde
from pathlib import Path
from evaluation.paths import load_fold, load_meta, flat_fakes, ddpm_fakes

DDPM_ROOT, INFO_PATH, TABSYN_ROOT = ".", "data/datasets_info.json", "../tabsyn-main"   # ← 按服务器实际位置修改
OUT = Path("paper_out/figs"); OUT.mkdir(parents=True, exist_ok=True)
DS = {"risk_factors_cervical_cancer": "Cervical", "diabetic_data": "Diabetic"}
MIN_CARD = 3
MIN_CNT = 1      
METHODS = [("TabKDE", "base"), ("TabSyn", "tabsyn"), ("CAR-Diff w/o AR", "ours"), ("CAR-Diff", "ours")]

def methods(ds):
    wo = "h256_t200_3_ep200" if ds == "diabetic_data" else "h256_t200_3_final"
    return [("TabKDE", "base", "tabkde"), ("TabSyn", "tabsyn", "tabsyn"),
            ("CAR-Diff w/o AR", "ours", wo), ("CAR-Diff", "ours", "h256_t200_3_abl_ar")]

# --- 参考 TabSyn 风格的配色（高对比度），所有线型统一实线，统一细线 ---
COL = {"TabKDE": "#ff7f0e", "TabSyn": "#2ca02c", "CAR-Diff w/o AR": "#d62728", "CAR-Diff": "#1f77b4"}
LS = {m: "-" for m, _ in METHODS}
LW = {m: 0.8 for m, _ in METHODS}  # 统一线宽，不加粗
REAL_COL = "#A9A9A9"                  # 稍深一点的灰色
REAL_EDGE = "#333333"                 

N_SEL, MAX_CAT, GW = 2, 5, 0.7           

def per_fold(kind, name, ds):            
    out = []
    for k in range(5):
        if kind == "ours": fs = [p for _, s, p in ddpm_fakes(DDPM_ROOT, ds, k, hp=name)]
        else: fs = [p for _, s, p in flat_fakes(TABSYN_ROOT if kind == "tabsyn" else DDPM_ROOT, name, ds, k)]
        dfs = []
        for p in fs:
            f = pd.read_csv(p); f.columns = f.columns.str.strip(); dfs.append(f)
        out.append(pd.concat(dfs, ignore_index=True))
    return out

def train_folds(ds):
    out = []
    for k in range(5):
        tr, _ = load_fold(DDPM_ROOT, ds, k); tr = tr.copy(); tr.columns = tr.columns.str.strip(); out.append(tr)
    return out

def fold_props(real_f, fake_f, c, cats):          
    return np.array([[props(norm(r[c]), cats), props(norm(f[c]), cats)] for r, f in zip(real_f, fake_f)])

def norm(s):
    n = pd.to_numeric(s, errors="coerce")
    if n.notna().all(): return n.map(lambda v: str(int(v)) if float(v).is_integer() else str(round(v, 6)))
    return s.astype(str).str.strip()

def props(s, cats): return s.value_counts(normalize=True).reindex(cats).fillna(0).values
def wrap(t): return textwrap.fill(t, 20)

nD = len(DS)
# 稍微拉宽一点画布，缓解拥挤感（从 5.4 提高到 7.0）
fig, axes = plt.subplots(2, N_SEL * nD, figsize=(PAGE_W, 2.8 * PAGE_W / 7.0), layout="constrained", squeeze=False)

for di, (ds, dname) in enumerate(DS.items()):
    mt = load_meta(INFO_PATH, ds); lab = mt["label_col"]
    num = [c for c in mt["num_cols"] if c != lab]; cat = [c for c in mt["cat_cols"] if c != lab]

    RF = train_folds(ds)                                         
    import pickle
    cp = Path(f"paper_out/coldist_cache_{ds}.pkl")
    if cp.exists(): FK = pickle.load(open(cp, "rb"))
    else:
        FK = {m: per_fold(kind, name, ds) for m, kind, name in methods(ds)}; pickle.dump(FK, open(cp, "wb"))
    OF = FK["CAR-Diff"]
    ks = {c: np.mean([ks_2samp(r[c].dropna(), f[c].dropna()).statistic for r, f in zip(RF, OF)]) for c in num}
    def allcats(c): return sorted(set().union(*[set(norm(r[c])) for r in RF]) | set().union(*[set(norm(f[c])) for f in OF]))
    tvd = {}
    for c in cat:
        cs = allcats(c)
        tvd[c] = np.mean([0.5 * np.abs(props(norm(r[c]), cs) - props(norm(f[c]), cs)).sum() for r, f in zip(RF, OF)])

    def min_cnt(c): return np.mean([norm(r[c]).value_counts().iloc[-1] for r in RF])
    cat_ok = [c for c in cat if len(allcats(c)) >= MIN_CARD and min_cnt(c) >= MIN_CNT]
    if len(cat_ok) < N_SEL: print("⚠", ds, "满足取值数/最小类计数的类别列不足：", cat_ok)
    tvd = {c: tvd[c] for c in cat_ok}
    sel_n = sorted(ks, key=ks.get, reverse=True)[:N_SEL]; sel_c = sorted(tvd, key=tvd.get, reverse=True)[:N_SEL]
    print("done", ds, "num:", sel_n, "cat:", sel_c)
    
    for j in range(N_SEL):
        col = di * N_SEL + j
        
        # --- 上排：数值列（KDE密度图） ---
        ax = axes[0, col]
        if j < len(sel_n):
            c = sel_n[j]
            allr_raw = np.concatenate([r[c].dropna().values for r in RF])
            
            # 【核心改进 1】：截断长尾。取 1% 到 99% 分位数，避免异常值把图压扁
            x_min, x_max = np.percentile(allr_raw, [1, 99])
            grid = np.linspace(x_min, x_max, 1000)  # 【核心改进 2】：网格从 200 提升到 1000，线条更丝滑
            
            # 【核心改进 3】：引入 bw_method 参数平滑曲线（越小越平缓，默认是 'scott'）
            def kde_mean(frames):
                return np.mean([gaussian_kde(f[c].dropna().values, bw_method=0.2)(grid) for f in frames], axis=0)
                
            try:
                real_dens = kde_mean(RF)
                # Real 底层填充：透明度降到 0.25，不抢戏
                ax.fill_between(grid, real_dens, color=REAL_COL, alpha=0.25, lw=0)
                ax.plot(grid, real_dens, color=REAL_EDGE, lw=0.8, alpha=0.9)
                
                for m, _ in METHODS:
                    ax.plot(grid, kde_mean(FK[m]), color=COL[m], lw=LW[m], ls=LS[m], zorder=3)
                    
            except Exception as e: print("kde 失败", ds, c, e)
            
            if j == 0: 
                ax.set_title(dname, fontsize=10, fontweight='bold') # 稍微加大字体
                ax.set_ylabel("Density", fontsize=8)
            
            ax.set_xlabel(wrap(c), fontsize=8, labelpad=2)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.tick_params(axis='both', labelsize=7)
            ax.set_xlim(x_min, x_max) # 锁定 X 轴
        else: 
            ax.axis("off")
            
        # --- 下排：类别列（Bar柱状图） ---
        ax = axes[1, col]
        if j < len(sel_c):
            c = sel_c[j]; cs = allcats(c)
            rp = np.mean([props(norm(r[c]), cs) for r in RF], axis=0)         
            cats = [cs[i] for i in np.argsort(-rp)[:MAX_CAT]]
            P = {"Real": np.mean([props(norm(r[c]), cats) for r in RF], axis=0)}
            for m, _ in METHODS: P[m] = np.mean([props(norm(f[c]), cats) for f in FK[m]], axis=0)
            allm = [(nm, P[nm], REAL_COL if nm == "Real" else COL[nm]) for nm in ["Real"] + [m for m, _ in METHODS]]
            w = GW / len(allm)
            
            for q, (nm, v, colr) in enumerate(allm):
                edge_c = "k" if nm != "Real" else REAL_EDGE
                ax.bar(np.arange(len(cats)) + (q - (len(allm) - 1) / 2) * w, v, width=w,
                       color=colr, edgecolor=edge_c, linewidth=0.4, alpha=0.9)
            
            if rp.max() > 0.7: 
                ax.set_yscale("log"); ax.set_ylim(bottom=1e-4)
                
            ax.set_xticks(range(len(cats))); 
            ax.set_xticklabels(cats, fontsize=7)
            ax.tick_params(axis="y", labelsize=6); 
            ax.set_xlabel(wrap(c), fontsize=7, labelpad=2)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        else: 
            ax.axis("off")

for a in axes.ravel():
    for s_ in a.spines.values(): s_.set_linewidth(0.5)
    if a.get_subplotspec().rowspan.start == 0: 
        a.locator_params(axis="x", nbins=4)

# --- 构建图例 ---
legend_handles = [Patch(facecolor=REAL_COL, edgecolor=REAL_EDGE, linewidth=0.5, alpha=0.5, label="Real")]
for m, _ in METHODS:
    legend_handles.append(Line2D([0], [0], color=COL[m], lw=1.0, label=m))

fig.legend(handles=legend_handles, loc="outside lower center", ncol=len(legend_handles), 
           fontsize=8, frameon=False, handletextpad=0.4, columnspacing=1.5)

fig.savefig(OUT / "fig_coldist.pdf"); 
fig.savefig(OUT / "fig_coldist.svg"); 
plt.close(fig)
print("done")