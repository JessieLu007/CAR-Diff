# scripts/make_paper_outputs.py      python scripts/make_paper_outputs.py
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
sys.path.insert(0, os.getcwd())
from scripts.paper_style import apply, COL_W, PAGE_W
apply()
import matplotlib.pyplot as plt
from pathlib import Path
from evaluation.paths import load_fold, load_meta

DDPM_ROOT, INFO_PATH = ".", "data/datasets_info.json"
EVAL_DIR = Path("eval"); OUT = Path("paper_out"); TAB, FIG = OUT / "tables", OUT / "figs"
for p in (TAB, FIG): p.mkdir(parents=True, exist_ok=True)
KEYS = ["dataset", "fold", "method", "sample"]; N_FOLDS = 5

# ---------------- 命名（显示名只在这里改） ----------------
DS_ORDER = ["hepatitis", "chronic_kidney_disease", "risk_factors_cervical_cancer",
            "hcv_egy_data", "framingham", "diabetic_data"]
DS_NAME = {"hepatitis": "Hepatitis", "chronic_kidney_disease": "CKD", "risk_factors_cervical_cancer": "Cervical",
           "hcv_egy_data": "HCV", "framingham": "Framingham", "diabetic_data": "Diabetic"}
BL_NAME = {"smote": "SMOTE", "ctgan": "CTGAN", "tvae": "TVAE", "tabddpm": "TabDDPM",
           "tabkde": "TabKDE", "tabsyn": "TabSyn"}
OURS = "ours:main" # 选定的主训练tag
def resolve(k, ds):                      # "WO" = CAR-Diff w/o AR（diabetic 取 ep200）
    return ("ours:ep200" if ds == "diabetic_data" else "ours:final") if k == "WO" else k
SEP = (None, None)                       # 分组分隔标记
ROWS_MAIN = [("SMOTE", "smote"), ("TabKDE", "tabkde"), SEP,
             ("CTGAN", "ctgan"), ("TVAE", "tvae"), ("TabDDPM", "tabddpm"), ("TabSyn", "tabsyn"), SEP,
             ("CAR-Diff", OURS)]
GROUP_COL = {"smote": "0.55", "tabkde": "0.55", OURS: "#1b9e77"}   # 其余（深度方法）用默认色
DEEP_COL = "#7fa6c9"
ABL_V = [OURS, "ours:abl_legacy", "ours:abl_nointeract", "ours:abl_sampled", "ours:abl_nosharedec"]
ABL_L = ["CAR-Diff", "legacy cat. input", "no interaction", "sampled cat. input", "no shared decoder"]
STAB = ["dh128", "dh512", "nt100", "nt500", "nl2", "nl4", "ep100", "ep400"]
STAB_L = ["dim=128", "dim=512", "T=100", "T=500", "layers=2", "layers=4", "epochs=100", "epochs=400"]

# ---------------- 指标：显示名、方向、缩放、小数位 ----------------
SPEC = {"shape_err": ("Shape error (\\%)", "low", 100, 2),
        "cpt_err_all": ("CPT error (\\%)", "low", 100, 2),
        "c2st_dev": ("C2ST deviation (\\%)", "low", 100, 1),
        "alpha_precision": ("$\\alpha$-precision (\\%)", "high", 100, 1),
        "beta_recall": ("$\\beta$-recall (\\%)", "high", 100, 1),
        "cpt_err_cc": ("CPT error, cat--cat (\\%)", "low", 100, 2),
        "cpt_err_nn": ("CPT error, num--num (\\%)", "low", 100, 2),
        "cpt_err_nc": ("CPT error, num--cat (\\%)", "low", 100, 2),
        "corr_diff_nn": ("Corr. difference, num--num (\\%)", "low", 100, 2),
        "corr_diff_nc": ("Corr. difference, num--cat (\\%)", "low", 100, 2),
        "corr_diff_cc": ("Corr. difference, cat--cat (\\%)", "low", 100, 2),
        "tstr_roc_auc": ("TSTR ROC-AUC (\\%)", None, 100, 1),
        "tstr_pr_auc": ("TSTR PR-AUC (\\%)", None, 100, 1),
        "dcr_ratio": ("DCR ratio", None, 1, 2)}
SHORT = {"shape_err": "Shape", "cpt_err_all": "CPT", "c2st_dev": "C2ST", "alpha_precision": r"$\alpha$-prec.",
         "beta_recall": r"$\beta$-rec.", "cpt_err_cc": "CPT cc", "cpt_err_nn": "CPT nn", "cpt_err_nc": "CPT nc", 
         "corr_diff_nn": "Corr. nn", "corr_diff_nc": "Corr. nc", "corr_diff_cc": "Corr. cc",
         "tstr_roc_auc": "TSTR ROC-AUC", "tstr_pr_auc": "TSTR PR-AUC"}
MAIN = ["shape_err", "cpt_err_all", "c2st_dev", "alpha_precision", "beta_recall"]
ABL_M = MAIN + ["cpt_err_cc", "corr_diff_cc"]
ALL = list(SPEC); DOWN = [c for c in ALL if c.startswith(("tstr_", "dcr_"))]; FID = [c for c in ALL if c not in DOWN]

# ---------------- 读取与折内聚合 ----------------
raw = pd.concat([pd.read_csv(p) for p in sorted(EVAL_DIR.glob("*/summary.csv"))], ignore_index=True)
raw["c2st_dev"] = (raw["c2st_auc"] - 0.5).abs()
tag = raw["method"].str.extract(r"^ours_h\d+_t\d+_\d+_(.+)$")[0]
tag = tag.where(~((raw["dataset"] == "diabetic_data") & (tag == "final")), "ep50")   # diabetic 旧 final = ep50
raw["method"] = np.where(tag.notna(), "ours:" + tag.fillna(""), raw["method"])

def fold_level(ref):
    fid = raw[(raw["ref"] == ref) & (raw["method"] != "real")][KEYS + FID]
    down = raw[(raw["ref"] == "both") & (raw["method"] != "real")][KEYS + DOWN]
    g = fid.merge(down, on=KEYS, how="left").groupby(["dataset", "fold", "method"])[ALL].mean()
    real = raw[(raw["method"] == "real") & (raw["ref"] == "both")].groupby(["dataset", "fold", "method"])[DOWN].mean()
    return pd.concat([g, real]).sort_index()
Gtr, Gte = fold_level("train"), fold_level("test")

def series(G, ds, key, m):               # 某数据集、某方法、某指标的折级序列
    return G.xs((ds, key), level=("dataset", "method"))[m]

PREAMBLE = r"""\documentclass[twocolumn]{article} % 单栏就去掉 twocolumn
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

\usepackage{booktabs}
\usepackage{multirow}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}

\begin{document}

"""
STANDALONE = True  # False时只输出表格本体，供正文\input
def write_tex(fname, lines):
    body = "\n".join(lines)
    txt = PREAMBLE + body + "\n\n\\end{document}\n" if STANDALONE else body
    (TAB / f"{fname}.tex").write_text(txt, encoding="utf-8")
    (TAB / f"{fname}.tex").write_text(PREAMBLE + "\n".join(lines) + "\n\n\\end{document}\n", encoding="utf-8")

# ---------------- LaTeX：面板表（行=方法，列=数据集，均值大、±std 小两档） ----------------
def cell(mu, sd, sc, dec, bold):
    if pd.isna(mu): return "--"
    t = f"{mu * sc:.{dec}f}"
    if bold: t = rf"\textbf{{{t}}}"
    return t + (rf"{{\tiny$\pm${sd * sc:.{dec}f}}}" if pd.notna(sd) else "")

def panel_table(fname, G, datasets, rows, metrics, caption, rank=True):
    gp = G.groupby(level=["dataset", "method"]); mu, sd = gp.mean(), gp.std(ddof=1)
    rows_f = [r for r in rows if r[0]]
    ncol = len(datasets) + (1 if rank else 0)
    L = [r"\begin{table*}[t]", r"\centering\footnotesize", rf"\caption{{{caption}}}", rf"\label{{tab:{fname}}}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * ncol + "}", r"\toprule",
         " & ".join(["Method"] + [DS_NAME[d] for d in datasets] + (["Avg. rank"] if rank else [])) + r" \\"]
    for m in metrics:
        title, dr, sc, dec = SPEC[m]
        arrow = {"low": r" $\downarrow$", "high": r" $\uparrow$", None: ""}[dr]
        L += [r"\midrule", rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{{title}{arrow}}}}} \\"]
        rk = {}
        for ds in datasets:
            v = pd.Series({lab: mu.loc[(ds, resolve(k, ds)), m] for lab, k in rows_f
                           if (ds, resolve(k, ds)) in mu.index})
            if dr and v.notna().any(): rk[ds] = v.rank(ascending=(dr == "low"))
        avg = pd.DataFrame(rk).mean(axis=1) if rk else None
        for lab, k in rows:
            if lab is None: L.append(rf"\cmidrule{{1-{ncol + 1}}}"); continue
            cs = []
            for ds in datasets:
                key = (ds, resolve(k, ds))
                if key not in mu.index: cs.append("--"); continue
                b = bool(dr) and ds in rk and lab in rk[ds].index and rk[ds][lab] == rk[ds].min()
                cs.append(cell(mu.loc[key, m], sd.loc[key, m], sc, dec, b))
            if rank:
                if avg is not None and lab in avg.index:
                    t = f"{avg[lab]:.2f}"; cs.append(rf"\textbf{{{t}}}" if avg[lab] == avg.min() else t)
                else: cs.append("--")
            L.append(" & ".join([lab] + cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    write_tex(fname, L)
    #(TAB / f"{fname}.tex").write_text("\n".join(L), encoding="utf-8")

def rank_summary(fname, G, rows, metrics, datasets, caption):
    rows = [r for r in rows if r[0]]
    mu = G.groupby(level=["dataset", "method"]).mean(); R = {}
    for m in metrics:
        per = [pd.Series({lab: mu.loc[(ds, resolve(k, ds)), m] for lab, k in rows}).rank(ascending=(SPEC[m][1] == "low"))
               for ds in datasets]
        R[m] = pd.concat(per, axis=1).mean(axis=1)
    A = pd.DataFrame(R); A["Overall"] = A.mean(axis=1)
    A.round(3).to_csv(TAB / f"{fname}.csv")
    L = [r"\begin{table}[t]", r"\centering\footnotesize", rf"\caption{{{caption}}}", rf"\label{{tab:{fname}}}",
         r"\begin{tabular}{l" + "c" * A.shape[1] + "}", r"\toprule",
         " & ".join(["Method"] + [SHORT[m] for m in metrics] + ["Overall"]) + r" \\", r"\midrule"]
    for lab in A.index:
        cs = [(rf"\textbf{{{A.loc[lab, c]:.2f}}}" if A.loc[lab, c] == A[c].min() else f"{A.loc[lab, c]:.2f}") for c in A.columns]
        L.append(" & ".join([lab] + cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex(fname, L)


def dump_long(G, ref, fname):
    gp = G.groupby(level=["dataset", "method"]); mu, sd = gp.mean(), gp.std(ddof=1)
    rows_f = [r for r in ROWS_MAIN if r[0]]; out = []
    for m in ALL:
        _, dr, sc, _ = SPEC[m]
        for ds in DS_ORDER:
            v = {lab: mu.loc[(ds, resolve(k, ds)), m] for lab, k in rows_f
                 if (ds, resolve(k, ds)) in mu.index and pd.notna(mu.loc[(ds, resolve(k, ds)), m])}
            rk = pd.Series(v).rank(ascending=(dr == "low")) if dr and v else pd.Series(dtype=float)
            for lab, k in rows_f:
                key = (ds, resolve(k, ds))
                if key in mu.index:
                    out.append(dict(ref=ref, dataset=DS_NAME[ds], method=lab, metric=m,
                                    mean=mu.loc[key, m] * sc, std=sd.loc[key, m] * sc, rank=rk.get(lab, np.nan)))
    for ds in DS_ORDER:                              # w/o AR 对照（不参与排名）
        for m in ALL:
            key = (ds, resolve("WO", ds))
            if key in mu.index:
                out.append(dict(ref=ref, dataset=DS_NAME[ds], method="CAR-Diff w/o AR", metric=m,
                                mean=mu.loc[key, m] * SPEC[m][2], std=sd.loc[key, m] * SPEC[m][2], rank=np.nan))

    for m in DOWN:                                   # Real 行（TSTR、DCR 的基线）
        for ds in DS_ORDER:
            if (ds, "real") in mu.index:
                out.append(dict(ref=ref, dataset=DS_NAME[ds], method="Real", metric=m,
                                mean=mu.loc[(ds, "real"), m] * SPEC[m][2], std=sd.loc[(ds, "real"), m] * SPEC[m][2], rank=np.nan))
    pd.DataFrame(out).round(4).to_csv(TAB / fname, index=False)
    

def dataset_table():
    rows = []
    for ds in DS_ORDER:
        tr, te = load_fold(DDPM_ROOT, ds, 0); full = pd.concat([tr, te]); mt = load_meta(INFO_PATH, ds)
        lab = mt["label_col"]; vc = full[lab].value_counts()
        mn = (full[lab].astype(str) == str(mt["minority_val"])).sum() or vc.min()
        rows.append((DS_NAME[ds], len(full), len([c for c in mt["num_cols"] if c != lab]),
                     len([c for c in mt["cat_cols"] if c != lab]), vc.size, vc.max() / mn))
    L = [r"\begin{table}[t]", r"\centering\footnotesize",
         r"\caption{Dataset statistics. Imbalance ratio: majority class size over minority class size of the label.}",
         r"\label{tab:datasets}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Dataset & \#Rows & \#Num. & \#Cat. & \#Classes & Imb. ratio \\", r"\midrule"]
    L += [f"{n} & {r} & {a} & {b} & {c} & {d:.1f} \\\\" for n, r, a, b, c, d in rows]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_tex("tab_datasets", L)
    #(TAB / "tab_datasets.tex").write_text("\n".join(L), encoding="utf-8")


def res_stab(k, ds):                     # diabetic 列：AR 稳定性未做，改用 CAR-Diff w/o AR 口径（图中以 † 标注）
    if ds != "diabetic_data": return k
    if k == OURS: return "ours:ep200"
    return "ours:" + k[len("ours:ar_"):] if k.startswith("ours:ar_") else k

FACTORS = [("Hidden dim", ("dh128", "dh512"), (128, 256, 512)),
           #("Timesteps T", ("nt100", "nt500"), (100, 200, 500)),
           #("Layers", ("nl2", "nl4"), (2, 3, 4)),
           ("Epochs", ("ep100", "ep400"), (100, 200, 400))]

def pair_gain(G, ds, key, ref_key, m):   # 方向调整后的折均值增益（pp）；缺数据返回 NaN
    sign = 1 if SPEC[m][1] == "high" else -1
    try: x, r = series(G, ds, key, m), series(G, ds, ref_key, m)
    except KeyError: return np.nan
    idx = x.index.intersection(r.index)
    return sign * (x.loc[idx] - r.loc[idx]).mean() * 100 if len(idx) else np.nan

def stability_table_old(G, metrics=MAIN, fname="tab_stability"):
    cols = [("dh128", "dim=128"), ("dh512", "dim=512"), ("nt100", "T=100"), ("nt500", "T=500"),
            ("nl2", "layers=2"), ("nl4", "layers=4"), ("ep100", "epochs=100"), ("ep400", "epochs=400")]
    ds_list = [d for d in DS_ORDER if d != "diabetic_data"]
    L = [r"\begin{table*}[t]", r"\centering\footnotesize",
         r"\caption{Sensitivity to hidden dimension, diffusion timesteps, network depth, and training epochs, "
         r"on the five small datasets (Diabetic excluded). Entries: paired gain relative to the default setting "
         r"(dim 256, $T$=200, 3 layers, 200 epochs), in percentage points, positive = better; mean$\pm$std over datasets.}",
         r"\label{tab:stability}", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         " & ".join(["Metric"] + [c[1] for c in cols]) + r" \\", r"\midrule"]
    for m in metrics:
        cs = []
        for t, _ in cols:
            g = np.array([pair_gain(G, ds, f"ours:ar_{t}", OURS, m) for ds in ds_list])
            cs.append(f"{np.nanmean(g):+.2f}" + rf"{{\tiny$\pm${np.nanstd(g, ddof=1):.2f}}}")
        L.append(" & ".join([SHORT[m]] + cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex(fname, L)

def stability_table(G, metrics=MAIN, fname="tab_stability"):
    cols = [("dh128", "dim=128"), ("dh512", "dim=512"), ("nt100", "T=100"), ("nt500", "T=500"),
            ("nl2", "layers=2"), ("nl4", "layers=4"), ("ep100", "epochs=100"), ("ep400", "epochs=400")]
    ds_list = [d for d in DS_ORDER if d != "diabetic_data"]
    rows = []
    L = [r"\begin{table*}[t]", r"\centering\footnotesize",
         r"\caption{Sensitivity to hidden dimension, diffusion timesteps, network depth, and training epochs, "
         r"on the five small datasets (Diabetic excluded). Entries: paired gain relative to the default setting "
         r"(dim 256, $T$=200, 3 layers, 200 epochs), in percentage points, positive = better; mean$\pm$std over datasets.}",
         r"\label{tab:stability}", r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
         " & ".join(["Metric"] + [c[1] for c in cols]) + r" \\", r"\midrule"]
    for m in metrics:
        cs = []
        for t, lab in cols:
            g = np.array([pair_gain(G, ds, f"ours:ar_{t}", OURS, m) for ds in ds_list])
            mu, sd = np.nanmean(g), np.nanstd(g, ddof=1)
            cs.append(f"{mu:+.2f}" + rf"{{\tiny$\pm${sd:.2f}}}")
            rows.append(dict(metric=m, col=t, label=lab, mean_gain_pp=round(mu, 4), std_gain_pp=round(sd, 4),
                             n_datasets=int(np.sum(~np.isnan(g)))))
        L.append(" & ".join([SHORT[m]] + cs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_tex(fname, L)
    pd.DataFrame(rows).to_csv(TAB / f"{fname}.csv", index=False)


def ablation_dots(G, metrics=ABL_M, fname="fig_ablation_dots"):
    fig, ax = plt.subplots(figsize=(PAGE_W, 2.8), layout="constrained")
    cols = ["#1b9e77", "#7570b3", "#d95f02", "#e7298a", "#66a61e"]; w = 0.7 / len(ABL_V)
    for q, (v, lab) in enumerate(zip(ABL_V, ABL_L)):
        mu, sd = [], []
        for m in metrics:
            g = np.array([pair_gain(G, ds, resolve(v, ds), resolve("WO", ds), m) for ds in DS_ORDER])
            mu.append(np.nanmean(g)); sd.append(np.nanstd(g, ddof=1))
        ax.errorbar(np.arange(len(metrics)) + (q - (len(ABL_V) - 1) / 2) * w, mu, yerr=sd, fmt="o", ms=3.5,
                    color=cols[q], capsize=2, lw=0.8, label=lab)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels([SHORT[m] for m in metrics], fontsize=8)
    ax.set_ylabel("gain vs. w/o AR (pp)", fontsize=8); ax.tick_params(axis="y", labelsize=7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.svg"); plt.close(fig)

# ---------------- 图 2：AR 头配对增益（Ours − w/o AR，方向调整后，单位 pp） ----------------
def ablation_gain_fig(G, metrics, fname="fig_ablation_gain"):
    nr, nc = 2, 4
    fig, axes = plt.subplots(nr, nc, figsize=(PAGE_W, 3.9), layout="constrained")
    names = [DS_NAME[d] for d in DS_ORDER]
    for ax, m in zip(axes.ravel(), metrics):
        sign = 1 if SPEC[m][1] == "high" else -1
        mus, sds = [], []
        for ds in DS_ORDER:
            r, x = series(G, ds, OURS, m), series(G, ds, resolve("WO", ds), m)
            idx = r.index.intersection(x.index)
            d = sign * (r.loc[idx] - x.loc[idx]) * 100
            mus.append(d.mean()); sds.append(d.std(ddof=1) if len(d) > 1 else 0)
        cols = ["#1b9e77" if v >= 0 else "#d95f02" for v in mus]
        ax.bar(range(len(mus)), mus, yerr=sds, color=cols, capsize=2, error_kw=dict(lw=0.7))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(SHORT[m], fontsize=9); ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7); ax.tick_params(axis="y", labelsize=7)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    for ax in axes.ravel()[len(metrics):]: ax.axis("off")
    axes[0, 0].set_ylabel("gain (pp)", fontsize=8); axes[1, 0].set_ylabel("gain (pp)", fontsize=8)
    fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.svg"); plt.close(fig)

def ablation_gain_csv(G, metrics=ABL_M, fname="ablation_gain.csv"):
    rows = []
    for ds in DS_ORDER:
        for m in metrics:
            sign = 1 if SPEC[m][1] == "high" else -1
            r, x = series(G, ds, OURS, m), series(G, ds, resolve("WO", ds), m)
            idx = r.index.intersection(x.index)
            d = sign * (r.loc[idx] - x.loc[idx]) * 100          # pp，正=AR 更好
            rows.append(dict(dataset=DS_NAME[ds], metric=m, gain_pp=d.mean(),
                             fold_std=d.std(ddof=1), folds_positive=int((d > 0).sum()), n_folds=len(d)))
    pd.DataFrame(rows).round(3).to_csv(TAB / fname, index=False)


def gain_csv(G, pairs, metrics, fname):
    # pairs: [(组名, 变体key, 参照key)]，key 可含 "WO"，走 resolve/res_stab
    rows = []
    for grp, v, ref in pairs:
        for ds in DS_ORDER:
            for m in metrics:
                sign = 1 if SPEC[m][1] == "high" else -1
                try: x, r = series(G, ds, res_stab(resolve(v, ds), ds) if grp.startswith("stab") else resolve(v, ds),
                                   m), series(G, ds, res_stab(resolve(ref, ds), ds) if grp.startswith("stab") else resolve(ref, ds), m)
                except KeyError: continue
                idx = x.index.intersection(r.index)
                if len(idx) == 0: continue
                d = sign * (x.loc[idx] - r.loc[idx]) * 100
                rows.append(dict(group=grp, variant=v, ref=ref, dataset=DS_NAME[ds], metric=m,
                                 gain_pp=d.mean(), fold_std=d.std(ddof=1),
                                 folds_positive=int((d > 0).sum()), n_folds=len(d)))
    pd.DataFrame(rows).round(3).to_csv(TAB / fname, index=False)

pairs  = [("ar", OURS, "WO")]
pairs += [("abl", v, "WO") for v in ABL_V[1:]]
pairs += [("stab", f"ours:ar_{t}", OURS) for t in STAB]
gain_csv(Gtr, pairs, ABL_M, "all_gains.csv")
gain_csv(Gtr, [("ar", OURS, "WO")], MAIN, "ar_gain_train.csv")
gain_csv(Gte, [("ar", OURS, "WO")], MAIN, "ar_gain_test.csv")
gain_csv(Gte, [("stab", f"ours:ar_{t}", OURS) for t in ["ep100", "ep400"]], ["beta_recall"], "hcv_epochs_test.csv")
#gain_csv(Gtr, [("prior", "ours:abl_ar_noprior", OURS)], MAIN, "prior_gain_train.csv")

def ar_dcr_csv(G, fname="ar_dcr.csv"):
    rows = []
    for ds in DS_ORDER:
        for m in ["dcr_ratio", "dcr_exact", "dcr_mean"]:
            if m not in G.columns: continue
            a, b = series(G, ds, OURS, m), series(G, ds, resolve("WO", ds), m)
            idx = a.index.intersection(b.index)
            rows.append(dict(dataset=DS_NAME[ds], metric=m, ar=a.loc[idx].mean(), wo_ar=b.loc[idx].mean(),
                             diff=(a.loc[idx] - b.loc[idx]).mean(), fold_std=(a.loc[idx] - b.loc[idx]).std(ddof=1)))
    pd.DataFrame(rows).round(4).to_csv(TAB / fname, index=False)
ar_dcr_csv(Gtr)


# ---------------- 图 3：TSTR 面板图（行=指标，列=数据集；点=均值，棒=折间 std，虚线=Real） ----------------
def tstr_fig(G, metrics, fname="fig_tstr"):
    meths = [(l, k) for l, k in ROWS_MAIN if l]; n = len(meths)
    cols = [GROUP_COL.get(k, DEEP_COL) for _, k in meths]
    fig, axes = plt.subplots(len(metrics), len(DS_ORDER), figsize=(PAGE_W, 1.5 * len(metrics) + 1.1),
                             squeeze=False, layout="constrained")
    for i, m in enumerate(metrics):
        for j, ds in enumerate(DS_ORDER):
            ax = axes[i, j]
            for x, ((lab, k), c) in enumerate(zip(meths, cols)):
                s = series(G, ds, k, m) * 100
                ax.errorbar(x, s.mean(), yerr=s.std(ddof=1), fmt="o", ms=3, color=c, capsize=1.5, lw=0.7)
            ax.axhline(series(G, ds, "real", m).mean() * 100, color="k", ls="--", lw=0.8)
            ax.set_xticks(range(n))
            ax.set_xticklabels([l for l, _ in meths] if i == len(metrics) - 1 else [], rotation=90, fontsize=6)
            ax.tick_params(axis="y", labelsize=6.5)
            for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
            if i == 0: ax.set_title(DS_NAME[ds], fontsize=8)
            if j == 0: ax.set_ylabel(SHORT[m] + " (%)", fontsize=7)
    fig.savefig(FIG / f"{fname}.pdf"); fig.savefig(FIG / f"{fname}.svg"); plt.close(fig)

# ================= 生成 =================
USED = {k for _, k in ROWS_MAIN if k} | {"ours:final", "ours:ep200"} | set(ABL_V) | {f"ours:ar_{t}" for t in STAB}
cnt = Gtr["shape_err"].groupby(level=["dataset", "method"]).count()
bad = cnt[(cnt != N_FOLDS) & cnt.index.get_level_values("method").isin(USED)]
if len(bad): print("⚠ 折数不是 5 的组合：\n", bad)

cap = "Mean$\\pm$std over 5 folds, reference: training set. Bold: best; Avg. rank: mean within-dataset rank."
dataset_table()

for fn, m, nm in [("fid_shape", "shape_err", "Column-wise density (Shape error)"),
                  ("fid_cpt", "cpt_err_all", "Pair-wise correlation (CPT error)"),
                  ("fid_c2st", "c2st_dev", "Classifier two-sample test (deviation of AUC from 0.5)"),
                  ("fid_alpha", "alpha_precision", r"$\alpha$-precision"),
                  ("fid_beta", "beta_recall", r"$\beta$-recall")]:
    panel_table(fn, Gtr, DS_ORDER, ROWS_MAIN, [m], nm + ". " + cap)

rank_summary("rank_test", Gte, ROWS_MAIN, MAIN, DS_ORDER,
             "Average within-dataset rank with the held-out test set as reference.")
rank_summary("rank_train", Gtr, ROWS_MAIN, MAIN, DS_ORDER,
             "Average within-dataset rank with the training set as reference.")
panel_table("dcr", Gtr, DS_ORDER, ROWS_MAIN, ["dcr_ratio"],
            "DCR ratio (close to 1 is desirable; clearly below 1 suggests memorization). Mean$\\pm$std over 5 folds.",
            rank=False)
panel_table("fid_cpt_blocks", Gtr, DS_ORDER, ROWS_MAIN, ["cpt_err_nn", "cpt_err_nc", "cpt_err_cc"],
            "CPT error by block: num--num, num--cat, cat--cat. " + cap)

ablation_gain_fig(Gtr, ABL_M)
ablation_gain_csv(Gtr)
stability_table(Gtr)
ablation_dots(Gtr)

dump_long(Gtr, "train", "metrics_long_train.csv")
dump_long(Gte, "test", "metrics_long_test.csv")

tstr_fig(Gtr, ["tstr_pr_auc"]) #"tstr_roc_auc", 
print("done ->", OUT)