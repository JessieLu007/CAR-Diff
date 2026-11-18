# scripts/make_cat_checks.py      python scripts/make_cat_checks.py
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from pathlib import Path
from evaluation.paths import load_fold, load_meta, flat_fakes, ddpm_fakes

DDPM_ROOT, INFO_PATH, TABSYN_ROOT = ".", "data/datasets_info.json", "../tabsyn-main"   # ← 按服务器实际位置修改
OUT = Path("paper_out/tables"); OUT.mkdir(parents=True, exist_ok=True)
DS = {"hepatitis": "Hepatitis", "chronic_kidney_disease": "CKD", "risk_factors_cervical_cancer": "Cervical",
      "hcv_egy_data": "HCV", "framingham": "Framingham", "diabetic_data": "Diabetic"}
def methods(ds):                          # Diabetic 的 w/o AR 口径是 ep200
    wo = "h256_t200_3_ep200" if ds == "diabetic_data" else "h256_t200_3_final"
    ms = [("SMOTE", "base", "smote"), ("TabKDE", "base", "tabkde"), ("CTGAN", "base", "ctgan"),
            ("TVAE", "base", "tvae"), ("TabDDPM", "base", "tabddpm"), ("TabSyn", "tabsyn", "tabsyn"),
            ("CAR-Diff w/o AR", "ours", wo), ("CAR-Diff", "ours", "h256_t200_3_abl_ar"),
            ("Legacy cat. input",  "ours", "h256_t200_3_abl_legacy"),
            ("No interaction",     "ours", "h256_t200_3_abl_nointeract"),
            ("Sampled cat. input", "ours", "h256_t200_3_abl_sampled"),
            ("No shared decoder",  "ours", "h256_t200_3_abl_nosharedec")]
    if ds != "diabetic_data":
        ms.append(("CAR-Diff w/o prior init", "ours", "h256_t200_3_abl_ar_noprior"))
    return ms
THRS, MIN_CNT, CAP = (0.02, 0.05, 0.10), 3, 20000      # 稀有阈值、稀有取值最小训练计数、合成数据每次采样最多行数

def fake_paths(kind, name, ds, k):
    if kind == "ours": return [p for _, s, p in ddpm_fakes(DDPM_ROOT, ds, k, hp=name)]
    return [p for _, s, p in flat_fakes(TABSYN_ROOT if kind == "tabsyn" else DDPM_ROOT, name, ds, k)]

def nc(s):                                 # 类别值统一成字符串（1.0 与 1 视为同一类）
    t = s.astype(str).str.strip(); n = pd.to_numeric(t, errors="coerce")
    return t.where(n.isna(), n.round(6).astype(str))
def prep(df, cat):
    df = df.copy(); df.columns = df.columns.str.strip()
    return pd.DataFrame({c: nc(df[c]) for c in cat})
def keys(P): return pd.util.hash_pandas_object(P, index=False).values

combo, rare = [], []
for ds, dname in DS.items():
    mt = load_meta(INFO_PATH, ds); lab = mt["label_col"]
    cat = [c for c in mt["cat_cols"] if c != lab]
    if not cat: continue
    for k in range(5):
        tr, te = load_fold(DDPM_ROOT, ds, k); Ptr, Pte = prep(tr, cat), prep(te, cat)
        trk = keys(Ptr); base = np.isin(keys(Pte), trk).mean()
        vcs = {c: Ptr[c].value_counts() for c in cat}
        rv = {t: {c: vc[(vc / len(Ptr) < t) & (vc >= MIN_CNT)].index for c, vc in vcs.items()} for t in THRS}
        for m, kind, name in methods(ds):
            for p in fake_paths(kind, name, ds, k):
                f = pd.read_csv(p)
                if len(f) > CAP: f = f.sample(CAP, random_state=0)
                P = prep(f, cat); n = len(P); eps = 0.5 / n
                combo.append(dict(dataset=dname, fold=k, method=m, in_train=np.isin(keys(P), trk).mean(), test_base=base))
                fq = {c: P[c].value_counts(normalize=True) for c in cat}
                for t in THRS:
                    lr, cv = [], []
                    for c in cat:
                        idx = rv[t][c]
                        if len(idx) == 0: continue
                        pr = vcs[c].reindex(idx) / len(Ptr); pf = fq[c].reindex(idx).fillna(0)
                        lr += list(np.log2((pf + eps) / pr)); cv += list((pf > 0).astype(float))
                    if lr: rare.append(dict(dataset=dname, fold=k, method=m, thr=t, n_rare=len(lr),
                                            bias=np.mean(lr), mabs=np.mean(np.abs(lr)), cov=np.mean(cv)))
    print("done", ds, flush=True)

def summ(rows, keys_, cols):              # 样本取均值 → 折级 → 跨折 mean/std
    df = pd.DataFrame(rows); g = df.groupby(["dataset", "method", "fold"] + keys_)[cols].mean().reset_index()
    a = g.groupby(["dataset", "method"] + keys_)[cols]
    return a.mean().join(a.std(ddof=1), rsuffix="_std").reset_index()
cs = summ(combo, [], ["in_train", "test_base"]); cs["gap"] = cs["in_train"] - cs["test_base"]
cs.round(4).to_csv(OUT / "cat_combo.csv", index=False)
rs = summ(rare, ["thr"], ["n_rare", "bias", "mabs", "cov"])
rs.round(4).to_csv(OUT / "cat_rare.csv", index=False)
print(cs.pivot(index="method", columns="dataset", values="gap").round(3))
print(rs[rs.thr == 0.05].pivot(index="method", columns="dataset", values="bias").round(2))