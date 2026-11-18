# scripts/make_cat_traits.py      python scripts/make_cat_traits.py
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from evaluation.paths import load_fold, load_meta
INFO_PATH = "data/datasets_info.json"
DS = {"hepatitis": "Hepatitis", "chronic_kidney_disease": "CKD", "risk_factors_cervical_cancer": "Cervical",
      "hcv_egy_data": "HCV", "framingham": "Framingham", "diabetic_data": "Diabetic"}
def nc(s):
    t = s.astype(str).str.strip(); n = pd.to_numeric(t, errors="coerce"); return t.where(n.isna(), n.round(6).astype(str))
def cv(a, b):                                  # 偏差校正 Cramér's V
    t = pd.crosstab(a, b); n = t.values.sum(); r, c = t.shape
    if r < 2 or c < 2: return np.nan
    p2 = chi2_contingency(t, correction=False)[0] / n
    p2c = max(0, p2 - (r - 1) * (c - 1) / (n - 1)); rc, cc = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    return np.sqrt(p2c / max(min(rc - 1, cc - 1), 1e-9))
rng = np.random.default_rng(0); rows = []
for ds, name in DS.items():
    mt = load_meta(INFO_PATH, ds); lab = mt["label_col"]; cat = [c for c in mt["cat_cols"] if c != lab]
    for k in range(5):
        tr, _ = load_fold(".", ds, k); tr.columns = tr.columns.str.strip()
        P = pd.DataFrame({c: nc(tr[c]) for c in cat}); n = len(P)
        key = lambda D: pd.util.hash_pandas_object(D, index=False).values
        trk = set(key(P))
        S = pd.DataFrame({c: rng.choice(P[c].values, n) for c in cat})        # 独立边际采样
        vs = [cv(P[a], P[b]) for i, a in enumerate(cat) for b in cat[i + 1:]]
        rows.append(dict(dataset=name, n_train=n, n_cat=len(cat), uniq_ratio=len(trk) / n,
                         mean_cramers_v=np.nanmean(vs) if vs else np.nan, indep_hit=np.mean([x in trk for x in key(S)])))
    print("done", name, flush=True)
r = pd.DataFrame(rows).groupby("dataset").mean().reindex(list(DS.values())); print(r.round(3))
r.round(4).to_csv("paper_out/tables/cat_traits.csv")