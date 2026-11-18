# scripts/make_hcv_combo_diag.py      python scripts/make_hcv_combo_diag.py
import sys, os; sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd
from pathlib import Path
from evaluation.paths import load_fold, load_meta, flat_fakes, ddpm_fakes

DDPM_ROOT, INFO_PATH, TABSYN_ROOT = ".", "data/datasets_info.json", "../tabsyn-main"   # ← 按服务器实际位置修改
OUT = Path("paper_out/tables"); OUT.mkdir(parents=True, exist_ok=True)
DS = {"hcv_egy_data": "HCV"}                       # 想一并看 Diabetic 就加进来（会慢）
METHODS = [("TabKDE", "base", "tabkde"), ("TabSyn", "tabsyn", "tabsyn"),
           ("CAR-Diff w/o AR", "ours", "h256_t200_3_final"), ("CAR-Diff", "ours", "h256_t200_3_abl_ar"),
          ("AR ep100", "ours", "h256_t200_3_ar_ep100"), ("AR ep400", "ours", "h256_t200_3_ar_ep400")]
CAP = 20000

def fake_paths(kind, name, ds, k):
    if kind == "ours": return [p for _, s, p in ddpm_fakes(DDPM_ROOT, ds, k, hp=name)]
    return [p for _, s, p in flat_fakes(TABSYN_ROOT if kind == "tabsyn" else DDPM_ROOT, name, ds, k)]
def nc(s):
    t = s.astype(str).str.strip(); n = pd.to_numeric(t, errors="coerce")
    return t.where(n.isna(), n.round(6).astype(str))
def keys(df, cat):
    df = df.copy(); df.columns = df.columns.str.strip()
    P = pd.DataFrame({c: nc(df[c]) for c in cat})
    return pd.Series(pd.util.hash_pandas_object(P, index=False).values)

def dist_stats(kv, trc, tec):
    """kv: 合成数据的组合键；trc/tec: 训练集/测试集的组合计数(Series, index=键)"""
    vc = kv.value_counts(); p = vc / vc.sum(); pt = trc / trc.sum()
    u = p.index.union(pt.index)
    return dict(n_uniq=len(vc),
                hit=p[p.index.isin(trc.index)].sum(),               # 合成行中，组合在训练集出现过的比例
                new_distinct=(~vc.index.isin(trc.index)).mean(),    # 合成的不同组合中，训练集没有的比例
                cov_train=trc.index.isin(vc.index).mean(),          # 训练集不同组合被合成覆盖的比例
                cov_test=tec.index.isin(vc.index).mean(),           # 测试集不同组合被合成覆盖的比例
                overlap=np.minimum(p.reindex(u).fillna(0), pt.reindex(u).fillna(0)).sum(),  # 与训练组合分布的重合度
                entropy=-(p * np.log2(p)).sum(), top1=p.max())

rows = []
for ds, dname in DS.items():
    mt = load_meta(INFO_PATH, ds); lab = mt["label_col"]; cat = [c for c in mt["cat_cols"] if c != lab]
    for k in range(5):
        tr, te = load_fold(DDPM_ROOT, ds, k); ktr, kte = keys(tr, cat), keys(te, cat)
        trc, tec = ktr.value_counts(), kte.value_counts()
        # 参照行：训练集自身、测试集相对训练集
        pt = trc / trc.sum()
        rows.append(dict(dataset=dname, fold=k, method="Train", n_uniq=len(trc), hit=1.0, new_distinct=0.0,
                         cov_train=1.0, cov_test=tec.index.isin(trc.index).mean(), overlap=1.0,
                         entropy=-(pt * np.log2(pt)).sum(), top1=pt.max()))
        for m, kind, name in METHODS:
            for p in fake_paths(kind, name, ds, k):
                f = pd.read_csv(p)
                if len(f) > CAP: f = f.sample(CAP, random_state=0)
                rows.append(dict(dataset=dname, fold=k, method=m, **dist_stats(keys(f, cat), trc, tec)))
    print(dname, "类别列数", len(cat), flush=True)

df = pd.DataFrame(rows)
g = df.groupby(["dataset", "method", "fold"]).mean(numeric_only=True).reset_index()   # 先对采样取均值

a = g.drop(columns=["fold"]).groupby(["dataset", "method"]).mean(numeric_only=True)
res = a.join(g.groupby(["dataset", "method"]).std(ddof=1, numeric_only=True), rsuffix="_std").reset_index()
res.round(4).to_csv(OUT / "combo_diag.csv", index=False)
pd.set_option("display.width", 200)
print(a.round(3)[["n_uniq", "hit", "new_distinct", "cov_train", "cov_test", "overlap", "entropy", "top1"]])