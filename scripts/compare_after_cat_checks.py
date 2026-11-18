import pandas as pd
r = pd.read_csv("paper_out/tables/cat_rare.csv")
m = ["CAR-Diff", "CAR-Diff w/o prior init"]
for thr in (0.05, 0.10):
    t = r[(r.thr == thr) & r.method.isin(m)]
    print(thr)
    print(t.pivot(index="dataset", columns="method", values=["bias", "mabs", "cov"]).round(2))
    print(t.pivot(index="dataset", columns="method", values=["mabs_std"]).round(2))