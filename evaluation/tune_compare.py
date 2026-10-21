# evaluation/tune_compare.py
"""
python -m evaluation.tune_compare noise   --prefix ours_h256_t200_3 --tags base base_s43 base_s44 --datasets ... --folds 0 1
python -m evaluation.tune_compare compare --prefix ours_h256_t200_3 --base base --cands e200 floor3000 \
       --noise-tags base base_s43 base_s44 --datasets ... --folds 0 1
"""
import argparse, glob, os
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)

METRICS = {                       # 列名: 方向
    "shape_err":       "min",
    "cpt_err_all":     "min",
    "corr_diff_nn":    "min",     # 均为绝对差，越低越好
    "corr_diff_nc":    "min",
    "corr_diff_cc":    "min",
    "c2st_auc":        "abs0.5",  # |AUC-0.5| 越小越好
    "beta_recall":     "max",
    "alpha_precision": "max",
}
Z_CLIP = 3.0


def load(eval_root, datasets=None, folds=None):
    dfs = [pd.read_csv(p) for p in glob.glob(os.path.join(eval_root, "*_fold*", "summary.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    if datasets: df = df[df["dataset"].isin(datasets)]
    if folds is not None: df = df[df["fold"].isin(folds)]
    return df


def per_fold(df, methods, ref="test"):
    d = df[(df["ref"] == ref) & df["method"].isin(methods)].copy()
    for m, k in METRICS.items():
        d[m] = {"min": d[m], "max": -d[m], "abs0.5": (d[m] - 0.5).abs()}[k]
    return d.groupby(["dataset", "fold", "method"])[list(METRICS)].mean()


def noise_table(df, methods, ref="test"):
    sd = per_fold(df, methods, ref).groupby(["dataset", "fold"]).std(ddof=1)
    return sd.groupby("dataset").mean()


def compare(df, base, cands, noise, ref="test"):
    pf = per_fold(df, [base] + cands, ref)
    rows = []
    for c in cands:
        for ds in noise.index:
            try:
                b = pf.xs(base, level="method").loc[ds]
                x = pf.xs(c, level="method").loc[ds]
            except KeyError:
                continue
            idx = b.index.intersection(x.index)
            delta = (x.loc[idx] - b.loc[idx]).mean()          # 负值 = 候选更好
            for m in METRICS:
                nz = noise.loc[ds, m]
                if not nz > 0:                                 # 噪声为 0/NaN：此格无区分度，不参与
                    continue
                z = float(np.clip(delta[m] / nz, -Z_CLIP, Z_CLIP))
                rows.append(dict(cand=c, dataset=ds, metric=m, delta=delta[m], z=z,
                                 verdict="better" if delta[m] < -nz else "worse" if delta[m] > nz else "tie"))
    if not rows:
        raise SystemExit("没有匹配到候选配置的评估结果：请确认已对候选标签运行 evaluation.run")
    res = pd.DataFrame(rows)

    by_ds = res.groupby(["cand", "dataset"])["verdict"].agg(
        better=lambda s: (s == "better").sum(), worse=lambda s: (s == "worse").sum())
    by_ds["result"] = np.where((by_ds["better"] > 0) & (by_ds["worse"] == 0), "win",
                       np.where((by_ds["worse"] > 0) & (by_ds["better"] == 0), "loss", "mixed/tie"))

    summ = res.groupby("cand").agg(mean_z=("z", "mean"),
                                   better=("verdict", lambda s: (s == "better").sum()),
                                   tie=("verdict", lambda s: (s == "tie").sum()),
                                   worse=("verdict", lambda s: (s == "worse").sum()))
    summ["ds_win"] = by_ds.groupby("cand")["result"].apply(lambda s: (s == "win").sum())
    summ["ds_loss"] = by_ds.groupby("cand")["result"].apply(lambda s: (s == "loss").sum())
    return res, by_ds, summ.sort_values("mean_z")


def dcr_ratio(df, methods):
    """fake→train / holdout→train 的 Gower DCR 比值（metrics.metric_dcr 的 dcr_ratio）。"""
    b = df[(df["ref"] == "both") & df["method"].isin(methods)]
    return b.groupby(["dataset", "fold", "method"])["dcr_ratio"].mean().rename("r")


def dcr_guard(df, base, cands, noise_methods):
    r = dcr_ratio(df, [base] + cands + noise_methods)
    sd = r.loc[r.index.get_level_values("method").isin(noise_methods)] \
          .groupby(["dataset", "fold"]).std(ddof=1).groupby("dataset").mean()
    m = r.groupby(["dataset", "method"]).mean().unstack("method")
    flag = pd.DataFrame({c: np.where(m[c] < m[base] - 2 * sd.reindex(m.index), "疑似记忆", "ok")
                         for c in cands}, index=m.index)
    return m.round(3), flag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["noise", "compare"])
    ap.add_argument("--eval-root", default="eval")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--base")
    ap.add_argument("--cands", nargs="*", default=[])
    ap.add_argument("--tags", nargs="*", default=[])
    ap.add_argument("--noise-tags", nargs="*", default=[])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--folds", nargs="*", type=int)
    a = ap.parse_args()

    #full = lambda ts: [f"{a.prefix}_{t}" for t in ts]
    full = lambda ts: [t if t.startswith("ours_") else f"{a.prefix}_{t}" for t in ts]
    short = lambda x: x.replace(a.prefix + "_", "")
    df = load(a.eval_root, a.datasets, a.folds)

    if a.mode == "noise":
        print(noise_table(df, full(a.tags)).round(5))
    else:
        base, cands, nm = full([a.base])[0], full(a.cands), full(a.noise_tags)
        miss = [m for m in [base] + cands + nm if m not in set(df["method"])]
        if miss:
            raise SystemExit(f"评估结果中找不到这些方法名: {miss}")
        nz = noise_table(df, nm)
        miss = [d for d in nz.index if nz.loc[d].isna().all()]
        if miss:
            raise SystemExit(f"这些数据集缺少噪声（需 ≥2 个训练种子的评估结果）: {miss}")
        dead = [m for m in METRICS if not (nz[m] > 0).any()]
        if dead:
            print(f"⚠ 噪声全为 0（饱和/无区分度），已排除: {dead}")
        res, by_ds, summ = compare(df, base, cands, nz)
        res["cand"] = res["cand"].map(short)
        by_ds = by_ds.rename(index=short, level="cand"); summ = summ.rename(index=short)
        print("噪声(sd):\n", nz.round(5))
        print("\n明细:\n", res.round(4).to_string(index=False))
        print("\n数据集级判定(win=至少一项更好且无变差):\n", by_ds)
        print("\n汇总(mean_z 越负越好，z 已截断到 ±3):\n", summ.round(3))
        ratio, flag = dcr_guard(df, base, cands, nm)
        ratio.columns = [short(c) for c in ratio.columns]; flag.columns = [short(c) for c in flag.columns]
        print("\ndcr 比值(fake→train / holdout→train):\n", ratio); print("\n守卫:\n", flag)