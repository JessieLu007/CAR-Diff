# evaluation/diag_temp.py
import io, os
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .paths import load_fold, ddpm_runs, load_meta, flat_fakes
from .metrics import alpha_diag
from experiments.model_runner import load_run, generate_samples

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS, FOLD, HP = "framingham", 1, "h256_t200_3_target_epochs=200" #"h256_t200_3_final"
TEMPS, SAMPLE_SEEDS = [0.8, 1.0], range(5)          # 同一批种子配对比较两个温度
BASELINES = {m: ROOT for m in ["ctgan", "smote", "tabddpm", "tabkde", "tvae"]}
BASELINES["tabsyn"] = ROOT.parent / "tabsyn-main"   # 与 ddpm_med 平级

meta = load_meta(ROOT / "data" / "datasets_info.json", DS)
NUM, CAT = meta["num_cols"], meta["cat_cols"]
train, test = load_fold(ROOT, DS, FOLD)


def rare_ratio(ref, fake, thr=0.05):
    """真实频率 < thr 的类别，合成频率之和 / 真实频率之和（理想值 1）"""
    r = f = 0.0
    for c in CAT:
        pr = ref[c].astype(str).value_counts(normalize=True)
        pf = fake[c].astype(str).value_counts(normalize=True).reindex(pr.index).fillna(0)
        m = pr < thr
        r += pr[m].sum(); f += pf[m].sum()
    return f / r


def summarize(fake):
    r, f = train[NUM], fake[NUM]
    row = dict(std_ratio=(f.std() / r.std()).mean(),
               p95_diff=((f.quantile(.95) - r.quantile(.95)) / r.std()).mean(),
               rare_ratio=rare_ratio(train, fake))
    for name, ref in [("test", test), ("train", train)]:
        d = alpha_diag(ref, fake, NUM, CAT)
        row[f"alpha_{name}"] = d.loc["all", "alpha_precision"]
        row[f"signed_{name}"] = d.loc["all", "signed_mean"]
        if name == "test":
            row["num_test"] = d.loc["num_only", "alpha_precision"]
            row["cat_test"] = d.loc["cat_only", "alpha_precision"]
    return row


rows = []

# 1) 同一权重，温度 0.8 vs 1.0 重新采样

from .paths import SEED

base = Path(ROOT) / "results" / DS / f"seed_{SEED}_fold_{FOLD}"
print("目录存在:", base.exists(), base)
if base.exists():
    print("下面的运行目录:", sorted(p.name for p in base.iterdir()))

runs = ddpm_runs(ROOT, DS, FOLD, hp=None, latest=True)
print("可识别的 hp:", [h for _, h, _ in runs])

sel = [r for r in runs if r[1] == HP]
if not sel:
    raise SystemExit(f"未找到 hp={HP}；从上面的 hp 列表里选一个填到 HP")
run_dir = sel[0][2]
fake = pd.read_csv(run_dir / "synthetic_data.csv")

# (_, _, run_dir), = ddpm_runs(ROOT, DS, FOLD, hp=HP, latest=True)
bundle = load_run(str(run_dir))
dm, cfg = bundle["diffusion"], bundle["exp_cfg"]
for T in TEMPS:
    dm.sampling_cfg = replace(cfg.sampling, cat_temperature_cap=T)   # 温度从这里读取
    for s in SAMPLE_SEEDS:
        torch.manual_seed(s); np.random.seed(s)
        fake = generate_samples(dm, bundle["processor"], dm.sampling_cfg, cfg.training.task, None,
                                bundle["target_dist"], bundle["target_col"], bundle["original_columns"])
        #fake = pd.read_csv(io.StringIO(fake.to_csv(index=False)))    # 与落盘 csv 的 dtype 一致
        rows.append(dict(name=f"ours_T{T}", **summarize(fake)))

# 2) 基线（含各自的重复采样 s0..s4）
for m, root in BASELINES.items():
    for _, _, p in flat_fakes(root, m, DS, FOLD):
        rows.append(dict(name=m, **summarize(pd.read_csv(p))))

g = pd.DataFrame(rows).groupby("name", sort=False)
print("均值:\n", g.mean().round(3).to_string())
print("\n标准差(单次采样的方法为 NaN):\n", g.std().round(3).to_string())