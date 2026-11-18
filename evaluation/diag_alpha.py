# evaluation/diag_alpha.py
"""
用法示例：
    python -m evaluation.diag_alpha --dataset framingham --fold 2 --hp h256_t200_3_final
    python -m evaluation.diag_alpha --dataset framingham --fold 2 --hp h256_t200_3_final --sample 1
    python -m evaluation.diag_alpha --dataset framingham --fold 2   # 不传 hp 则列出可选的 hp
    python -m evaluation.diag_alpha --dataset framingham --fold 2 --hp h256_t200_3_final --seed 42
"""
import argparse
import os
from pathlib import Path
import pandas as pd
from evaluation.metrics import alpha_diag
from .paths import load_fold, ddpm_runs, load_meta, SEED


ROOT = Path(__file__).resolve().parents[1]   # ddpm_root


def spread_diag(ref_df, fake_df, num_cols, cat_cols, rare_thr=0.15):
    r, f = ref_df[num_cols], fake_df[num_cols]
    num = pd.DataFrame({
        'std_ratio': f.std() / r.std(),
        'p05_diff': (f.quantile(.05) - r.quantile(.05)) / r.std(),
        'p95_diff': (f.quantile(.95) - r.quantile(.95)) / r.std(),
    }).round(3)
    rows = []
    for c in cat_cols:
        pr = ref_df[c].astype(str).value_counts(normalize=True)
        pf = fake_df[c].astype(str).value_counts(normalize=True).reindex(pr.index).fillna(0)
        rows.append(pd.DataFrame({'col': c, 'cls': pr.index, 'real': pr.values, 'fake': pf.values}))
    cat = pd.concat(rows)
    cat['ratio'] = cat['fake'] / cat['real']
    return num, cat[cat['real'] < rare_thr].sort_values('real').round(3)


def find_run_dir(ds: str, fold: int, hp: str, seed: int):
    """按 hp 找 run 目录；hp=None 时返回 None 并把可选 hp 打印出来。"""
    base = ROOT / "results" / ds / f"seed_{seed}_fold_{fold}"
    print(f"[diag] 数据集={ds}  fold={fold}  seed={seed}  hp={hp}")
    print(f"[diag] run 根目录: {base}  (存在: {base.exists()})")
    if base.exists():
        print(f"[diag] 目录下运行项: {sorted(p.name for p in base.iterdir())}")

    runs = ddpm_runs(ROOT, ds, fold, seed=seed, hp=None, latest=True,
                     require=("synthetic_data.csv",))
    available = [h for _, h, _ in runs]
    print(f"[diag] 可识别的 hp: {available}")
    if hp is None:
        return None
    sel = [r for r in runs if r[1] == hp]
    if not sel:
        raise SystemExit(f"未找到 hp={hp}；请从上面的 hp 列表里选一个再传 --hp")
    return sel[0][2]


def main():
    ap = argparse.ArgumentParser(description="alpha_precision / 离散度 / 稀有类别诊断")
    ap.add_argument("--dataset", required=True, help="数据集名（原 DS），如 framingham")
    ap.add_argument("--fold", type=int, required=True, help="折编号（0-based，原 FOLD）")
    ap.add_argument("--hp", default=None,
                    help="超参标签（原 HP），如 h256_t200_3_final；不传则只列出可选项")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"目录名 seed_<seed>；默认 {SEED}")
    ap.add_argument("--sample", type=int, default=0,
                    help="第几次采样：0 → synthetic_data.csv；J>0 → synthetic_data_sJ.csv")
    ap.add_argument("--rare-thr", type=float, default=0.15,
                    help="稀有类别阈值（真实频率 < 该值才显示），默认 0.15")
    ap.add_argument("--info", default=None,
                    help="datasets_info.json 路径，默认 <ROOT>/data/datasets_info.json")
    args = ap.parse_args()

    info_path = Path(args.info) if args.info else ROOT / "data" / "datasets_info.json"
    meta = load_meta(info_path, args.dataset)

    train, test = load_fold(ROOT, args.dataset, args.fold)

    run_dir = find_run_dir(args.dataset, args.fold, args.hp, args.seed)
    if run_dir is None:
        print("[diag] 未指定 --hp，仅列出可用 hp 后退出。")
        return

    fname = "synthetic_data.csv" if args.sample == 0 else f"synthetic_data_s{args.sample}.csv"
    fake_path = run_dir / fname
    if not fake_path.exists():
        raise SystemExit(f"找不到合成样本文件: {fake_path}")
    print(f"[diag] 使用合成样本: {fake_path}")
    fake = pd.read_csv(fake_path)

    num_t, cat_t = spread_diag(train, fake,
                               meta["num_cols"], meta["cat_cols"],
                               rare_thr=args.rare_thr)
    print("\n数值列离散度:\n", num_t)
    print(f"\n稀有类别频率(real<{args.rare_thr}):\n", cat_t.to_string(index=False))

    for name, ref in [("test", test), ("train", train)]:
        print(f"\n== ref = {name} ==")
        print(alpha_diag(ref, fake, meta["num_cols"], meta["cat_cols"]))


if __name__ == "__main__":
    main()