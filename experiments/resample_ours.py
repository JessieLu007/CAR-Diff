# 文件位置: experiments/resample_ours.py
"""
python -m experiments.resample_ours --dataset <ds> --fold 0            # 该折下所有 hp，各取最新运行
python -m experiments.resample_ours --dataset <ds> --hp h256_t200_3    # 指定 hp，所有折
python -m experiments.resample_ours --dataset <ds> --device cuda
不传 --dataset 则遍历 common.DATASETS；不传 --fold 则 0..4。
"""
import os, sys, re, argparse
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 必须在 import torch 之前

ROOT = Path(__file__).resolve().parents[1]    # ddpm_root
sys.path.insert(0, str(ROOT))

import random
import numpy as np
import torch
import utils.common as common
from experiments.model_runner import load_run, generate_samples
from evaluation.paths import ddpm_runs

if _pre_args.device == "cpu":
    assert not torch.cuda.is_available(), "仍检测到 GPU，环境变量未在 import torch 前生效"
else:
    assert torch.cuda.is_available(), "指定了 --device cuda，但没有可用 GPU"


def set_all_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resample_run(run_dir: Path, js, base_seed: int | None = None):
    bundle = load_run(str(run_dir))
    exp_cfg = bundle["exp_cfg"]
    assert exp_cfg.training.task == "full_data"
    base_seed = exp_cfg.pipeline.random_seed if base_seed is None else base_seed
    diffusion, processor = bundle["diffusion"], bundle["processor"]

    for j in js:
        set_all_seed(base_seed + j)
        df_fake = generate_samples(
            diffusion, processor, exp_cfg.sampling, exp_cfg.training.task,
            None, bundle["target_dist"], bundle["target_col"], bundle["original_columns"],
        )
        fname = "synthetic_data.csv" if j == 0 else f"synthetic_data_s{j}.csv"
        df_fake.to_csv(run_dir / fname, index=False)
        print(f"✅ {run_dir.name} j={j} seed={base_seed + j} [{diffusion.device}] -> {fname} ({len(df_fake)} 行)")
    (run_dir / "resample_device.txt").write_text(str(diffusion.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="名称或索引；不传则遍历 common.DATASETS")
    ap.add_argument("--fold", type=int, default=None, help="0-based；不传则 0..n_folds-1")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--hp", default=None, help="如 h256_t200_3；不传则该折下所有 hp")
    ap.add_argument("--seed", type=int, default=common.SEEDS[0], help="目录名中的 seed_<seed>")
    ap.add_argument("--js", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--base_seed", type=int, default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    datasets = list(common.DATASETS) if args.dataset is None else [common.resolve_dataset(args.dataset)]
    folds = [args.fold] if args.fold is not None else range(args.n_folds)

    for ds in datasets:
        for k in folds:
            runs = ddpm_runs(ROOT, ds, k, args.seed, args.hp,
                             require=("model.pt", "run_bundle.pkl"))
            if not runs:
                print(f"⚠️ 无有效运行: {ds} fold{k} hp={args.hp}")
                continue
            for ts, hp, d in runs:
                print(f"采用最新的训练模型进行采样：{hp}_{ts}")
                resample_run(d, args.js, args.base_seed)


if __name__ == "__main__":
    main()