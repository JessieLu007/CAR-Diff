# 文件位置: experiments/resample_ours.py
"""
对已训练的模型重复采样，输出 synthetic_data.csv (j=0)、synthetic_data_s<j>.csv。
python -m experiments.resample_ours --dataset <ds> --fold 0            # 该折下所有 hp，各取最新运行
python -m experiments.resample_ours --dataset <ds> --hp h256_t200_3    # 指定 hp，所有折
python -m experiments.resample_ours --dataset <ds> --device cuda --js 1 2 3 4
不传 --dataset 则遍历 common.DATASETS；不传 --fold 则 0..4。
"""
import os, sys, ast, argparse
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 必须在 import torch 之前

ROOT = Path(__file__).resolve().parents[1]    # 项目根目录
sys.path.insert(0, str(ROOT))

import random
import numpy as np
import torch
import utils.common as common
from experiments.model_runner import load_run, generate_samples
from evaluation.paths import ddpm_runs, PREFIX

if _pre_args.device == "cpu":
    assert not torch.cuda.is_available(), "仍检测到 GPU，环境变量未在 import torch 前生效"
else:
    assert torch.cuda.is_available(), "指定了 --device cuda，但没有可用 GPU"


def set_all_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fname(j: int) -> str:
    return "synthetic_data.csv" if j == 0 else f"synthetic_data_s{j}.csv"


def _read_info(out_dir: Path):
    p, d = out_dir / "resample_info.txt", out_dir / "resample_device.txt"
    if not (p.exists() and d.exists()):
        return None
    try:
        info = dict(l.split("=", 1) for l in p.read_text().splitlines() if "=" in l)
        info["js"] = ast.literal_eval(info["js"])
        info["base_seed"] = int(info["base_seed"])
        info["device"] = d.read_text().strip().split(":")[0]
    except Exception:
        return None
    return info


def _done_js(out_dir: Path, device: str, base_seed: int | None):
    """与本次参数一致且文件存在的已采样 j 集合；无 info 或参数不一致返回 None。
    base_seed 为 None 时不比对该项。"""
    info = _read_info(out_dir)
    if info is None or info["device"] != device:
        return None
    if base_seed is not None and info["base_seed"] != base_seed:
        return None
    return {j for j in info["js"] if (out_dir / _fname(j)).exists()}


def resample_run(run_dir: Path, js, base_seed: int | None = None,
                 out_dir: Path | None = None, force: bool = False, device: str = "cpu"):
    out_dir = run_dir if out_dir is None else out_dir
    js = list(js)

    # 加载模型前的预检查：全部已采样则直接跳过
    if not force:
        done = _done_js(out_dir, device, base_seed)
        if done is not None and set(js) <= done:
            print(f"⏭️ {run_dir.name} js={js} 均已采样且参数一致，跳过 ({out_dir.name})")
            return

    bundle = load_run(str(run_dir))
    exp_cfg = bundle["exp_cfg"]
    assert exp_cfg.training.task == "full_data"
    base_seed = exp_cfg.pipeline.random_seed if base_seed is None else base_seed
    diffusion, processor = bundle["diffusion"], bundle["processor"]
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = str(diffusion.device).split(":")[0]
    matched = _done_js(out_dir, dev, base_seed)              # 用解析后的 base_seed 严格检查
    done = matched if matched is not None else set()
    if matched is None and _read_info(out_dir) is not None:
        print(f"⚠️ {out_dir.name} 现有 resample_info 与本次参数不一致，将按本次参数重写；"
              f"目录内不在 js={js} 中的旧 synthetic_data*.csv 不会被删除，与新 info 不匹配")
    skip = set() if force else done

    for j in js:
        if j in skip:
            print(f"⏭️ {run_dir.name} j={j} 已采样，跳过 ({out_dir.name}/{_fname(j)})")
            continue
        set_all_seed(base_seed + j)
        df_fake = generate_samples(
            diffusion, processor, exp_cfg.sampling, exp_cfg.training.task,
            None, bundle["target_dist"], bundle["target_col"], bundle["original_columns"],
        )
        df_fake.to_csv(out_dir / _fname(j), index=False)
        print(f"✅ {run_dir.name} j={j} seed={base_seed + j} [{diffusion.device}] "
              f"-> {out_dir.name}/{_fname(j)} ({len(df_fake)} 行)")

    (out_dir / "resample_device.txt").write_text(str(diffusion.device))
    (out_dir / "resample_info.txt").write_text(
        f"src={run_dir.name}\nbase_seed={base_seed}\njs={sorted(done | set(js))}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="名称或索引；不传则遍历 common.DATASETS")
    ap.add_argument("--fold", type=int, default=None, help="0-based；不传则 0..n_folds-1")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--hp", default=None, help="如 h256_t200_3；不传则该折下所有 hp")
    ap.add_argument("--seed", type=int, default=common.SEEDS[0], help="目录名中的 seed_<seed>")
    ap.add_argument("--js", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--base_seed", type=int, default=None)
    ap.add_argument("--suffix", default="", help="非空则写入新目录 full_data_<hp><suffix>_<ts>，不动原目录")
    ap.add_argument("--force", action="store_true", help="已采样的 synthetic_data*.csv 也重新采样覆盖")
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
                out_dir = d.parent / f"{PREFIX}{hp}{args.suffix}_{ts}" if args.suffix else None
                resample_run(d, args.js, args.base_seed, out_dir, args.force, args.device)


if __name__ == "__main__":
    main()