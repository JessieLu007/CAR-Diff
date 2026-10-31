# quickstart.py
"""单数据集入口：训练 → 采样 → 评估，不依赖五折目录结构。

两种输入方式：
  A) 数据集名称：--dataset <名称>，读取 data/<名称>.csv（作为训练集）与 data/datasets_info.json 中同名条目
  B) 自定义文件：--train train.csv --spec spec.json（spec 格式同 datasets_info.json 的一个条目）
spec 只用到：numerical、categorical、target（可省略）、minority_class（可省略，缺省取训练集最少类）。
target 给出时自动并入类别列并联合建模，评估时用于 tstr / shap。--test 可选，仅用于评估。

命令行：
    python quickstart.py fit       --dataset hepatitis --test test.csv
    python quickstart.py sample    --dataset hepatitis --repeats 5
    python quickstart.py eval      --dataset hepatitis
    python quickstart.py run       --train train.csv --test test.csv --spec spec.json --out runs/demo
    python quickstart.py baselines --dataset hepatitis --methods smote ctgan --n-samples 5
（--dataset 模式下 --out 缺省为 runs/<名称>；对比方法输出在 <out>/baselines/<方法>_s<j>.csv，
  评估用 eval --fake <该文件>）

常用覆盖项：--epochs 200  --seed 42  --model "dim_hidden=128"  --training "batch_size=32"

说明：无验证集，固定训练预算（总步数 = epochs × ⌊N / batch_size⌋），取最终权重；类别列采用自回归类别头。
评估需要 shap、xgboost、lightgbm、synthcity（仅 eval 时导入）；baselines 需要对应方法的依赖，
tabkde 另需先运行 third_party/tabkde/setup_tabkde.sh。请在项目根目录运行。
"""
from __future__ import annotations

import os
import sys


def _fix_omp() -> None:
    """必须在导入 numpy / torch 之前：OMP_NUM_THREADS 非法（空、0、非数字）时改为合法值。"""
    v = os.environ.get("OMP_NUM_THREADS")
    if v is None:
        return
    try:
        ok = int(v.strip()) >= 1
    except ValueError:
        ok = False
    if not ok:
        os.environ["OMP_NUM_THREADS"] = str(min(8, os.cpu_count() or 1))
        print(f"[quickstart] OMP_NUM_THREADS={v!r} 不合法，已改为 {os.environ['OMP_NUM_THREADS']}")


_fix_omp()
if sys.platform.startswith("win"):
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # Windows 上 Inductor 有路径 bug，禁用 compile

import argparse
import ast
import json
import pickle
import random
import re
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from configs.experiment_configs import make_default_experiment_config
from core.data_processor import TabularDataProcessor
from core.denoise_model_v2 import TabMLPDenoise
from experiments.model_runner import (
    build_dataloader, build_meta_info, generate_samples, load_run, save_run, train_model,
)

NA_VALUES = ["?", "", " ", "NA", "nan", "None"]
_DETAIL = re.compile(r"^(colshape_|cpt_pair_|c2st_fimp_)")
KEY_METRICS = ["shape_err", "cpt_err_all", "corr_diff_cc", "c2st_auc", "alpha_precision",
               "beta_recall", "dcr_ratio", "dcr_exact", "tstr_roc_auc"]
BASELINE_METHODS = ["smote", "ctgan", "tvae", "tabddpm", "tabkde"]


# ─────────────────────────── 工具 ───────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_kv(s: str) -> dict:
    out = {}
    for item in filter(None, (s or "").split(",")):
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


def read_table(path) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    return df.apply(
        lambda s: s.str.replace(r"^[\s\u00A0\u3000]+|[\s\u00A0\u3000]+$", "", regex=True)
        if s.dtype == object else s
    )


def load_spec(spec) -> dict:
    if isinstance(spec, (str, Path)):
        spec = json.loads(Path(spec).read_text(encoding="utf-8"))
    num = list(spec["numerical"])
    cat = list(spec.get("categorical", []))
    target = spec.get("target")
    if target and target not in cat:
        cat.append(target)
    if set(num) & set(cat):
        raise ValueError("同一列不能同时出现在 numerical 和 categorical 中")
    return dict(numerical=num, categorical=cat, target=target, minority_class=spec.get("minority_class"))


def dataset_inputs(name: str, data_dir=None):
    """--dataset 模式：返回 (csv 路径, datasets_info.json 中同名条目)。"""
    d = Path(data_dir) if data_dir else ROOT / "data"
    info = json.loads((d / "datasets_info.json").read_text(encoding="utf-8"))
    if name not in info:
        raise KeyError(f"{d / 'datasets_info.json'} 中没有数据集 {name!r}")
    csv = d / f"{name}.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    return csv, info[name]


def _restrict(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    need = set(spec["numerical"]) | set(spec["categorical"])
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"数据中缺少这些列: {sorted(missing)}")
    return df[[c for c in df.columns if c in need]].copy()


def _as_df(x) -> pd.DataFrame:
    return x.copy() if isinstance(x, pd.DataFrame) else read_table(x)


def _layout(df: pd.DataFrame, spec: dict):
    cols = df.columns.tolist()
    num_idx = [i for i, c in enumerate(cols) if c in set(spec["numerical"])]
    cat_idx = [i for i, c in enumerate(cols) if c in set(spec["categorical"])]
    return cols, num_idx, cat_idx


# ─────────────────────────── 预处理（补全） ───────────────────────────
def prepare(train, spec, out_dir, test=None, seed: int = 42):
    """补全缺失值（数值 MICE、类别众数），保存 train_imputed.csv / test_imputed.csv / spec.json / imputer.pkl。
    训练与评估、对比方法共用补全后的训练集。返回 (spec, train_imp, cols, num_idx, cat_idx)。"""
    spec = load_spec(spec)
    train = _restrict(_as_df(train), spec)
    test = _restrict(_as_df(test), spec) if test is not None else None
    cols, num_idx, cat_idx = _layout(train, spec)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imputer = TabularDataProcessor(num_idx, cat_idx, "quantile", "ordinal", random_state=seed).fit(train)
    train_imp = imputer.transform_readable(train)
    train_imp.to_csv(out / "train_imputed.csv", index=False)
    if test is not None:
        imputer.transform_readable(test).to_csv(out / "test_imputed.csv", index=False)
    (out / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "imputer.pkl", "wb") as f:
        pickle.dump(imputer, f)
    return spec, train_imp, cols, num_idx, cat_idx


# ─────────────────────────── 训练 ───────────────────────────
def fit(train, spec, out_dir, test=None, epochs: int = 200, seed: int = 42,
        model_kv: dict | None = None, train_kv: dict | None = None, compile: bool | None = None):
    """在训练集上拟合并保存到 out_dir。test 仅用于保存补全后的测试集（评估用），不参与训练。"""
    set_seed(seed)
    out = Path(out_dir)
    spec, train_imp, cols, num_idx, cat_idx = prepare(train, spec, out, test, seed)

    # 在补全后的训练集上重新拟合预处理器（分位数→正态、有序编码），用于训练与逆变换
    processor = TabularDataProcessor(num_idx, cat_idx, "quantile", "ordinal", random_state=seed).fit(train_imp)

    null_idx = int(train_imp[spec["target"]].nunique()) if spec["target"] else 2
    use_compile = compile if compile is not None else (
        torch.cuda.is_available() and not sys.platform.startswith("win"))
    cfg = make_default_experiment_config().with_overrides(
        model={"cat_ar": True, "null_class_idx": null_idx, **(model_kv or {})},
        training={"target_epochs": epochs, "use_early_stop": False,
                  "enable_compile": use_compile, **(train_kv or {})},
        pipeline={"random_seed": seed, "save_base_dir": str(out)},
    )
    if len(train_imp) < cfg.training.batch_size:
        raise ValueError(f"训练样本数 {len(train_imp)} 小于 batch_size {cfg.training.batch_size}，"
                         f"请用 --training batch_size=... 调小")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y = torch.full((len(train_imp),), null_idx, dtype=torch.long)     # full_data：条件恒为 null
    loader = build_dataloader(processor, train_imp, y, cfg.training.batch_size, shuffle=True,
                              device=dev, last_batch_strategy=cfg.training.last_batch_strategy)
    meta_info = build_meta_info(processor, train_imp)

    diffusion = train_model(TabMLPDenoise, cfg.model, cfg.training, cfg.sampling, loader, None, meta_info)

    target_dist = {"target_class": None, "target_count": 0, "background_count": len(train_imp),
                   "total": len(train_imp), "IR": float("inf")}
    save_run(str(out), processor, meta_info, diffusion, target_dist, spec["target"] or "",
             processor.cat_cols, train_imp.columns.tolist(), cols, cfg, TabMLPDenoise)
    print(f"[fit] 完成，已保存到 {out}")
    return out


# ─────────────────────────── 采样 ───────────────────────────
def sample(out_dir, num_rows: int | None = None, seed: int = 0, repeats: int = 1) -> list[Path]:
    """用已训练的模型采样；默认行数 = 训练集行数。输出 synthetic_s0.csv ... s{repeats-1}.csv"""
    out = Path(out_dir)
    b = load_run(str(out))
    diffusion, processor, cfg = b["diffusion"], b["processor"], b["exp_cfg"]
    samp = replace(cfg.sampling, num_samples=num_rows) if num_rows else cfg.sampling
    paths = []
    for j in range(repeats):
        set_seed(seed + j)
        df = generate_samples(diffusion, processor, samp, "full_data", None,
                              b["target_dist"], b["target_col"], b["original_columns"])
        p = out / f"synthetic_s{j}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
        print(f"[sample] {p}  rows={len(df)}")
    return paths


# ─────────────────────────── 对比方法 ───────────────────────────
def _write_tabkde_info(name: str, train_imp: pd.DataFrame, spec: dict, out: Path) -> None:
    """生成 data/Info/<name>.json（TabKDE 读取的列声明；标签列须在 cat_col_idx 中）。"""
    cols = train_imp.columns.tolist()
    tgt = spec["target"]
    t = cols.index(tgt)
    info = {
        "name": name,
        "task_type": "binclass" if train_imp[tgt].nunique() == 2 else "multiclass",
        "header": 0,
        "column_names": None,
        "num_col_idx": [i for i, c in enumerate(cols) if c in set(spec["numerical"])],
        "cat_col_idx": [i for i, c in enumerate(cols) if c in set(spec["categorical"]) and c != tgt] + [t],
        "target_col_idx": [t],
        "file_type": "csv",
        "data_path": (out / "train_imputed.csv").resolve().as_posix(),
        "test_path": ((out / "test_imputed.csv") if (out / "test_imputed.csv").exists()
                      else (out / "train_imputed.csv")).resolve().as_posix(),
    }
    d = ROOT / "data" / "Info"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(info, indent=4, ensure_ascii=False), encoding="utf-8")


def baselines(train, spec, out_dir, test=None, methods=None, n_samples: int = 5, seed: int = 42,
              ddpm_lr: float | None = None, ctgan_bs: int = 500, force: bool = False) -> Path:
    """用 evaluation/gen_baselines.py 的 GEN 函数在补全后的训练集上训练各对比方法，
    各采样 n_samples 次，输出 <out>/baselines/<方法>_s<j>.csv（已按训练集格式对齐）。"""
    from evaluation import gen_baselines as G
    from evaluation.align import make_aligner

    out = Path(out_dir)
    methods = methods or BASELINE_METHODS
    if (out / "train_imputed.csv").exists() and (out / "spec.json").exists():
        spec = load_spec(out / "spec.json")
        train_imp = pd.read_csv(out / "train_imputed.csv")
    else:
        set_seed(seed)
        spec, train_imp, *_ = prepare(train, spec, out, test, seed)
    test_imp = pd.read_csv(out / "test_imputed.csv") if (out / "test_imputed.csv").exists() else None

    tgt = spec["target"]
    meta = dict(num_cols=spec["numerical"], cat_cols=spec["categorical"],
                label_col=tgt, minority_val=spec.get("minority_class"))
    if ddpm_lr:
        G.DDPM_EXTRA["lr"] = ddpm_lr
    align = make_aligner(train_imp, meta)
    d = out / "baselines"
    d.mkdir(exist_ok=True)
    name = out.resolve().name

    for m in methods:
        js = [j for j in range(n_samples) if force or not (d / f"{m}_s{j}.csv").exists()]
        if not js:
            print(f"[skip] {m}")
            continue
        if m in ("smote", "tabkde") and not tgt:
            print(f"[fail] {m} 需要 spec 中声明 target")
            continue
        print(f"[baselines] {m} samples={js}", flush=True)
        try:
            if m == "tabkde":
                _write_tabkde_info(name, train_imp, spec, out)
                frames = G.GEN[m](train_imp, test_imp if test_imp is not None else train_imp, meta, name, seed, js)
            elif m == "ctgan":
                frames = G.gen_ctgan(train_imp, test_imp, meta, name, seed, js, ctgan_bs)
            else:
                frames = G.GEN[m](train_imp, test_imp, meta, name, seed, js)
            frames = [align(f) for f in frames]
        except Exception:
            print(f"[fail] {m}")
            traceback.print_exc()
            continue
        for j, df in zip(js, frames):
            df.to_csv(d / f"{m}_s{j}.csv", index=False)
        print(f"[done] {m}  rows={len(frames[0])}")
    return d


# ─────────────────────────── 评估 ───────────────────────────
def evaluate(out_dir, fake_path=None) -> pd.DataFrame:
    """评估一份合成数据。有测试集时给出 ref=train / test / both / real 四组结果，没有则只有 ref=train。
    结果写入 out_dir/eval_<文件名>.csv（汇总）与 eval_detail_<文件名>.csv（逐列/逐对明细）。"""
    from evaluation import metrics as M

    out = Path(out_dir)
    spec = json.loads((out / "spec.json").read_text(encoding="utf-8"))
    fake_path = Path(fake_path) if fake_path else out / "synthetic_s0.csv"
    train = pd.read_csv(out / "train_imputed.csv")
    test = pd.read_csv(out / "test_imputed.csv") if (out / "test_imputed.csv").exists() else None
    fake = pd.read_csv(fake_path)

    target = spec["target"]
    minority = spec.get("minority_class")
    if target and minority is None:
        minority = train[target].value_counts().idxmin()
    config = M.EvalConfig(default_mode="all", minority_col=target, minority_val=minority)
    kw = dict(num_cols=spec["numerical"], cat_cols=spec["categorical"], config=config)
    M._REF_CORR.clear()

    fid = {"shape": M.metric_shape, "cpt": M.metric_cpt, "corr": M.metric_corr_diff, "c2st": M.metric_c2st,
           "prdc": M.metric_prdc, "ab": M.metric_alpha_beta, "shap": M.metric_shap}
    rows, details = [], []

    def add(ref, res):
        rows.append({"ref": ref, **{k: v for k, v in res.items() if not _DETAIL.match(k)}})
        det = {k: v for k, v in res.items() if _DETAIL.match(k)}
        if det:
            details.append({"ref": ref, **det})

    refs = [("train", train)] + ([("test", test)] if test is not None else [])
    for name, ref in refs:
        res = {}
        for n, f in fid.items():
            if n == "shap" and (name == "test" or not target):
                continue                                    # shap 只在 ref=train 上算，且需要标签列
            print(f"[eval] ref={name} {n}", flush=True)
            res.update(f(ref_df=ref, fake_df=fake, **kw))
        add(name, res)

    if test is not None:
        dcr_real = M.holdout_dcr(train, test, **kw)
        res = M.metric_dcr(train_df=train, test_df=test, fake_df=fake, dcr_real=dcr_real, **kw)
        if target:
            res.update(M.metric_tstr(train_df=train, test_df=test, fake_df=fake, **kw))
        add("both", res)
        if target:
            add("real", M.metric_real(train, test, dcr_real=dcr_real, **kw))
        else:
            add("real", {"dcr_mean": float(dcr_real.mean()), "dcr_exact": float((dcr_real <= 1e-9).mean())})

    summary = pd.DataFrame(rows)
    stem = fake_path.stem
    summary.to_csv(out / f"eval_{stem}.csv", index=False)
    if details:
        pd.DataFrame(details).to_csv(out / f"eval_detail_{stem}.csv", index=False)

    show = ["ref"] + [c for c in KEY_METRICS if c in summary.columns]
    print("\n" + summary[show].round(4).to_string(index=False))
    print(f"[eval] 已保存到 {out / f'eval_{stem}.csv'}")
    return summary


# ─────────────────────────── 命令行 ───────────────────────────
def _compile_flag(s: str):
    return {"auto": None, "on": True, "off": False}[s]


def _out(a) -> str:
    if a.out:
        return a.out
    if a.dataset:
        return f"runs/{a.dataset}"
    raise SystemExit("需要 --out，或使用 --dataset（输出到 runs/<名称>）")


def _source(a):
    """返回 (train, spec)。"""
    if a.dataset:
        return dataset_inputs(a.dataset, a.data_dir)
    if not (a.train and a.spec):
        raise SystemExit("需要 --dataset，或同时给出 --train 与 --spec")
    return a.train, a.spec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def out_args(p):
        p.add_argument("--dataset", default=None, help="数据集名称（仅用于缺省输出目录 runs/<名称>）")
        p.add_argument("--out", default=None)

    def src_args(p):
        out_args(p)
        p.add_argument("--data-dir", default=None, help="默认 <项目根>/data")
        p.add_argument("--train", default=None)
        p.add_argument("--test", default=None, help="可选；仅用于评估，不参与训练")
        p.add_argument("--spec", default=None)

    def fit_args(p):
        p.add_argument("--epochs", type=int, default=200)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--model", default="", help='覆盖模型超参，如 "dim_hidden=128,num_layers=2"')
        p.add_argument("--training", default="", help='覆盖训练超参，如 "batch_size=32"')
        p.add_argument("--compile", choices=["auto", "on", "off"], default="auto")

    def sample_args(p):
        p.add_argument("--num-rows", type=int, default=None, help="默认等于训练集行数")
        p.add_argument("--repeats", type=int, default=1)
        p.add_argument("--sample-seed", type=int, default=0)

    p = sub.add_parser("fit");       src_args(p); fit_args(p)
    p = sub.add_parser("sample");    out_args(p); sample_args(p)
    p = sub.add_parser("eval");      out_args(p); p.add_argument("--fake", default=None)
    p = sub.add_parser("run");       src_args(p); fit_args(p); sample_args(p)
    p = sub.add_parser("baselines"); src_args(p)
    p.add_argument("--methods", nargs="+", choices=BASELINE_METHODS, default=BASELINE_METHODS)
    p.add_argument("--n-samples", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ddpm-lr", type=float, default=None, help="tabddpm 学习率；diabetic_data 需 1e-4")
    p.add_argument("--ctgan-bs", type=int, default=500, help="须为 10 的倍数")
    p.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = _out(a)
    if a.cmd in ("fit", "run"):
        train, spec = _source(a)
        fit(train, spec, out, test=a.test, epochs=a.epochs, seed=a.seed,
            model_kv=parse_kv(a.model), train_kv=parse_kv(a.training), compile=_compile_flag(a.compile))
    if a.cmd in ("sample", "run"):
        paths = sample(out, a.num_rows, a.sample_seed, a.repeats)
    if a.cmd == "eval":
        evaluate(out, a.fake)
    if a.cmd == "run":
        evaluate(out, paths[0])        # run 只评估第一份采样；其余请用 eval --fake 指定
    if a.cmd == "baselines":
        train, spec = _source(a) if (a.dataset or a.train) else (None, None)
        baselines(train, spec, out, test=a.test, methods=a.methods, n_samples=a.n_samples,
                  seed=a.seed, ddpm_lr=a.ddpm_lr, ctgan_bs=a.ctgan_bs, force=a.force)


if __name__ == "__main__":
    main()