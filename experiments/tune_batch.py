# experiments/tune_batch.py
"""
项目根目录运行：
  python -m experiments.tune_batch <stage> [--tags ...] [--datasets ...] [--folds ...] [--eval | --eval-only] [--redo]

阶段与运行位置（训练在各自目录里进行，评估一律回主目录用 --eval-only）：
  主目录 ddpm_med         : final | abl_main | stab | diab_epochs | diab_abl_main | diab_stab
  ddpm_med_bk_cat         : abl_cat | diab_abl_cat      （备份 backup-cat-switches；开关在主目录里不存在，放错目录会报错）
  ddpm_med_bk_ar          : abl_ar  | diab_abl_ar | stab_ar    （备份 backup-slim-ar）
已训练完成的 (数据集, 折, tag) 自动跳过（--redo 强制重训）。
"""
import argparse, subprocess, sys
from pathlib import Path
from utils.common import SEEDS

PY       = sys.executable
LAUNCH   = [PY, "-m", "experiments.run_cv"]
RESAMPLE = [PY, "-m", "experiments.resample_ours"]
EVAL     = [PY, "-m", "evaluation.run"]

# ---- 与 run_cv.py / ModelConfig 的默认值一致；hp 名由这三项生成：h{dim_hidden}_t{num_timesteps}_{num_layers} ----
BASE       = dict(dim_hidden=256, num_timesteps=200, num_layers=3)
MAIN_TRAIN = "target_epochs=200,enable_compile=True"   # 变体的 train 覆盖拼在后面，同名键后者生效
MAIN_FLOOR = 0
MAIN_SEED  = 42
N_SAMPLES  = 5
DEVICE     = "cuda"

DS_SMALL  = ["hepatitis", "chronic_kidney_disease", "risk_factors_cervical_cancer",
             "hcv_egy_data", "framingham"]            # 按规模从小到大
DS_DIAB   = ["diabetic_data"]
ALL_FOLDS = [0, 1, 2, 3, 4]

# tag: (训练种子, --model 覆盖项, --train 覆盖项)；tag 只能含字母数字下划线
FINAL = {"final": (MAIN_SEED, "", "")}

ABL_CAT = {      # 在 ddpm_med_bk_cat 中训练
    "abl_legacy":     (MAIN_SEED, "", ""),
    "abl_nointeract": (MAIN_SEED, "use_num_from_cat=False,use_cat_from_num=False,use_cat_self_attn=False", ""),
    "abl_sampled":    (MAIN_SEED, "cat_input_mode=sampled", ""),
}
ABL_AR = {       # 在 ddpm_med_bk_ar 中训练
    "abl_ar": (MAIN_SEED, "cat_ar=True", ""),
}
ABL_MAIN = {     # 在主目录训练；训练日志里的参数量必须与 final 不同，否则开关未生效
    "abl_nosharedec": (MAIN_SEED, "share_decoder=False", ""),
}
STAB = {         # 超参数稳定性：一次只改一个，默认值复用 final
    "dh128": (MAIN_SEED, "dim_hidden=128", ""),
    "dh512": (MAIN_SEED, "dim_hidden=512", ""),
    "nt100": (MAIN_SEED, "num_timesteps=100", ""),
    "nt500": (MAIN_SEED, "num_timesteps=500", ""),
    "nl2":   (MAIN_SEED, "num_layers=2", ""),
    "nl4":   (MAIN_SEED, "num_layers=4", ""),
    "ep100": (MAIN_SEED, "", "target_epochs=100"),
    "ep400": (MAIN_SEED, "", "target_epochs=400"),
}
STAB_DIAB = {k: v for k, v in STAB.items() if not k.startswith("ep")}   # diabetic 的 epochs 由 diab_epochs 覆盖
STAB_AR = {      # 在 ddpm_med_bk_ar 中训练；AR 头 + 单因素变化，参照为 abl_ar
    f"ar_{k}": (s, ",".join(filter(None, ["cat_ar=True", m])), t)
    for k, (s, m, t) in STAB.items()
}
DIAB_EPOCHS = {  # diabetic_data 训练预算：ep200 为统一协议的正式结果；ep50 即 final 标签；ep12≈旧 10600 步
    "ep12":  (MAIN_SEED, "", "target_epochs=12"),
    "ep200": (MAIN_SEED, "", "target_epochs=200"),
}

STAGES = {   # 阶段: (变体表, 数据集, 默认 folds)
    "final":         (FINAL,       DS_SMALL, ALL_FOLDS),
    "abl_cat":       (ABL_CAT,     DS_SMALL, ALL_FOLDS),
    "abl_ar":        (ABL_AR,      DS_SMALL, ALL_FOLDS),
    "abl_main":      (ABL_MAIN,    DS_SMALL, ALL_FOLDS),
    "stab":          (STAB,        DS_SMALL, ALL_FOLDS),
    "diab_epochs":   (DIAB_EPOCHS, DS_DIAB,  ALL_FOLDS),
    "diab_abl_cat":  (ABL_CAT,     DS_DIAB,  ALL_FOLDS),
    "diab_abl_ar":   (ABL_AR,      DS_DIAB,  ALL_FOLDS),
    "diab_abl_main": (ABL_MAIN,    DS_DIAB,  ALL_FOLDS),
    "diab_stab":     (STAB_DIAB,   DS_DIAB,  [0]),      # 先 fold 0 做方向性验证，需要时 --folds 1 2 追加
    "stab_ar":       (STAB_AR,     DS_SMALL, ALL_FOLDS),
}


def hp_of(model: str) -> str:
    kv = dict(BASE)
    for item in filter(None, model.split(",")):
        k, v = item.split("=", 1)
        if k.strip() in kv:
            kv[k.strip()] = int(v)
    return f"h{kv['dim_hidden']}_t{kv['num_timesteps']}_{kv['num_layers']}"


def is_done(ds, fold, hp, tag) -> bool:
    """结果目录里已有 synthetic_data.csv 与 ≥ N_SAMPLES-1 个 synthetic_data_s*.csv 视为训练+采样完成。"""
    base = Path("results") / ds / f"seed_{SEEDS[0]}_fold_{fold}"
    for d in base.glob(f"full_data_{hp}_{tag}_*"):
        if (d / "synthetic_data.csv").exists() and len(list(d.glob("synthetic_data_s*.csv"))) >= N_SAMPLES - 1:
            return True
    return False


def run_tag(tag, seed, model, train, datasets, folds, redo=False):
    hp = hp_of(model)
    train_all = ",".join(filter(None, [MAIN_TRAIN, train]))
    for ds in datasets:
        for k in folds:
            if not redo and is_done(ds, k, hp, tag):
                print(f"[skip] {ds} fold{k} {hp}_{tag} 已完成", flush=True)
                continue
            print(f"[train] {ds} fold{k} tag={tag} hp={hp}", flush=True)
            cmd = LAUNCH + ["--dataset", ds, "--fold", str(k), "--seed", str(seed),
                            "--step_floor", str(MAIN_FLOOR), "--tag", tag, "--train", train_all]
            if model:
                cmd += ["--model", model]
            subprocess.run(cmd, check=True)
            subprocess.run(RESAMPLE + ["--dataset", ds, "--fold", str(k),
                                       "--hp", f"{hp}_{tag}", "--device", DEVICE], check=True)


def evaluate(tag_models, datasets, folds):
    """评估必须在主目录（新指标版 evaluation）运行；对比方法与 real 行已评估过的会自动跳过。"""
    for ds in datasets:
        for tag, model in tag_models.items():
            subprocess.run(EVAL + ["--dataset", ds, "--fold", *map(str, folds),
                                   "--hp", f"{hp_of(model)}_{tag}", "--baselines"], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--tags", nargs="*", default=None, help="只跑指定 tag；默认该阶段全部")
    ap.add_argument("--datasets", nargs="*", default=None, help="只跑指定数据集；默认该阶段全部")
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="只跑指定折；默认该阶段全部")
    ap.add_argument("--eval", action="store_true", help="训练+采样后接着评估（只应在主目录使用）")
    ap.add_argument("--eval-only", action="store_true", help="只评估，不训练")
    ap.add_argument("--redo", action="store_true", help="已完成的也重训")
    a = ap.parse_args()

    table, dss, folds = STAGES[a.stage]
    dss   = a.datasets or dss
    folds = a.folds if a.folds else folds
    tags  = a.tags or list(table)
    bad = [t for t in tags if t not in table]
    if bad:
        raise SystemExit(f"阶段 {a.stage} 中没有这些 tag: {bad}；可选: {list(table)}")

    if not a.eval_only:
        for tag in tags:
            seed, model, train = table[tag]
            run_tag(tag, seed, model, train, dss, folds, a.redo)
    if a.eval or a.eval_only:
        evaluate({t: table[t][1] for t in tags}, dss, folds)