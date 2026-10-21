# experiments/tune_batch.py
import argparse, subprocess, sys

PY       = sys.executable
LAUNCH   = [PY, "-m", "experiments.run_cv"]
RESAMPLE = [PY, "-m", "experiments.resample_ours"]
EVAL     = [PY, "-m", "evaluation.run"]

# ---- 与生成 main 时的命令保持一致（请核对）----
HP         = "h256_t200_3"
MAIN_TRAIN = "target_epochs=200,enable_compile=True"
MAIN_FLOOR = 0
MAIN_SEED  = 42
DEVICE     = "cuda"          # 重采样设备

# ---- 原有阶段使用 ----
DS3 = ["hepatitis", "chronic_kidney_disease", "hcv_egy_data"]
DS_EXTRA = ["risk_factors_cervical_cancer", "framingham"]
FOLDS = [0, 1]

# ---- 新阶段：数据集按规模从小到大 ----
DS_SCREEN = ["hepatitis", "chronic_kidney_disease", "framingham"]
DS_ABL = ["hepatitis", "chronic_kidney_disease", "risk_factors_cervical_cancer",
          "hcv_egy_data", "framingham"]
DS_FINAL = ["hepatitis", "chronic_kidney_disease", "risk_factors_cervical_cancer",
            "hcv_egy_data", "framingham"]

# tag: (训练种子, --model 覆盖项)
FINAL = {"final": (MAIN_SEED, "")}

NOISE = {
    "slim_s43": (43, ""),
    "slim_s44": (44, ""),
}
ABL = {
    "slim": (MAIN_SEED, ""),
    "slim2": (MAIN_SEED, ""),
    "nosharedec": (MAIN_SEED, "share_decoder=False"),
}
STAGES = {                    # 阶段: (变体表, 数据集, folds)
    "noise":  (NOISE, DS_SCREEN, [0, 1]),
    "abl01":  (ABL,   DS_ABL,    [0, 1]),
    "abl234": (ABL,   DS_ABL,    [2, 3, 4]),
    "final":(FINAL, DS_FINAL, [0, 1, 2, 3, 4]),
}


def run(tag, seed, ds_list, model="", train="", floor=0):      # 原有，未改动
    for ds in ds_list:
        for k in FOLDS:
            cmd = LAUNCH + ["--dataset", ds, "--fold", str(k), "--seed", str(seed),
                            "--step_floor", str(floor), "--tag", tag]
            if model: cmd += ["--model", model]
            if train: cmd += ["--train", train]
            subprocess.run(cmd, check=True)


def run_tag(tag, seed, model, datasets, folds):
    for ds in datasets:
        for k in folds:
            cmd = LAUNCH + ["--dataset", ds, "--fold", str(k), "--seed", str(seed),
                            "--step_floor", str(MAIN_FLOOR), "--tag", tag,
                            "--train", MAIN_TRAIN]
            if model: cmd += ["--model", model]
            subprocess.run(cmd, check=True)
            subprocess.run(RESAMPLE + ["--dataset", ds, "--fold", str(k),
                                       "--hp", f"{HP}_{tag}", "--device", DEVICE], check=True)


def evaluate(datasets, folds):
    for ds in datasets:
        subprocess.run(EVAL + ["--dataset", ds, "--fold", *map(str, folds)], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["epochs", "noise_extra", *STAGES])
    ap.add_argument("--tags", nargs="*", default=None, help="只跑指定 tag；默认该阶段全部")
    ap.add_argument("--datasets", nargs="*", default=None, help="只跑指定数据集；默认该阶段全部")
    ap.add_argument("--no-eval", action="store_true", help="只训练+采样，不评估")
    a = ap.parse_args()

    if a.stage == "epochs":
        run("e200", 42, DS3, train="target_epochs=200")
    elif a.stage == "noise_extra":
        run("base", 42, DS_EXTRA); run("base_s43", 43, DS_EXTRA); run("base_s44", 44, DS_EXTRA)
    else:
        table, dss, folds = STAGES[a.stage]
        dss  = a.datasets or dss
        tags = a.tags or list(table)
        for tag in tags:
            seed, model = table[tag]
            run_tag(tag, seed, model, dss, folds)
        if not a.no_eval:
            evaluate(dss, folds)
