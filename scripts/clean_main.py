# clean_main.py
import re, shutil, sys
from pathlib import Path
import pandas as pd

METHOD = "ours_h256_t200_3_main"
RUN_RE = re.compile(r"^full_data_h256_t200_3_main_\d{4}_\d{6}$")
apply = "--apply" in sys.argv
root = Path(".")

# 1) results 下 main 运行目录（含 synthetic_data*.csv、模型、日志）
for d in sorted((root / "results").glob("*/seed_*_fold_*/*")):
    if d.is_dir() and RUN_RE.match(d.name):
        print("[dir]", d)
        if apply:
            shutil.rmtree(d)

# 2) eval 下各折 summary.csv / detail.csv 中 method == METHOD 的行
for f in sorted((root / "eval").glob("*_fold*/*.csv")):
    if f.name not in ("summary.csv", "detail.csv"):
        continue
    df = pd.read_csv(f)
    if "method" not in df.columns:
        continue
    mask = df["method"] == METHOD
    if mask.any():
        print(f"[csv] {f}: 删除 {mask.sum()} 行")
        if apply:
            df[~mask].to_csv(f, index=False)