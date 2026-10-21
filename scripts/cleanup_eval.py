# cleanup_eval.py   在 ddpm_med 目录下运行
import sys, shutil
import pandas as pd
from pathlib import Path
from collections import Counter

EVAL_DIR = Path("eval")
APPLY = "--apply" in sys.argv


def is_trash(m: str) -> bool:
    # 以 ours 开头且名字里不含 main 的，视为测试类结果；baselines 与含 main 的 ours 保留
    return m.startswith("ours") and "main" not in m


del_cnt, keep_cnt = Counter(), Counter()
for fname in ("summary.csv", "detail.csv"):
    for p in sorted(EVAL_DIR.glob(f"*/{fname}")):
        df = pd.read_csv(p)
        mask = df["method"].map(is_trash)
        for m, n in df.loc[mask, "method"].value_counts().items():
            del_cnt[(fname, m)] += n
        ours_keep = ~mask & df["method"].str.startswith("ours")
        for m, n in df.loc[ours_keep, "method"].value_counts().items():
            keep_cnt[(fname, m)] += n
        if APPLY and mask.any():
            shutil.copy2(p, str(p) + ".bak")
            df[~mask].to_csv(p, index=False)

print("== 将删除（文件, method, 行数）==")
for k, n in sorted(del_cnt.items()):
    print(k, n)
print("\n== 保留的 ours 类（文件, method, 行数）==")
for k, n in sorted(keep_cnt.items()):
    print(k, n)
print("\n已执行删除，备份为 *.csv.bak" if APPLY else "\n预演结束，未改动任何文件；确认后加 --apply 执行")