# scripts/clean_runs.py
"""python -m tools.clean_runs --keep-hp h256_t200_3_main            # 预览
   python -m tools.clean_runs --keep-hp h256_t200_3_main --apply    # 执行
只清理 ours 运行目录里的 model.pt、run_bundle.pkl、synthetic_data*.csv，保留 config.json；
--keep-hp 指定的 hp（完整名）整体保留。"""
import argparse, re
from pathlib import Path

HEAVY = ("model.pt", "run_bundle.pkl")
SYN = re.compile(r"synthetic_data(_s\d+)?\.csv")
TS = re.compile(r"(\d{4}_\d{6})$")
PREFIX = "full_data_"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--keep-hp", nargs="+", required=True)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    base = Path(a.root) / "results"
    total = 0
    for d in sorted(base.glob("*/seed_*_fold_*/" + PREFIX + "*")):
        m = TS.search(d.name)
        if not (d.is_dir() and m):
            continue
        hp = d.name[len(PREFIX):m.start() - 1]
        if hp in a.keep_hp:
            continue
        files = [p for p in d.iterdir() if p.name in HEAVY or SYN.fullmatch(p.name)]
        size = sum(p.stat().st_size for p in files)
        total += size
        print(f"{'删除' if a.apply else '将删除'} {d.relative_to(base)}: {len(files)} 个文件, {size / 1e6:.1f} MB")
        if a.apply:
            for p in files:
                p.unlink()
    print(f"合计 {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()