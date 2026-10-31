# 用法:
#   python run_tabsyn_all.py --dataset hepatitis
#   python run_tabsyn_all.py --datasets hepatitis ckd heart --n_folds 5
#   python run_tabsyn_all.py --all --idx_json /path/to/datasets_idx.json
#   python run_tabsyn_all.py --datasets hepatitis --steps vae,tabsyn
import os, sys, json, subprocess, argparse, time
from datetime import timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
TABSYN_ROOT = _HERE
DEFAULT_IDX_JSON = os.path.join(_HERE, "..", "..", "..", "data", "datasets_idx.json") #指向项目根目录
DEFAULT_LOG_DIR  = os.path.join(_HERE, "logs")


def fmt_duration(seconds):
    if seconds < 1:
        return f"{int(seconds * 1000)}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return str(timedelta(seconds=int(seconds)))


def tail(path, n=30):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "(no log)"


def step_previously_succeeded(log_file, cmd):
    """在 log_file 里找这条 cmd 最近一次的执行记录，返回它是否以 exit=0 结束。
    只信"exit=0"这行，不是看 log 文件存不存在——log 一开始执行就会创建，不代表跑完。
    """
    if not log_file or not os.path.exists(log_file):
        return False
    start_marker = f"] $ {' '.join(cmd)}\n"
    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    last_exit0 = False
    i = 0
    while i < len(lines):
        if lines[i].endswith(start_marker):
            for j in range(i + 1, len(lines)):
                if lines[j].startswith("[end"):
                    last_exit0 = "exit=0" in lines[j]
                    i = j
                    break
        i += 1
    return last_exit0


def run(cmd, log_file=None, cwd=TABSYN_ROOT):
    """执行一条命令；返回 (ok, elapsed_seconds)。"""
    printable = " ".join(cmd)
    t0 = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"  $ {printable}", flush=True)

    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n[{'=' * 60}]\n[start {start_str}] $ {printable}\n")
            f.flush()
            r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
            elapsed = time.time() - t0
            f.write(f"[end   {time.strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"exit={r.returncode} elapsed={fmt_duration(elapsed)}\n")
        if r.returncode != 0:
            print(f"  --- {log_file} (tail -30) ---", flush=True)
            print(tail(log_file, 30), flush=True)
            print("  --- end ---", flush=True)
    else:
        r = subprocess.run(cmd, cwd=cwd)
        elapsed = time.time() - t0

    mark = "✓" if r.returncode == 0 else "✗"
    print(f"  {mark} {fmt_duration(elapsed)}", flush=True)
    return r.returncode == 0, elapsed


def build_step_cmd(step, name, python, tabsyn_root):
    if step == "process":
        return [python, "process_dataset.py", "--dataname", name]
    if step == "vae":
        return [python, "main.py", "--dataname", name, "--method", "vae", "--mode", "train"]
    if step == "tabsyn":
        return [python, "main.py", "--dataname", name, "--method", "tabsyn", "--mode", "train"]
    if step == "sample":
        save_path = os.path.join(tabsyn_root, "synthetic", name, "tabsyn.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        return [python, "main.py", "--dataname", name, "--method", "tabsyn",
                "--mode", "sample", "--save_path", save_path]
    raise ValueError(f"unknown step: {step}")


def resolve_datasets(args, all_ds):
    if args.datasets:
        return list(args.datasets)
    if args.all:
        if not all_ds:
            raise SystemExit(f"--all 需要 {args.idx_json} 存在且非空")
        return list(all_ds)
    if args.dataset:
        return [args.dataset]
    raise SystemExit("请指定 --dataset / --datasets / --all 之一")


def write_summary(records, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("dataset\tfold\tstep\tstatus\telapsed_sec\telapsed_human\n")
        for r in records:
            f.write(f"{r['dataset']}\t{r['fold']}\t{r['step']}\t{r['status']}\t"
                    f"{r['elapsed']:.1f}\t{fmt_duration(r['elapsed'])}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",  default=None)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--all",      action="store_true")
    p.add_argument("--idx_json", default=DEFAULT_IDX_JSON)
    p.add_argument("--n_folds",  type=int, default=5)
    p.add_argument("--steps",    default="process,vae,tabsyn,sample")
    p.add_argument("--python",   default=sys.executable)
    p.add_argument("--log_dir",  default=DEFAULT_LOG_DIR)
    p.add_argument("--no_log",   action="store_true")
    p.add_argument("--force",    action="store_true",
                    help="忽略 log 里的历史成功记录，强制重跑所有 step（默认跳过 log 中 exit=0 的 step，实现断点续跑；配合 --no_log 时无历史可查，等同于强制重跑）")
    args = p.parse_args()

    all_ds = []
    if os.path.exists(args.idx_json):
        with open(args.idx_json, "r", encoding="utf-8") as f:
            all_ds = list(json.load(f).keys())

    datasets = resolve_datasets(args, all_ds)
    os.makedirs(args.log_dir, exist_ok=True)
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    run_t0 = time.time()
    print(f"run started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"datasets    : {datasets}")
    print(f"n_folds     : {args.n_folds}")
    print(f"steps       : {steps}")
    print(f"log_dir     : {'(disabled)' if args.no_log else args.log_dir}", flush=True)

    records = []
    failed = []

    for d in datasets:
        for f in range(args.n_folds):
            name = f"{d}_fold{f}"
            log_file = None if args.no_log else os.path.join(args.log_dir, f"{name}.log")
            print(f"\n========== {name} ==========", flush=True)
            fold_t0 = time.time()
            ok = True

            for step in steps:
                if not ok:
                    records.append(dict(dataset=d, fold=f, step=step,
                                        status="skipped", elapsed=0.0))
                    continue

                cmd = build_step_cmd(step, name, args.python, TABSYN_ROOT)

                if not args.force and step_previously_succeeded(log_file, cmd):
                    print(f"  ↷ {step} log 里已有成功记录，跳过（用 --force 可强制重跑）", flush=True)
                    records.append(dict(dataset=d, fold=f, step=step,
                                        status="skipped(done)", elapsed=0.0))
                    continue

                ok_step, elapsed = run(cmd, log_file)
                records.append(dict(dataset=d, fold=f, step=step,
                                    status="ok" if ok_step else "fail",
                                    elapsed=elapsed))
                ok = ok and ok_step

            fold_elapsed = time.time() - fold_t0
            print(f"  fold total: {fmt_duration(fold_elapsed)}", flush=True)

            if not ok:
                failed.append(name)

    run_elapsed = time.time() - run_t0

    summary_path = os.path.join(args.log_dir, "summary.tsv")
    write_summary(records, summary_path)

    # Top-5 最耗时步骤，一眼看瓶颈
    top = sorted([r for r in records if r["status"] == "ok"],
                 key=lambda x: -x["elapsed"])[:5]
    if top:
        print("\nTop-5 最耗时步骤:")
        for r in top:
            print(f"  {r['dataset']}_fold{r['fold']}  {r['step']:8s}  {fmt_duration(r['elapsed'])}")

    print("\n" + "=" * 60)
    print(f"总耗时: {fmt_duration(run_elapsed)}")
    print(f"总览:   {summary_path}")
    if failed:
        print(f"⚠ 失败 {len(failed)} 个: {failed}")
        sys.exit(1)
    print("✅ 全部完成")


if __name__ == "__main__":
    main()