#!/usr/bin/env bash
# scripts/overnight.sh —— 无人值守：等待已启动任务结束 → 各阶段 补跑训练 / 评估 / 聚合 → 全部结束（无论成败）后关机
#
# 用法（在 (ddpm) 环境、主目录 ddpm_med 下）：
#   nohup bash scripts/overnight.sh > logs/overnight.log 2>&1 &
#   WITH_DIAB=0 nohup bash scripts/overnight.sh > logs/overnight.log 2>&1 &     # 不含 diabetic_data 的消融阶段（diab_epochs 仍会运行）
# 手动终止（Ctrl-C / kill 脚本）不会触发关机；注意 kill 脚本不会停掉正在跑的 python 子进程，需另外：
#   pkill -f experiments.tune_batch; pkill -f experiments.run_cv; pkill -f experiments.resample_ours; pkill -f evaluation.run
# 想让脚本正常走完但不关机：touch logs/NO_SHUTDOWN
# 进度与各步骤退出码见 logs/overnight_status.txt；每步的详细输出在 logs/overnight_<步骤名>.log。
# 训练阶段按 tune_batch 的"已完成则跳过"逻辑运行，因此脚本可以安全地重复启动，用来补跑失败或缺失的部分。

cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
PARENT=$(dirname "$ROOT")
BK_CAT="$PARENT/ddpm_med_bk_cat"
BK_AR="$PARENT/ddpm_med_bk_ar"
WITH_DIAB=${WITH_DIAB:-1}
STATUS="$ROOT/logs/overnight_status.txt"
mkdir -p "$ROOT/logs"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

finish() {
  log "全部流程结束（含失败的步骤，见上面各步 rc）"
  if [ -e "$ROOT/logs/NO_SHUTDOWN" ]; then
    log "检测到 logs/NO_SHUTDOWN，不关机"
    return
  fi
  log "30 秒后关机"
  sync
  sleep 30
  shutdown || poweroff
}
trap finish EXIT
# 手动终止（Ctrl-C / kill）不关机；只有走完全部步骤或脚本自身出错退出才关机
trap 'log "收到中断信号（手动终止），不关机"; trap - EXIT; exit 130' INT TERM HUP

# step <名称> <运行目录> <命令...>：某一步失败不会中断后续步骤，退出码记入状态文件
step() {
  local name=$1 dir=$2
  shift 2
  log "START $name"
  ( cd "$dir" && "$@" ) >> "$ROOT/logs/overnight_${name}.log" 2>&1
  log "END   $name rc=$?"
}

TB="python -m experiments.tune_batch"

agg() {   # 聚合很快，多处调用，保证中途失败时也有可用的中间结果
  step agg_main     "$ROOT" python scripts/aggregate_main.py
  step agg_ablation "$ROOT" python scripts/aggregate_variants.py ablation
  step agg_stab     "$ROOT" python scripts/aggregate_variants.py stability
  step agg_diab_ep  "$ROOT" python scripts/aggregate_variants.py diab_epochs
}

log "等待已启动的训练/评估任务结束..."
while pgrep -f "experiments\.tune_batch|experiments\.run_cv|experiments\.resample_ours|evaluation\.run" > /dev/null; do
  sleep 60
done
log "无进行中的任务，开始"

# ---------- A. 消融（小数据集）：补跑缺失 → 评估（评估一律在主目录）→ 聚合 ----------
step train_abl_cat  "$BK_CAT" $TB abl_cat
step train_abl_ar   "$BK_AR"  $TB abl_ar
step train_abl_main "$ROOT"   $TB abl_main
step eval_abl_cat   "$ROOT"   $TB abl_cat  --eval-only
step eval_abl_ar    "$ROOT"   $TB abl_ar   --eval-only
step eval_abl_main  "$ROOT"   $TB abl_main --eval-only
agg

# ---------- B. diabetic_data 训练预算（ep200 是主表里 diabetic_data 的正式结果，ep12 为敏感性）；不受 WITH_DIAB 控制 ----------
step train_diab_epochs "$ROOT" $TB diab_epochs
step eval_diab_epochs  "$ROOT" $TB diab_epochs --eval-only
agg

# ---------- C. 超参数稳定性（小数据集 × 5 折 × 8 个配置）----------
step train_stab "$ROOT" $TB stab
step eval_stab  "$ROOT" $TB stab --eval-only
agg

# ---------- D. diabetic_data 消融（耗时最长，放最后；便宜的在前）----------
if [ "$WITH_DIAB" = "1" ]; then
  step train_diab_abl_main "$ROOT"   $TB diab_abl_main
  step eval_diab_abl_main  "$ROOT"   $TB diab_abl_main --eval-only
  step train_diab_abl_ar   "$BK_AR"  $TB diab_abl_ar
  step eval_diab_abl_ar    "$ROOT"   $TB diab_abl_ar --eval-only
  step train_diab_abl_cat  "$BK_CAT" $TB diab_abl_cat
  step eval_diab_abl_cat   "$ROOT"   $TB diab_abl_cat --eval-only
  agg
fi
# 脚本结束 → EXIT trap → finish → 关机