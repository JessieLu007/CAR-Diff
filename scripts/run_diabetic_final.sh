#!/bin/bash
cd "$(dirname "$0")/.."          # 回到项目根目录
export PYTHONUNBUFFERED=1
LIMIT=4h                          # 最长运行时限，到点强制关机

# 看门狗：不管下面的流程跑到哪里，到时限都关机
( sleep $LIMIT; echo "[$(date)] 到达时限，强制关机"; shutdown -h now ) &

for k in 0 1 2 3 4; do
  python -m experiments.run_cv --dataset diabetic_data --fold $k --seed 42 \
      --step_floor 0 --tag final --train "target_epochs=50,enable_compile=True" \
  && python -m experiments.resample_ours --dataset diabetic_data --fold $k \
      --hp h256_t200_3_final --device cuda \
  || { echo "[$(date)] fold $k 失败，停止"; break; }
done

echo "[$(date)] 流程结束，30 秒后关机"
sleep 30
shutdown -h now
