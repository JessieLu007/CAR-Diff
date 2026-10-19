#!/usr/bin/env bash
# 用法: bash scripts/run_main.sh
cd "$(dirname "$0")/.."        # 切到项目根
mkdir -p logs

nohup bash -c '
  timeout -k 30s 20h bash -c "
    for ds in chronic_kidney_disease framingham hcv_egy_data hepatitis risk_factors_cervical_cancer; do
      python -m experiments.run_cv --dataset \$ds --tag main --train target_epochs=200 &&
      python -m experiments.resample_ours --dataset \$ds --device cuda --js 1 2 3 4 &&
      python -m evaluation.run --dataset \$ds || echo \"FAIL \$ds\"
    done
    python -m experiments.run_cv --dataset diabetic_data --tag main --train total_steps=10600 &&
    python -m experiments.resample_ours --dataset diabetic_data --device cuda --js 1 2 3 4 &&
    python -m evaluation.run --dataset diabetic_data || echo \"FAIL diabetic_data\"
  "
  sync
  shutdown -h now
' > logs/log_main_all.txt 2>&1 &

echo "started, log: logs/log_main_all.txt"