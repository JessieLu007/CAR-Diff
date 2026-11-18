#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1

DATASETS=(hepatitis chronic_kidney_disease framingham)
FOLDS=(0 1)

HPS=(
  h256_t200_3_main_indexfix
  h256_t200_3_main_s43_indexfix
  h256_t200_3_main_s44_indexfix
  h256_t200_3_abl_prob
  h256_t200_3_abl_sampled
)

# 先确保各个方法的指标都已写入 eval/summary.csv
for hp in "${HPS[@]}"; do
  echo "===== Evaluating $hp ====="
  python -m evaluation.run \
    --dataset "${DATASETS[@]}" \
    --fold "${FOLDS[@]}" \
    --hp "$hp"
done

# 以修正后的 main 为基线，用两个训练种子估计训练波动
python -m evaluation.tune_compare compare \
  --prefix ours_h256_t200_3 \
  --base main_indexfix \
  --cands abl_prob abl_sampled \
  --noise-tags main_indexfix main_s43_indexfix main_s44_indexfix \
  --datasets "${DATASETS[@]}" \
  --folds "${FOLDS[@]}"