export OMP_NUM_THREADS=1

for ds in hepatitis chronic_kidney_disease framingham; do
  for fold in 0 1; do
    if [ "$ds" = hepatitis ] && [ "$fold" = 0 ]; then
      continue
    fi

    python -m experiments.resample_ours \
      --dataset "$ds" \
      --fold "$fold" \
      --hp h256_t200_3_main \
      --js 0 \
      --base_seed 2026 \
      --cat_input prob \
      --suffix _indexfix \
      --force \
      --device cuda || exit 1
  done
done