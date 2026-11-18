#!/usr/bin/env bash
# Usage:
#   bash install-core.sh                # auto-detect: GPU (cu130) if nvidia-smi is found, else CPU
#   bash install-core.sh --cuda cu121   # force a specific CUDA build (see https://pytorch.org for tags)
#   bash install-core.sh --cpu          # force CPU-only torch
set -e

TORCH_TARGET="auto"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) TORCH_TARGET="$2"; shift 2 ;;
    --cpu)  TORCH_TARGET="cpu"; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$TORCH_TARGET" == "auto" ]]; then
  if command -v nvidia-smi &> /dev/null; then
    TORCH_TARGET="cu130"
  else
    TORCH_TARGET="cpu"
  fi
fi

echo "[1/2] Installing numpy/pandas/scikit-learn/scipy..."
pip install -r requirements-core.txt

echo "[2/2] Installing torch (${TORCH_TARGET})..."
if [[ "$TORCH_TARGET" == "cpu" ]]; then
  pip install torch==2.14.1
else
  pip install torch==2.14.1 --index-url "https://download.pytorch.org/whl/${TORCH_TARGET}"
fi

echo "Done. This environment supports 'fit' / 'sample' only — run install.sh for eval/baselines."