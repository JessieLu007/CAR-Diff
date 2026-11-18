#!/usr/bin/env bash
# Usage:
#   bash install.sh            # auto-detect: GPU (cu130) if nvidia-smi is found, else CPU
#   bash install.sh --cuda cu121   # force a specific CUDA build (see https://pytorch.org for tags)
#   bash install.sh --cpu          # force CPU-only torch
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

echo "[1/5] Installing base dependencies..."
pip install -r requirements.txt

echo "[2/5] Installing torch (${TORCH_TARGET})..."
if [[ "$TORCH_TARGET" == "cpu" ]]; then
  pip install torch==2.14.1
else
  pip install torch==2.14.1 --index-url "https://download.pytorch.org/whl/${TORCH_TARGET}"
fi

echo "[3/5] Pinning setuptools back to 75.8.2 (torch pulls a newer one)..."
pip install setuptools==75.8.2 --no-deps

echo "[4/5] Installing shap without resolving its dependencies..."
pip install shap==0.51.0 --no-deps

echo "[5/5] Installing tsai / fastai (overrides the stale versions synthcity pulls in)..."
pip install tsai==1.0.1 fastai==2.8.12

echo "Done."