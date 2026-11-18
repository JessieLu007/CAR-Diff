#!/usr/bin/env bash
# GPU-only setup for fit/sample only (no eval, no baselines).
# Usage:
#   bash install-core.sh                 # torch==2.14.1, cu130
#   bash install-core.sh --cuda cu121    # a different CUDA build (see https://pytorch.org for tags)
set -e

export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
FALLBACK_INDEX_URL="https://pypi.org/simple"

CUDA_TAG="cu130"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cuda) CUDA_TAG="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if ! command -v nvidia-smi &> /dev/null; then
  echo "未检测到 nvidia-smi：此脚本仅支持 GPU 环境"
  exit 1
fi

pip_install() {
  if ! pip install "$@"; then
    echo "  镜像安装失败，改用官方 PyPI 源重试: $*"
    pip install "$@" --index-url "$FALLBACK_INDEX_URL"
  fi
}

echo "[1/2] Installing base dependencies..."
pip_install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 scipy==1.17.1

echo "[2/2] Installing torch==2.14.1 (${CUDA_TAG})..."
pip_install torch==2.14.1 --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --trusted-host pypi.tuna.tsinghua.edu.cn

echo "Done. This environment supports 'fit' / 'sample' only — eval/baselines need install.sh."