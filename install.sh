#!/usr/bin/env bash
# GPU-only setup (matches the paper's training environment).
# Usage:
#   bash install.sh                 # torch==2.14.1, cu130
#   bash install.sh --cuda cu121    # a different CUDA build (see https://pytorch.org for tags)
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
  echo "未检测到 nvidia-smi：此脚本仅支持 GPU 环境，CPU 验证请用 install-cpu.sh"
  exit 1
fi

pip_install() {
  if ! pip install "$@"; then
    echo "  镜像安装失败，改用官方 PyPI 源重试: $*"
    pip install "$@" --index-url "$FALLBACK_INDEX_URL"
  fi
}

echo "[1/5] Installing base dependencies (this step can be slow — see README)..."
pip_install -r requirements.txt

echo "[2/5] Installing torch==2.14.1 (${CUDA_TAG})..."
pip_install torch==2.14.1 --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
  --trusted-host pypi.tuna.tsinghua.edu.cn

echo "[3/5] Pinning setuptools back to 75.8.2..."
pip_install setuptools==75.8.2 --no-deps

echo "[4/5] Installing shap without resolving its dependencies..."
pip_install shap==0.51.0 --no-deps

echo "[5/5] Installing tsai / fastai..."
pip_install tsai==1.0.1 fastai==2.8.12

echo "Done."