#!/usr/bin/env bash
# Verified CPU-only configuration (torch 2.2.2). Covers fit / sample / eval / baselines.
# For GPU / latest-version setup use install.sh instead.
set -e

export PIP_INDEX_URL=${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}
FALLBACK_INDEX_URL="https://pypi.org/simple"

pip_install() {
  if ! pip install "$@"; then
    echo "  镜像安装失败，改用官方 PyPI 源重试: $*"
    pip install "$@" --index-url "$FALLBACK_INDEX_URL"
  fi
}

echo "[1/4] Installing base dependencies (includes opacus==1.4.1, required for torch 2.2.2 compatibility)..."
pip_install -r requirements-cpu.txt

echo "[2/4] Installing torch (CPU, 2.2.2)..."
pip_install torch==2.2.2 torchvision==0.17.2

echo "[3/4] Installing shap without resolving its dependencies..."
pip_install shap==0.49.1 --no-deps

echo "[4/4] Installing tsai / fastai..."
pip_install tsai==0.4.0 fastai==2.7.19

echo "Done."