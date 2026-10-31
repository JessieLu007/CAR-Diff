#!/usr/bin/env bash
# 用法: bash third_party/tabkde/setup_tabkde.sh <官方仓库 zip 或 解压后的目录> [--force]
# 结果: 在仓库根目录生成 tabkde/（官方代码 + 本仓库补丁）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SRC="${1:?用法: bash third_party/tabkde/setup_tabkde.sh <tabkde-main.zip 或目录> [--force]}"
FORCE="${2:-}"
PATCH="$HERE/tabkde_changes.patch"

[ -f "$PATCH" ] || { echo "缺少 $PATCH"; exit 1; }
if [ -e "$ROOT/tabkde" ] && [ "$FORCE" != "--force" ]; then
  echo "$ROOT/tabkde 已存在；如需重建请加 --force"; exit 1
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [ -f "$SRC" ]; then unzip -q "$SRC" -d "$TMP/raw"; else cp -r "$SRC" "$TMP/raw"; fi

# 官方仓库根 = 含 process_dataset.py 的最浅一层（兼容 tabkde-main/ 或 tabkde-main-main/ 等顶层目录）
PD="$(find "$TMP/raw" -maxdepth 3 -name process_dataset.py | awk '{print length, $0}' | sort -n | head -1 | cut -d' ' -f2-)"
[ -n "$PD" ] || { echo "在 $SRC 中找不到 process_dataset.py"; exit 1; }
OFF="$(dirname "$PD")"
for f in tabkde process_dataset.py utils_train.py src; do
  [ -e "$OFF/$f" ] || { echo "官方仓库缺少 $f"; exit 1; }
done

# 组装：官方 tabkde/ 包 + 根目录的 process_dataset.py、utils_train.py、src/ + 空 __init__.py
mkdir -p "$TMP/asm"
cp -r "$OFF/tabkde" "$TMP/asm/tabkde"
cp "$OFF/process_dataset.py" "$OFF/utils_train.py" "$TMP/asm/tabkde/"
cp -r "$OFF/src" "$TMP/asm/tabkde/src"
find "$TMP/asm/tabkde" -type d | while read -r d; do
  ls "$d"/*.py >/dev/null 2>&1 && [ ! -e "$d/__init__.py" ] && : > "$d/__init__.py" || true
done
[ -e "$TMP/asm/tabkde/__init__.py" ] || : > "$TMP/asm/tabkde/__init__.py"

# 补丁路径头为 a/ b/（patch -p1）；若头里带 tabkde/ 前缀则在上一级应用
if grep -q '^--- a/tabkde/' "$PATCH"; then WD="$TMP/asm"; else WD="$TMP/asm/tabkde"; fi
( cd "$WD" && patch -p1 --dry-run < "$PATCH" >/dev/null && patch -p1 < "$PATCH" )

rm -rf "$ROOT/tabkde"
cp -r "$TMP/asm/tabkde" "$ROOT/tabkde"
echo "完成: $ROOT/tabkde"