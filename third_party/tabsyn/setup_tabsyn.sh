#!/usr/bin/env bash
# 用法: bash third_party/tabsyn/setup_tabsyn.sh <官方 TabSyn zip 或目录> [--force]
# 结果: third_party/tabsyn/tabsyn-main/（官方代码 + 补丁 + 本仓库的运行脚本）
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:?用法: bash third_party/tabsyn/setup_tabsyn.sh <zip 或目录> [--force]}"
FORCE="${2:-}"
DST="$HERE/tabsyn-main"
PATCH="$HERE/tabsyn_changes.patch"

[ -f "$PATCH" ] || { echo "缺少 $PATCH"; exit 1; }
if [ -e "$DST" ] && [ "$FORCE" != "--force" ]; then
  echo "$DST 已存在；如需重建请加 --force"; exit 1
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
if [ -f "$SRC" ]; then unzip -q "$SRC" -d "$TMP/raw"; else cp -r "$SRC" "$TMP/raw"; fi

# 官方根 = 含 process_dataset.py 的最浅一层（兼容 tabsyn-main/ 或 tabsyn-main-main/）
PD="$(find "$TMP/raw" -maxdepth 3 -name process_dataset.py | awk '{print length, $0}' | sort -n | head -1 | cut -d' ' -f2-)"
[ -n "$PD" ] || { echo "在 $SRC 中找不到 process_dataset.py"; exit 1; }
OFF="$(dirname "$PD")"
[ -d "$OFF/tabsyn" ] || { echo "官方仓库缺少 tabsyn/"; exit 1; }

find "$OFF" -name '*.py' -exec sed -i 's/\r$//' {} +      # 统一换行，避免补丁因 CRLF 失败
( cd "$OFF" && patch -p1 --dry-run < "$PATCH" >/dev/null && patch -p1 < "$PATCH" )

cp "$HERE/run_tabsyn_all.py" "$HERE/resample_tabsyn.py" "$OFF/"
rm -rf "$DST"
cp -r "$OFF" "$DST"
echo "完成: $DST"