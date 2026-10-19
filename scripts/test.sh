python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

names = {
    "inputnormal": "full_data_h256_t200_3_main_indexfix_1017_225103",
    "catzero": "full_data_h256_t200_3_main_catzero_1017_225103",
}

files = {
    p.parent.name: p
    for p in Path(".").rglob("synthetic_data.csv")
    if p.parent.name in names.values()
}

for label, dirname in names.items():
    if dirname not in files:
        raise FileNotFoundError(f"没有找到 {dirname}/synthetic_data.csv")

a = pd.read_csv(files[names["inputnormal"]])
b = pd.read_csv(files[names["catzero"]])

print("文件 A:", files[names["inputnormal"]])
print("文件 B:", files[names["catzero"]])
print("Shape:", a.shape, b.shape)

if a.shape != b.shape or list(a.columns) != list(b.columns):
    raise ValueError("两份数据的形状或列顺序不同，不能直接配对比较")

print("CSV 完全相同:", a.equals(b))

changed_rows = np.zeros(len(a), dtype=bool)
changed_cells = 0

for col in a.columns:
    if (pd.api.types.is_numeric_dtype(a[col])
            and pd.api.types.is_numeric_dtype(b[col])):
        x = a[col].to_numpy(dtype=float)
        y = b[col].to_numpy(dtype=float)
        mask = ~np.isclose(x, y, rtol=1e-7, atol=1e-8, equal_nan=True)
        valid = np.isfinite(x) & np.isfinite(y)
        max_diff = np.max(np.abs(x[valid] - y[valid])) if valid.any() else float("nan")
        detail = f"最大绝对差={max_diff:.8g}"
    else:
        x = a[col].astype("string").fillna("<NA>")
        y = b[col].astype("string").fillna("<NA>")
        mask = (x != y).to_numpy()
        detail = "类别/文本列"

    n = int(mask.sum())
    changed_cells += n
    changed_rows |= mask

    if n:
        print(f"{col}: {n}/{len(a)} 行变化，{detail}")

print(f"\n变化单元格总数: {changed_cells}/{a.size}")
print(f"变化样本数: {int(changed_rows.sum())}/{len(a)}")
PY