# tests/test_quantize_alignment.py
# 验证 inverse_transform 的量化对齐行为：
#   - q=0.5 列：127.3 → 127.5，120.2 → 120.0
#   - 整数列：  3.7 → 4，2.2 → 2
#   - 连续列：  保持精度，不做量化
# 运行：python tests/test_quantize_alignment.py

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from core.data_processor import TabularDataProcessor


def check(desc, actual, expected, tol=1e-4):
    ok = abs(float(actual) - float(expected)) <= tol
    print(f"  {'✓' if ok else '✗'}  {desc}")
    if not ok:
        print(f"       期望: {expected}   实际: {actual}")
    return ok


# =============================================================================
# 构造最小训练集：让 Processor 学到各列的结构
#   sysBP  : 半整数列（q=0.5）
#   age    : 整数列（q=1.0）
#   bmi    : 连续列（q=None）
# =============================================================================
df_train = pd.DataFrame({
    'sysBP': [100.0, 100.5, 110.0, 120.0, 120.5, 127.5, 130.0, 140.0, 150.5, 160.0],
    'age':   [1, 2, 3, 4, 5, 30,    34,    40,    45,    70],
    'bmi':   [18.4,  22.7,  25.1,  27.13,  30.08, 19, 24, 31.2,  28.77, 21 ],
})

num_cols = ['sysBP', 'age', 'bmi']
cat_cols = []

proc = TabularDataProcessor(num_cols=num_cols, cat_cols=cat_cols)
proc.fit(df_train)

print("── 检测结构 ──")
for col in num_cols:
    q    = proc.meta[col]['quantize_step']
    prec = proc.precisions[col]
    print(f"  {col:<8} quantize_step={str(q):<6}  precision={prec}")


# =============================================================================
# 构造"生成模型的原始输出"：直接以原始数值空间的值塞进去，跳过 transform
# 模拟扩散模型输出了需要量化对齐的中间值
# =============================================================================
# 用 'none' 模式的 Processor 做透传，避免量化变换干扰
proc_none = TabularDataProcessor(num_cols=num_cols, cat_cols=cat_cols,
                                  num_transform_method='none')
proc_none.fit(df_train)

# 构造一组"脏"数值，直接放入逆变换入口
dirty = pd.DataFrame({
    'sysBP': [127.3,  120.2,  130.7,  99.8,   140.25],
    'age':   [3.7,    2.2,    47.6,   64.9,   30.1  ],
    'bmi':   [25.134, 18.401, 30.082, 27.991, 22.713],
})

result = proc_none.inverse_transform(dirty)

print("\n── sysBP（q=0.5）量化对齐 ──")
passed = 0
cases_sysbp = [
    (127.3, 127.5),
    (120.2, 120.0),
    (130.7, 130.5),
    ( 99.8, 100.0),
    (140.25, 140.0),  # 140.25 / 0.5 = 280.5 → round → 280 → 140.0
]
for (raw, expected), actual in zip(cases_sysbp, result['sysBP']):
    passed += check(f"sysBP {raw} → {expected}", actual, expected)
print(f"  {passed}/{len(cases_sysbp)} 通过")

print("\n── age（整数列）量化对齐 ──")
passed = 0
cases_age = [
    (3.7,  4),
    (2.2,  2),
    (47.6, 48),
    (64.9, 65),
    (30.1, 30),
]
for (raw, expected), actual in zip(cases_age, result['age']):
    passed += check(f"age {raw} → {expected}", actual, expected)
print(f"  {passed}/{len(cases_age)} 通过")

print("\n── bmi（连续列）不做量化，只做精度截断 ──")
passed = 0
cases_bmi = [
    (25.134, 25.13),
    (18.401, 18.40),
    (30.082, 30.08),
    (27.991, 27.99),
    (22.713, 22.71),
]
for (raw, expected), actual in zip(cases_bmi, result['bmi']):
    passed += check(f"bmi {raw} → {expected}", actual, expected)
print(f"  {passed}/{len(cases_bmi)} 通过")

print("\n── 验证不依赖训练集中出现过的值 ──")
# 127.3 → 127.5：训练集里有 127.5，但这是数学格点，不是查表
# 130.7 → 130.5：训练集里没有 130.5，必须是格点计算才能得到
passed = 0
passed += check("130.7 → 130.5（训练集中无此值，确认非查表）",
                result['sysBP'].iloc[2], 130.5)
passed += check("99.8 → 100.0（格点对齐，非最近训练样本）",
                result['sysBP'].iloc[3], 100.0)
print(f"  {passed}/2 通过")