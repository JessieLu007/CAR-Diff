# tests/test_processor_with_csv.py
# 用法: python tests/test_processor_with_csv.py --csv data/framingham.csv
# 通过给变换后的数据加噪声来模拟生成模型输出，再逆变换回原空间，检查结构约束是否恢复

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import pandas as pd
from core.data_processor import TabularDataProcessor


# =============================================================================
# 参数
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv',       required=True,  help='输入 CSV 路径')
    parser.add_argument('--target',    default='label', help='目标列名（会被排除在特征之外）')
    parser.add_argument('--cat_cols',  nargs='+',
                        default=['male','education','currentSmoker',
                                 'BPMeds','prevalentStroke','prevalentHyp','diabetes'],
                        help='类别特征列名')
    parser.add_argument('--method',    default='quantile',
                        choices=['quantile','standard','none'],
                        help='数值变换方法')
    parser.add_argument('--noise',     type=float, default=0.1,
                        help='加噪强度（相对标准差，默认 0.1 即 10%%）')
    parser.add_argument('--seed',      type=int,   default=42)
    parser.add_argument('--save_dir',  default=None, help='若指定则保存对比 CSV 到该目录')
    return parser.parse_args()


# =============================================================================
# 核心流程
# =============================================================================
def run(args):
    rng = np.random.default_rng(args.seed)

    # ---------- 读数据 ----------
    df_all = pd.read_csv(args.csv)
    if 'label' not in df_all.columns:
        df_all['label'] = 1
    print(f"数据集: {args.csv}  shape={df_all.shape}")

    # 排除目标列
    feature_cols = [c for c in df_all.columns if c != args.target]
    cat_cols = [c for c in args.cat_cols if c in feature_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    df = df_all[feature_cols].copy()

    print(f"数值列 ({len(num_cols)}): {num_cols}")
    print(f"类别列 ({len(cat_cols)}): {cat_cols}")

    # ---------- fit ----------
    proc = TabularDataProcessor(
        num_cols=num_cols,
        cat_cols=cat_cols,
        num_transform_method=args.method,
        random_state=args.seed,
    )
    proc.fit(df)

    # 打印检测到的结构信息
    print("\n── 数值列结构检测结果 ──")
    for col in num_cols:
        meta = proc.meta[col]
        q    = meta['quantize_step']
        prec = proc.precisions[col]
        print(f"  {col:<20} quantize_step={str(q):<6}  precision={prec}  is_int={meta['is_int']}")

    # ---------- transform ----------
    df_t = proc.transform(df)

    # ---------- 加噪（模拟生成模型输出）----------
    noise_std = args.noise
    df_noisy  = df_t.copy()

    # 数值列加高斯噪声
    for col in num_cols:
        col_std = float(df_t[col].std())
        df_noisy[col] += rng.normal(0, noise_std * col_std, size=len(df_t))

    # 类别列加少量噪声（模拟 logit 抖动后取整误差）
    for col in cat_cols:
        n_classes = proc.meta[col]['classes']
        flip_mask = rng.random(len(df_t)) < (noise_std * 0.5)
        noise_val = rng.integers(0, n_classes, size=len(df_t))
        df_noisy[col] = np.where(flip_mask, noise_val.astype(float), df_noisy[col])

    # 随机注入 2% NaN（模拟极端生成值被 mask 的场景）
    for col in num_cols + cat_cols:
        idx = rng.choice(len(df_noisy), size=max(1, int(len(df_noisy) * 0.02)), replace=False)
        df_noisy.iloc[idx, df_noisy.columns.get_loc(col)] = np.nan

    print(f"\n加噪强度: {noise_std * 100:.0f}%，注入 NaN 比例: 2%")

    # ---------- inverse_transform ----------
    df_inv = proc.inverse_transform(df_noisy)

    # ==========================================================================
    # 结构约束验证
    # ==========================================================================
    print("\n── 逆变换结构约束验证 ──")
    all_pass = True

    for col in num_cols:
        meta = proc.meta[col]
        q    = meta['quantize_step']
        vals = df_inv[col].dropna().values

        issues = []

        # 范围检查：允许 3% 超出（注入了 2% NaN + 10% 噪声，边缘超出属正常）
        out_of_range = np.mean((vals < meta['min'] - 1) | (vals > meta['max'] + 1))
        if out_of_range > 0.03:
            issues.append(f"超范围 {out_of_range*100:.1f}%")

        # 量化对齐检查
        if q is not None:
            remainder = np.abs(vals % q)
            not_aligned = np.mean(np.minimum(remainder, np.abs(remainder - q)) > 1e-3)
            if not_aligned > 0.01:
                issues.append(f"未对齐到 q={q} 的比例 {not_aligned*100:.1f}%")
            else:
                aligned_str = f"✓ 对齐到 q={q}"
        else:
            aligned_str = "─ 连续列（无量化约束）"

        status = "✓" if not issues else "✗"
        issue_str = "  " + ", ".join(issues) if issues else ""
        print(f"  {status} {col:<20} {aligned_str}{issue_str}")
        if issues:
            all_pass = False

    for col in cat_cols:
        valid_labels = set(proc.meta[col]['encoder'].classes_)
        inv_labels   = set(df_inv[col].dropna().astype(str).unique())
        out_of_set   = inv_labels - valid_labels
        if out_of_set:
            print(f"  ✗ {col:<20} 出现未知类别: {out_of_set}")
            all_pass = False
        else:
            print(f"  ✓ {col:<20} 类别值全部合法 {sorted(valid_labels)}")

    # ==========================================================================
    # 数值对比统计（原始 vs 逆变换）
    # ==========================================================================
    print("\n── 数值列分布对比（原始 vs 加噪后逆变换）──")
    print(f"  {'列':<20} {'原始均值':>10} {'恢复均值':>10} {'原始std':>10} {'恢复std':>10}")
    print("  " + "-" * 62)
    for col in num_cols:
        orig_mean = df[col].mean()
        inv_mean  = df_inv[col].mean()
        orig_std  = df[col].std()
        inv_std   = df_inv[col].std()
        print(f"  {col:<20} {orig_mean:>10.3f} {inv_mean:>10.3f} {orig_std:>10.3f} {inv_std:>10.3f}")

    # ==========================================================================
    # 可选：保存对比 CSV
    # ==========================================================================
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        df[num_cols + cat_cols].to_csv(
            os.path.join(args.save_dir, 'original.csv'), index=False)
        df_noisy.to_csv(
            os.path.join(args.save_dir, 'noisy_transformed.csv'), index=False)
        df_inv.to_csv(
            os.path.join(args.save_dir, 'inverse_recovered.csv'), index=False)
        print(f"\n对比文件已保存至: {args.save_dir}")

    print(f"\n{'全部约束通过 ✓' if all_pass else '存在约束违反，请检查上方 ✗ 项'}")


# =============================================================================
if __name__ == '__main__':
    run(parse_args())