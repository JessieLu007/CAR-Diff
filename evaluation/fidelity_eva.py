# core/fidelity_eva.py
"""
Tabular Synthetic Data Fidelity Evaluation
===========================================
三层评估框架，服务于「少数类局部流形学习」研究主线：

  Layer 1 —— Basic Statistical Fidelity（基础合理性验证，简要）
      TVD (Total Variation Distance)：有界 [0,1]，直觉清晰
      Correlation Matrix Difference：全局成对关联差异摘要

  Layer 2 —— Minority Structure Preservation（核心）
      2.1 Dependency Preservation
          仅在少数类子集上计算混合类型相关矩阵（real-minority vs fake-minority），
          揭示多数类主导下被掩盖的少数类内部结构偏差。
      2.2 Minority Coverage
          α-Precision：生成少数类样本中有多少"真实可信"（精确度）
          β-Recall   ：真实少数类局部模式有多少比例被覆盖（召回率）
          两者共同量化 mode collapse vs mode dropping 风险。

  Layer 3 —— Auxiliary Evaluation（辅助验证，收尾）
      C2ST (Classifier Two-Sample Test)：全局联合分布可分性 + 特征重要性诊断
      DCR  (Distance to Closest Record)：隐私/记忆化风险，比较 train/holdout DCR 分布

使用方法
--------
from core.fidelity_eva import evaluate, FidelityReport

report = evaluate(
    real_train_df = real_train,
    real_hold_df  = real_holdout,
    fake_df       = synthetic,
    num_cols      = ['age', 'totChol', 'sysBP', ...],
    cat_cols      = ['male', 'education', 'currentSmoker', ...],
    minority_col  = 'label',     # 用于定位少数类的列
    minority_val  = 1,           # 少数类的取值
)
print(report)

# 各层可独立调用
from core.fidelity_eva import (
    compute_layer1_fidelity,
    compute_layer2_minority,
    compute_layer3_auxiliary,
)

依赖
----
pip install pandas numpy scipy scikit-learn xgboost
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False


# ═════════════════════════════════════════════════════════════════════════════
# 结果容器
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Layer1Report:
    """基础统计一致性"""
    tvd_per_col:     dict[str, float] = field(default_factory=dict)
    avg_tvd:         float = 0.0
    corr_diff_mean:  float = 0.0   # 全局相关矩阵平均绝对差
    corr_diff_fro:   float = 0.0   # Frobenius norm
    corr_real:       Optional[pd.DataFrame] = None
    corr_fake:       Optional[pd.DataFrame] = None
    corr_diff:       Optional[pd.DataFrame] = None


@dataclass
class Layer2Report:
    """少数类结构学习"""
    # 2.1 Dependency Preservation
    minority_corr_real:      Optional[pd.DataFrame] = None
    minority_corr_fake:      Optional[pd.DataFrame] = None
    minority_corr_diff:      Optional[pd.DataFrame] = None
    minority_corr_diff_mean: float = 0.0
    minority_corr_diff_fro:  float = 0.0
    minority_n_real:         int   = 0
    minority_n_fake:         int   = 0
    # 2.2 Minority Coverage
    alpha_precision: float = 0.0   # 生成少数类样本的可信比例
    beta_recall:     float = 0.0   # 真实少数类模式的覆盖比例
    k_neighbors:     int   = 5     # 使用的 k 值


@dataclass
class Layer3Report:
    """辅助验证"""
    # C2ST
    c2st_auc:                float = 0.0
    c2st_feature_importance: dict  = field(default_factory=dict)
    # DCR
    dcr_train_mean:   float = 0.0
    dcr_hold_mean:    float = 0.0
    dcr_ks_stat:      float = 0.0   # KS(train_DCR, hold_DCR)，越小越好
    dcr_ks_pval:      float = 0.0   # p 大说明两分布无显著差异 → 无隐私风险
    dcr_train_scores: Optional[np.ndarray] = None
    dcr_hold_scores:  Optional[np.ndarray] = None


@dataclass
class FidelityReport:
    layer1: Layer1Report = field(default_factory=Layer1Report)
    layer2: Layer2Report = field(default_factory=Layer2Report)
    layer3: Layer3Report = field(default_factory=Layer3Report)

    def __repr__(self) -> str:
        l1, l2, l3 = self.layer1, self.layer2, self.layer3
        lines = [
            "═" * 60,
            "  Fidelity Evaluation Report",
            "  (minority manifold learning framework)",
            "═" * 60,
            "",
            "── Layer 1  Basic Statistical Fidelity ─────────────────",
            f"  Avg TVD       : {l1.avg_tvd:.4f}  (↓ lower is better, 0=perfect)",
            f"  CorrDiff mean : {l1.corr_diff_mean:.4f}",
            f"  CorrDiff Fro  : {l1.corr_diff_fro:.4f}",
            "  TVD per column:",
        ]
        for col, v in sorted(l1.tvd_per_col.items(), key=lambda x: -x[1]):
            bar = "█" * int(v * 30)
            lines.append(f"    {col:<24s}  {v:.4f}  {bar}")

        lines += [
            "",
            "── Layer 2  Minority Structure Preservation ─────────────",
            f"  Minority subset  real={l2.minority_n_real}, fake={l2.minority_n_fake}",
            "  2.1 Dependency Preservation (minority-only correlation):",
            f"      CorrDiff mean : {l2.minority_corr_diff_mean:.4f}",
            f"      CorrDiff Fro  : {l2.minority_corr_diff_fro:.4f}",
        ]
        if l2.minority_corr_diff is not None:
            upper = (
                l2.minority_corr_diff
                .where(np.triu(np.ones(l2.minority_corr_diff.shape, dtype=bool), k=1))
                .stack()
                .sort_values(ascending=False)
            )
            lines.append("      Top-5 most deviated pairs (minority):")
            for (r, c), v in upper.head(5).items():
                lines.append(f"        {r}  ↔  {c:<22s}  Δ={v:.4f}")

        lines += [
            f"  2.2 Minority Coverage:",
            f"      α-Precision : {l2.alpha_precision:.4f}  "
            f"({'✓' if l2.alpha_precision > 0.8 else '⚠'}  fake minority in real support)",
            f"      β-Recall    : {l2.beta_recall:.4f}  "
            f"({'✓' if l2.beta_recall > 0.8 else '⚠'}  real minority modes covered)",
        ]

        lines += [
            "",
            "── Layer 3  Auxiliary Evaluation ───────────────────────",
            f"  C2ST AUC      : {l3.c2st_auc:.4f}  "
            f"({'≈ indistinguishable' if l3.c2st_auc < 0.55 else '⚠ detectable gap' if l3.c2st_auc < 0.75 else '✗ highly distinguishable'})",
        ]
        if l3.c2st_feature_importance:
            top = sorted(l3.c2st_feature_importance.items(), key=lambda x: -x[1])[:5]
            lines.append("  C2ST discriminative features (top-5):")
            for fname, imp in top:
                lines.append(f"    {fname:<28s}  {imp:.4f}")

        lines += [
            f"  DCR train mean: {l3.dcr_train_mean:.5f}",
            f"  DCR hold  mean: {l3.dcr_hold_mean:.5f}",
            f"  DCR KS stat   : {l3.dcr_ks_stat:.4f}  "
            f"p={l3.dcr_ks_pval:.3f}  "
            f"({'✓ no privacy risk' if l3.dcr_ks_pval > 0.05 else '⚠ possible memorization'})",
            "═" * 60,
        ]
        return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# 内部工具函数
# ═════════════════════════════════════════════════════════════════════════════

def _tvd_continuous(real: np.ndarray, fake: np.ndarray, n_bins: int = 50) -> float:
    """
    TVD for continuous column via equal-width histogram on combined range.
    TVD = 0.5 * sum |p_i - q_i|，取值 [0, 1]。
    """
    combined = np.concatenate([real, fake])
    bins     = np.histogram_bin_edges(combined, bins=n_bins)
    p, _     = np.histogram(real, bins=bins, density=False)
    q, _     = np.histogram(fake, bins=bins, density=False)
    p        = p.astype(float) / (p.sum() + 1e-12)
    q        = q.astype(float) / (q.sum() + 1e-12)
    return float(0.5 * np.abs(p - q).sum())


def _tvd_categorical(real: np.ndarray, fake: np.ndarray) -> float:
    """TVD for categorical column."""
    cats = np.union1d(np.unique(real), np.unique(fake))
    p    = np.array([np.mean(real == c) for c in cats])
    q    = np.array([np.mean(fake == c) for c in cats])
    return float(0.5 * np.abs(p - q).sum())


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramér's V."""
    ct   = pd.crosstab(a, b).values.astype(float)
    n    = ct.sum()
    if n == 0:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    r, k = ct.shape
    phi2   = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom  = min(r_corr - 1, k_corr - 1)
    return float(np.sqrt(phi2 / denom)) if denom > 0 else 0.0


def _eta_squared(num: np.ndarray, cat: np.ndarray) -> float:
    """η²: fraction of variance in continuous col explained by categorical col."""
    grand_mean = num.mean()
    ss_between = sum(
        np.sum(cat == c) * (num[cat == c].mean() - grand_mean) ** 2
        for c in np.unique(cat)
        if np.sum(cat == c) > 0
    )
    ss_total = np.sum((num - grand_mean) ** 2)
    return float(ss_between / ss_total) if ss_total > 1e-10 else 0.0


def _build_corr_matrix(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """混合类型相关矩阵：num-num=Spearman，cat-cat=Cramér's V，num-cat=η²"""
    all_cols = num_cols + cat_cols
    n        = len(all_cols)
    C        = np.zeros((n, n))
    num_set  = set(num_cols)

    for i, ci in enumerate(all_cols):
        for j, cj in enumerate(all_cols):
            if i == j:
                C[i, j] = 1.0
                continue
            if j < i:
                C[i, j] = C[j, i]
                continue
            ai, aj = df[ci].values, df[cj].values
            if ci in num_set and cj in num_set:
                C[i, j] = _spearman(ai.astype(float), aj.astype(float))
            elif ci not in num_set and cj not in num_set:
                C[i, j] = _cramers_v(ai, aj)
            else:
                num_arr = ai.astype(float) if ci in num_set else aj.astype(float)
                cat_arr = aj if ci in num_set else ai
                C[i, j] = _eta_squared(num_arr, cat_arr)

    return pd.DataFrame(C, index=all_cols, columns=all_cols)


def _standardize_for_distance(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    统一编码（LabelEncode 类别列）+ 标准化（z-score 以真实数据为基准）。
    用于需要计算欧氏距离的场景（α/β, DCR）。
    """
    real_enc = real_df.copy()
    fake_enc = fake_df.copy()
    for col in cat_cols:
        le = LabelEncoder().fit(real_df[col].astype(str))
        real_enc[col] = le.transform(real_df[col].astype(str))
        fake_enc[col] = fake_df[col].astype(str).map(
            lambda v, le=le: le.transform([v])[0] if v in le.classes_ else -1
        )
    X_r = real_enc.values.astype(float)
    X_f = fake_enc.values.astype(float)
    mu  = X_r.mean(axis=0)
    sig = X_r.std(axis=0) + 1e-8
    return (X_r - mu) / sig, (X_f - mu) / sig


# ═════════════════════════════════════════════════════════════════════════════
# Layer 1：基础统计一致性
# ═════════════════════════════════════════════════════════════════════════════

def compute_layer1_fidelity(
    real_df:  pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> Layer1Report:
    """
    Layer 1：基础合理性验证。

    TVD 作为主指标（有界、对称），全局相关矩阵差异作为结构概览。
    这一层只是"门槛检验"，不是论文创新点。
    """
    report = Layer1Report()

    # TVD
    for col in num_cols:
        report.tvd_per_col[col] = _tvd_continuous(real_df[col].values, fake_df[col].values)
    for col in cat_cols:
        report.tvd_per_col[col] = _tvd_categorical(real_df[col].values, fake_df[col].values)
    report.avg_tvd = float(np.mean(list(report.tvd_per_col.values())))

    # 全局相关矩阵差异
    report.corr_real = _build_corr_matrix(real_df, num_cols, cat_cols)
    report.corr_fake = _build_corr_matrix(fake_df, num_cols, cat_cols)
    report.corr_diff = (report.corr_real - report.corr_fake).abs()

    mask = np.triu(np.ones(report.corr_diff.shape, dtype=bool), k=1)
    upper = report.corr_diff.values[mask]
    report.corr_diff_mean = float(upper.mean())
    report.corr_diff_fro  = float(np.linalg.norm(report.corr_diff.values))

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Layer 2：少数类结构学习
# ═════════════════════════════════════════════════════════════════════════════

def _alpha_beta(
    X_real_min: np.ndarray,
    X_fake_min: np.ndarray,
    k: int = 5,
) -> tuple[float, float]:
    """
    α-Precision 和 β-Recall（k-NN 版本，参考 Alaa et al. 2022）。

    α-Precision
    -----------
    对每个生成少数类样本，找其在真实少数类中的第 k 近邻，
    用第 k 近邻的距离作为"真实支撑集半径"的估计。
    若生成样本落在该半径内，视为"可信"。
    α = 可信生成样本比例。

    β-Recall
    ---------
    对每个真实少数类样本，找其在生成少数类中的第 k 近邻，
    若最近邻距离 ≤ 该真实样本在真实集内的第 k 近邻距离，
    视为该真实模式"被覆盖"。
    β = 被覆盖真实样本比例。

    直觉
    ----
    α 低  → 生成了大量不切实际的少数类样本（模式幻觉）
    β 低  → 真实少数类的某些局部模式被模型忽略（模式遗漏）
    """
    if len(X_real_min) < k + 1 or len(X_fake_min) < k + 1:
        warnings.warn(f"少数类样本不足 {k+1} 个，α/β 估计不可靠。")
        return float('nan'), float('nan')

    # 真实少数类内部的第 k 近邻距离（支撑集半径估计）
    nn_rr = NearestNeighbors(n_neighbors=k + 1).fit(X_real_min)
    dist_rr, _ = nn_rr.kneighbors(X_real_min)
    radius_real = dist_rr[:, k]   # 每个真实样本的"局部半径"，[N_real]

    # α-Precision：生成样本 → 真实集最近邻距离 vs 真实局部半径
    nn_r  = NearestNeighbors(n_neighbors=k).fit(X_real_min)
    dist_fr, idx_fr = nn_r.kneighbors(X_fake_min)   # [N_fake, k]
    # 取第 1 近邻距离，与对应真实样本的局部半径比较
    dist_fake_to_real = dist_fr[:, 0]               # [N_fake]
    radius_of_nn      = radius_real[idx_fr[:, 0]]   # 该最近邻样本的局部半径
    alpha = float(np.mean(dist_fake_to_real <= radius_of_nn))

    # β-Recall：真实样本 → 生成集最近邻距离 vs 真实局部半径
    nn_f  = NearestNeighbors(n_neighbors=k).fit(X_fake_min)
    dist_rf, _ = nn_f.kneighbors(X_real_min)       # [N_real, k]
    dist_real_to_fake = dist_rf[:, 0]              # [N_real]
    beta = float(np.mean(dist_real_to_fake <= radius_real))

    return alpha, beta


def compute_layer2_minority(
    real_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    minority_col: str,
    minority_val: Any,
    k_neighbors:  int = 5,
) -> Layer2Report:
    """
    Layer 2：少数类结构学习。

    参数
    ----
    minority_col : 用于识别少数类的列名（如 'label'、'diabetes'）
    minority_val : 少数类的取值（如 1、True、'positive'）
    k_neighbors  : α/β 计算使用的 k 值，建议 5~10

    注意
    ----
    minority_col 本身不参与相关矩阵计算（它是分层依据而非特征），
    但会自动保留在 num_cols/cat_cols 中如果原本就在其中。
    """
    report          = Layer2Report()
    report.k_neighbors = k_neighbors

    # ── 提取少数类子集 ───────────────────────────────────────────────
    real_min = real_df[real_df[minority_col] == minority_val].copy()
    fake_min = fake_df[fake_df[minority_col] == minority_val].copy()
    report.minority_n_real = len(real_min)
    report.minority_n_fake = len(fake_min)

    # 少数类内相关矩阵（排除分层列本身，它在子集里是常数）
    feat_num = [c for c in num_cols if c != minority_col]
    feat_cat = [c for c in cat_cols if c != minority_col]

    if len(feat_num) + len(feat_cat) > 1 and len(real_min) > 5:
        # ── 2.1 Dependency Preservation ─────────────────────────────
        report.minority_corr_real = _build_corr_matrix(real_min, feat_num, feat_cat)
        report.minority_corr_fake = _build_corr_matrix(fake_min, feat_num, feat_cat)
        report.minority_corr_diff = (
            report.minority_corr_real - report.minority_corr_fake
        ).abs()

        mask  = np.triu(np.ones(report.minority_corr_diff.shape, dtype=bool), k=1)
        upper = report.minority_corr_diff.values[mask]
        report.minority_corr_diff_mean = float(upper.mean())
        report.minority_corr_diff_fro  = float(np.linalg.norm(
            report.minority_corr_diff.values
        ))

    # ── 2.2 Minority Coverage（α-Precision, β-Recall）──────────────
    if len(real_min) >= k_neighbors + 1 and len(fake_min) >= k_neighbors + 1:
        all_feat_cols = feat_num + feat_cat

        # 用全集的均值/标准差标准化（保证尺度一致）
        real_all_enc, fake_all_enc = _standardize_for_distance(
            real_df[all_feat_cols],
            fake_df[all_feat_cols],
            [c for c in feat_cat],
        )
        # 从标准化后的全集中取少数类行
        real_min_mask = (real_df[minority_col] == minority_val).values
        fake_min_mask = (fake_df[minority_col] == minority_val).values
        X_real_min = real_all_enc[real_min_mask]
        X_fake_min = fake_all_enc[fake_min_mask]

        report.alpha_precision, report.beta_recall = _alpha_beta(
            X_real_min, X_fake_min, k=k_neighbors
        )
    else:
        warnings.warn(
            f"少数类样本量过小（real={len(real_min)}, fake={len(fake_min)}），"
            f"跳过 α/β 计算。建议 k_neighbors ≤ {min(len(real_min), len(fake_min)) - 1}"
        )

    return report


# ═════════════════════════════════════════════════════════════════════════════
# Layer 3：辅助验证
# ═════════════════════════════════════════════════════════════════════════════

def _c2st(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    cat_cols: list[str],
    n_folds: int,
    random_state: int,
) -> tuple[float, dict]:
    if not _HAS_XGB and not _HAS_LGB:
        warnings.warn("xgboost/lightgbm 均未安装，跳过 C2ST。")
        return float('nan'), {}

    real_enc = real_df.copy();  real_enc['__y__'] = 0
    fake_enc = fake_df.copy();  fake_enc['__y__'] = 1
    combined = pd.concat([real_enc, fake_enc], ignore_index=True)
    for col in cat_cols:
        combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
    y = combined.pop('__y__').values
    X = combined.values.astype(float)
    col_names = list(real_df.columns)

    if _HAS_XGB:
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', verbosity=0,
            random_state=random_state,
        )
    else:
        clf = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=random_state,
        )

    cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    auc = float(cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    fimp = (
        {col_names[i]: float(clf.feature_importances_[i]) for i in range(len(col_names))}
        if hasattr(clf, 'feature_importances_') else {}
    )
    return auc, fimp


def _dcr(
    fake_df:       pd.DataFrame,
    real_train_df: pd.DataFrame,
    real_hold_df:  pd.DataFrame,
    cat_cols:      list[str],
    max_samples:   int,
    random_state:  int,
) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """
    Distance to Closest Record。

    对每个生成样本，分别计算到训练集和 holdout 集最近邻的 L2 距离。
    用 KS 检验比较两个 DCR 分布是否有显著差异：
      p >> 0.05 → 两分布无显著差异 → 无明显记忆化风险
      p << 0.05 → 训练集 DCR 显著偏小 → 可能存在隐私泄露

    标准化：以训练集统计量为基准（与真实数据一致）。
    """
    rng = np.random.default_rng(random_state)

    X_fake, X_train = _standardize_for_distance(
        fake_df, real_train_df, cat_cols
    )
    _,      X_hold  = _standardize_for_distance(
        fake_df, real_hold_df, cat_cols
    )

    # 采样限制
    ns = min(max_samples, len(X_fake))
    idx_s = rng.choice(len(X_fake), ns, replace=False)
    X_fake_s = X_fake[idx_s]

    nn_train = NearestNeighbors(n_neighbors=1).fit(X_train)
    nn_hold  = NearestNeighbors(n_neighbors=1).fit(X_hold)

    dcr_train, _ = nn_train.kneighbors(X_fake_s)
    dcr_hold,  _ = nn_hold.kneighbors(X_fake_s)
    dcr_train = dcr_train[:, 0]
    dcr_hold  = dcr_hold[:, 0]

    ks_stat, ks_pval = stats.ks_2samp(dcr_train, dcr_hold)

    return (
        float(dcr_train.mean()),
        float(dcr_hold.mean()),
        float(ks_stat),
        float(ks_pval),
        dcr_train,
        dcr_hold,
    )


def compute_layer3_auxiliary(
    real_train_df: pd.DataFrame,
    real_hold_df:  pd.DataFrame,
    fake_df:       pd.DataFrame,
    num_cols:      list[str],
    cat_cols:      list[str],
    n_folds:       int = 5,
    dcr_max_samples: int = 2000,
    random_state:  int = 42,
) -> Layer3Report:
    """
    Layer 3：辅助验证。

    参数
    ----
    real_train_df : 训练集（DCR 隐私分析需要）
    real_hold_df  : holdout 集（DCR 隐私分析需要）
    fake_df       : 生成数据集
    """
    report = Layer3Report()

    # C2ST（全局联合分布可分性）
    report.c2st_auc, report.c2st_feature_importance = _c2st(
        real_train_df, fake_df, cat_cols, n_folds, random_state
    )

    # DCR（隐私/记忆化风险）
    (report.dcr_train_mean,
     report.dcr_hold_mean,
     report.dcr_ks_stat,
     report.dcr_ks_pval,
     report.dcr_train_scores,
     report.dcr_hold_scores) = _dcr(
        fake_df, real_train_df, real_hold_df,
        cat_cols, dcr_max_samples, random_state,
    )

    return report


# ═════════════════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════════════════

def evaluate(
    real_train_df: pd.DataFrame,
    fake_df:       pd.DataFrame,
    num_cols:      list[str],
    cat_cols:      list[str],
    minority_col:  str,
    minority_val:  Any,
    real_hold_df:  Optional[pd.DataFrame] = None,
    k_neighbors:   int  = 5,
    n_folds:       int  = 5,
    dcr_max_samples: int = 2000,
    random_state:  int  = 42,
    verbose:       bool = True,
    run_layers:    tuple = (1, 2, 3),
) -> FidelityReport:
    """
    三层评估一站式入口。

    参数
    ----
    real_train_df : 真实训练数据
    fake_df       : 生成数据
    num_cols      : 连续特征列名列表
    cat_cols      : 类别特征列名列表
    minority_col  : 少数类判别列（如 'label'、'diabetes'）
    minority_val  : 少数类取值（如 1）
    real_hold_df  : holdout 集（仅 Layer3 DCR 需要，None 则跳过 DCR）
    k_neighbors   : α/β 计算的 k 值
    n_folds       : C2ST 交叉验证折数
    dcr_max_samples: DCR 计算的最大采样数
    random_state  : 随机种子
    verbose       : 是否打印报告
    run_layers    : 运行哪些层，默认 (1, 2, 3)，可传 (1, 2) 跳过辅助验证

    返回
    ----
    FidelityReport，各层子报告可单独访问。
    """
    report = FidelityReport()

    if 1 in run_layers:
        report.layer1 = compute_layer1_fidelity(
            real_train_df, fake_df, num_cols, cat_cols
        )

    if 2 in run_layers:
        report.layer2 = compute_layer2_minority(
            real_train_df, fake_df, num_cols, cat_cols,
            minority_col=minority_col,
            minority_val=minority_val,
            k_neighbors=k_neighbors,
        )

    if 3 in run_layers:
        hold = real_hold_df if real_hold_df is not None else real_train_df
        if real_hold_df is None:
            warnings.warn("未提供 real_hold_df，DCR 将使用训练集作为 holdout 的替代（结果仅供参考）。")
        report.layer3 = compute_layer3_auxiliary(
            real_train_df=real_train_df,
            real_hold_df=hold,
            fake_df=fake_df,
            num_cols=num_cols,
            cat_cols=cat_cols,
            n_folds=n_folds,
            dcr_max_samples=dcr_max_samples,
            random_state=random_state,
        )

    if verbose:
        print(report)

    return report