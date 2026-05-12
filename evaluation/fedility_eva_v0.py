# core/evaluation.py
"""
Tabular Synthetic Data Evaluation Suite
========================================
超越单列 KL，从三个层次评估生成质量：

  Level 1 —— 边缘分布（Marginal）
      per-column KL divergence（已有，保留作为基线）

  Level 2 —— 成对联合结构（Pairwise Joint）
      Correlation Matrix Difference
        · 连续-连续：Spearman ρ（对非线性/非高斯更鲁棒）
        · 类别-类别：Cramér's V
        · 连续-类别：Point-Biserial（二值类别）/ eta²（多值类别）
      CorrDiff = mean(|C_real − C_fake|)  +  Frobenius norm

  Level 3 —— 高维联合分布（Global Joint）
      C2ST  : XGBoost 真假分类器 AUC（0.5 = 完美，1.0 = 完全可分）
      MMD   : RBF kernel，使用 median heuristic 自动选带宽

  汇总
      EvaluationReport dataclass，包含全部指标和可读 summary。

使用方法
--------
from core.evaluation import evaluate

report = evaluate(
    real_df   = real_df,       # pd.DataFrame，真实数据
    fake_df   = fake_df,       # pd.DataFrame，生成数据（同列名）
    num_cols  = [...],         # 连续列名列表
    cat_cols  = [...],         # 类别列名列表
    verbose   = True,          # 打印摘要
)

# 所有指标
print(report)

# 单项访问
print(report.c2st_auc)
print(report.corr_diff_mean)
print(report.mmd)

依赖
----
pip install pandas numpy scipy scikit-learn xgboost
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import kl_div as scipy_kl_div
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

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


# ─────────────────────────────────────────────────────────────────────────────
# 结果容器
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvaluationReport:
    # Level 1：边缘分布
    marginal_kl: dict[str, float]         = field(default_factory=dict)
    avg_kl: float                          = 0.0

    # Level 2：成对联合结构
    corr_real: pd.DataFrame | None        = None
    corr_fake: pd.DataFrame | None        = None
    corr_diff: pd.DataFrame | None        = None   # |C_real - C_fake|
    corr_diff_mean: float                  = 0.0   # CorrDiff（平均绝对差）
    corr_diff_fro: float                   = 0.0   # Frobenius norm

    # Level 3：高维联合分布
    c2st_auc: float                        = 0.0
    c2st_feature_importance: dict         = field(default_factory=dict)
    mmd: float                             = 0.0

    def __repr__(self) -> str:
        lines = [
            "=" * 56,
            "  Synthetic Data Evaluation Report",
            "=" * 56,
            "",
            "── Level 1  Marginal KL Divergence ──────────────────",
            f"  Avg KL:  {self.avg_kl:.5f}",
        ]
        for col, kl in self.marginal_kl.items():
            lines.append(f"    {col:<24s}  {kl:.5f}")

        lines += [
            "",
            "── Level 2  Correlation Matrix Difference ───────────",
            f"  CorrDiff (mean |ΔC|):  {self.corr_diff_mean:.5f}",
            f"  CorrDiff (Frobenius):  {self.corr_diff_fro:.5f}",
        ]
        if self.corr_diff is not None:
            # 按差异排名，列出 top-5 偏差最大的对
            upper = (
                self.corr_diff.where(
                    np.triu(np.ones(self.corr_diff.shape, dtype=bool), k=1)
                )
                .stack()
                .sort_values(ascending=False)
            )
            lines.append("  Top-5 most deviated pairs:")
            for (r, c), v in upper.head(5).items():
                lines.append(f"    {r}  ↔  {c:<24s}  Δ={v:.4f}")

        lines += [
            "",
            "── Level 3  Global Joint Distribution ───────────────",
            f"  C2ST AUC:  {self.c2st_auc:.4f}  "
            f"({'≈ perfect' if self.c2st_auc < 0.55 else '⚠ distinguishable' if self.c2st_auc < 0.75 else '✗ highly distinguishable'})",
            f"  MMD:       {self.mmd:.6f}",
        ]
        if self.c2st_feature_importance:
            top_feats = sorted(
                self.c2st_feature_importance.items(), key=lambda x: -x[1]
            )[:5]
            lines.append("  C2ST discriminative features (top-5):")
            for fname, imp in top_feats:
                lines.append(f"    {fname:<28s}  importance={imp:.4f}")

        lines.append("=" * 56)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Level 1：边缘分布 KL
# ─────────────────────────────────────────────────────────────────────────────

def _marginal_kl_continuous(real: np.ndarray, fake: np.ndarray, n_bins: int = 50) -> float:
    """
    直方图估计 + KL 散度。
    使用真实数据的分位数边界，确保 fake 也能正确被捕获。
    """
    combined = np.concatenate([real, fake])
    bins = np.histogram_bin_edges(combined, bins=n_bins)
    p, _ = np.histogram(real, bins=bins, density=True)
    q, _ = np.histogram(fake, bins=bins, density=True)

    # 平滑防止 0/log(0)
    eps = 1e-8
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _marginal_kl_categorical(real: np.ndarray, fake: np.ndarray) -> float:
    cats = np.union1d(np.unique(real), np.unique(fake))
    eps  = 1e-8
    p = np.array([np.mean(real == c) for c in cats]) + eps;  p /= p.sum()
    q = np.array([np.mean(fake == c) for c in cats]) + eps;  q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_marginal_kl(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> dict[str, float]:
    kl = {}
    for col in num_cols:
        kl[col] = _marginal_kl_continuous(real_df[col].values, fake_df[col].values)
    for col in cat_cols:
        kl[col] = _marginal_kl_categorical(real_df[col].values, fake_df[col].values)
    return kl


# ─────────────────────────────────────────────────────────────────────────────
# Level 2：相关矩阵差异
# ─────────────────────────────────────────────────────────────────────────────

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关，对单调非线性关系更鲁棒"""
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return 0.0
    return float(stats.spearmanr(a, b).statistic)


def _cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cramér's V：衡量两个类别变量的关联强度，[0, 1]。
    使用 bias-corrected 版本（Bergsma & Wicher, 2013）。
    """
    ct   = pd.crosstab(a, b).values.astype(float)
    n    = ct.sum()
    if n == 0:
        return 0.0
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    r, k = ct.shape
    # bias-corrected
    phi2 = max(0.0, chi2 / n - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom  = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2 / denom))


def _eta_squared(num: np.ndarray, cat: np.ndarray) -> float:
    """
    η²（eta squared）：类别变量解释连续变量方差的比例，[0, 1]。
    用于连续-类别对，等价于单因素 ANOVA 中的效应量。
    """
    grand_mean = num.mean()
    cats       = np.unique(cat)
    ss_between = sum(
        np.sum(cat == c) * (num[cat == c].mean() - grand_mean) ** 2
        for c in cats
        if np.sum(cat == c) > 0
    )
    ss_total = np.sum((num - grand_mean) ** 2)
    if ss_total < 1e-10:
        return 0.0
    return float(ss_between / ss_total)


def _build_corr_matrix(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> pd.DataFrame:
    """
    构建混合类型相关矩阵：
      num-num  → Spearman ρ（有方向，[-1, 1]）
      cat-cat  → Cramér's V（无方向，[0, 1]）
      num-cat  → η²（无方向，[0, 1]）

    注意：不同对的取值范围不同，差值矩阵仍然有意义（两个数据集
    使用相同的度量），但跨类型比较时应注意解释。
    """
    all_cols = num_cols + cat_cols
    n        = len(all_cols)
    C        = np.zeros((n, n))
    num_set  = set(num_cols)
    cat_set  = set(cat_cols)

    for i, ci in enumerate(all_cols):
        for j, cj in enumerate(all_cols):
            if i == j:
                C[i, j] = 1.0
                continue
            if j < i:
                C[i, j] = C[j, i]   # 对称
                continue

            ai, aj = df[ci].values, df[cj].values

            if ci in num_set and cj in num_set:
                C[i, j] = _spearman(ai, aj)
            elif ci in cat_set and cj in cat_set:
                C[i, j] = _cramers_v(ai, aj)
            else:
                # 连续-类别：num 方向不变，cat 方向取 η²
                num_arr = ai if ci in num_set else aj
                cat_arr = aj if ci in num_set else ai
                C[i, j] = _eta_squared(num_arr.astype(float), cat_arr)

    return pd.DataFrame(C, index=all_cols, columns=all_cols)


def compute_correlation_diff(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
    """
    返回 (corr_real, corr_fake, corr_diff, mean_abs_diff, frobenius_norm)
    """
    corr_real = _build_corr_matrix(real_df, num_cols, cat_cols)
    corr_fake = _build_corr_matrix(fake_df, num_cols, cat_cols)
    corr_diff = (corr_real - corr_fake).abs()

    # 只取上三角（对角线除外），避免重复计数
    mask     = np.triu(np.ones(corr_diff.shape, dtype=bool), k=1)
    upper    = corr_diff.values[mask]
    mean_abs = float(upper.mean())
    fro      = float(np.linalg.norm(corr_diff.values))

    return corr_real, corr_fake, corr_diff, mean_abs, fro


# ─────────────────────────────────────────────────────────────────────────────
# Level 3-a：C2ST（Classifier Two-Sample Test）
# ─────────────────────────────────────────────────────────────────────────────

def _encode_for_classifier(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    合并真假数据，对类别列做 LabelEncoding，返回 (X, y)。
    y=0 为真实，y=1 为生成。
    """
    real = real_df.copy();  real['__label__'] = 0
    fake = fake_df.copy();  fake['__label__'] = 1
    combined = pd.concat([real, fake], ignore_index=True)

    for col in cat_cols:
        le = LabelEncoder()
        combined[col] = le.fit_transform(combined[col].astype(str))

    y = combined.pop('__label__').values
    X = combined.values.astype(float)
    return X, y


def compute_c2st(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    cat_cols: list[str],
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[float, dict[str, float]]:
    """
    训练 XGBoost（或 LightGBM）分类器区分真假数据。

    返回
    ----
    auc : float
        5 折 CV 平均 ROC-AUC。0.5 = 完美不可分，1.0 = 完全可分。
    feature_importance : dict
        特征名 → 重要性（用于诊断哪列驱动了真假差异）。

    解读
    ----
    AUC < 0.55  : 优秀，真假数据几乎不可分
    0.55–0.65   : 良好，存在轻微可分结构
    0.65–0.75   : 一般，联合分布有明显漏洞
    > 0.75      : 较差，生成数据与真实数据差异显著
    """
    if not _HAS_XGB and not _HAS_LGB:
        warnings.warn("xgboost 和 lightgbm 均未安装，跳过 C2ST。pip install xgboost")
        return float('nan'), {}

    X, y = _encode_for_classifier(real_df, fake_df, cat_cols)
    col_names = list(real_df.columns)

    if _HAS_XGB:
        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            random_state=random_state,
        )
    else:
        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            verbose=-1,
            random_state=random_state,
        )

    cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    auc = float(cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean())

    # 在全量数据上 fit 一次以获取 feature importance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)

    if hasattr(clf, 'feature_importances_'):
        imp  = clf.feature_importances_
        fimp = {col_names[i]: float(imp[i]) for i in range(len(col_names))}
    else:
        fimp = {}

    return auc, fimp


# ─────────────────────────────────────────────────────────────────────────────
# Level 3-b：MMD（Maximum Mean Discrepancy）
# ─────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    """RBF kernel: K(x,y) = exp(-||x-y||² / (2σ²))"""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    XY = X @ Y.T
    sq_dists = XX + YY.T - 2 * XY
    return np.exp(-sq_dists / (2 * bandwidth ** 2))


def compute_mmd(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    cat_cols: list[str],
    max_samples: int = 2000,
    random_state: int = 42,
) -> float:
    """
    无偏 MMD² 估计（RBF kernel，median heuristic 自动选带宽）。

    为什么用 median heuristic
    -------------------------
    MMD 对 kernel 带宽非常敏感。Median heuristic（σ = 样本间距离的中位数）
    是 Gretton et al. 的标准推荐，在实践中对大多数表格数据表现稳健，
    无需手动调参。

    返回
    ----
    float：MMD²，越小越好（0 表示分布完全相同）。

    注意
    ----
    类别列使用 LabelEncoding 后参与距离计算，这在混合数据上是近似处理，
    对于细粒度类别差异可能低估或高估。当前阶段作为全局信号使用。
    """
    rng = np.random.default_rng(random_state)

    # 编码
    X_r, _ = _encode_for_classifier(real_df, real_df, cat_cols)   # 只用编码
    X_f, _ = _encode_for_classifier(fake_df, fake_df, cat_cols)

    # 重新编码：用真实数据的 LabelEncoder fit，再 transform fake
    real_enc = real_df.copy()
    fake_enc = fake_df.copy()
    for col in cat_cols:
        le = LabelEncoder().fit(real_df[col].astype(str))
        real_enc[col] = le.transform(real_df[col].astype(str))
        # fake 中可能有 unseen label，统一映射为 -1
        fake_enc[col] = fake_df[col].astype(str).apply(
            lambda v: le.transform([v])[0] if v in le.classes_ else -1
        )

    X_r = real_enc.values.astype(float)
    X_f = fake_enc.values.astype(float)

    # 采样（避免 O(n²) 计算过慢）
    nr = min(max_samples, len(X_r))
    nf = min(max_samples, len(X_f))
    X_r = X_r[rng.choice(len(X_r), nr, replace=False)]
    X_f = X_f[rng.choice(len(X_f), nf, replace=False)]

    # Standardize（让不同量纲的列对距离贡献均衡）
    mu  = X_r.mean(axis=0);  sigma = X_r.std(axis=0) + 1e-8
    X_r = (X_r - mu) / sigma
    X_f = (X_f - mu) / sigma

    # Median heuristic：用真假数据合并后的成对距离中位数作为带宽
    combined  = np.vstack([X_r, X_f])
    n_sample  = min(500, len(combined))
    sub       = combined[rng.choice(len(combined), n_sample, replace=False)]
    dists     = np.sqrt(np.sum((sub[:, None] - sub[None, :]) ** 2, axis=-1))
    bandwidth = float(np.median(dists[dists > 0]))
    if bandwidth < 1e-8:
        bandwidth = 1.0

    # 无偏 MMD² 估计
    # MMD²(P,Q) = E[K(x,x')] - 2E[K(x,y)] + E[K(y,y')]
    K_rr = _rbf_kernel(X_r, X_r, bandwidth)
    K_ff = _rbf_kernel(X_f, X_f, bandwidth)
    K_rf = _rbf_kernel(X_r, X_f, bandwidth)

    # 去掉对角线（无偏估计）
    nr, nf = len(X_r), len(X_f)
    np.fill_diagonal(K_rr, 0)
    np.fill_diagonal(K_ff, 0)
    mmd2 = (K_rr.sum() / (nr * (nr - 1))
            + K_ff.sum() / (nf * (nf - 1))
            - 2 * K_rf.mean())

    return float(max(mmd2, 0.0))   # 数值误差可能导致极小负值，clip 到 0


# ─────────────────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    real_df: pd.DataFrame,
    fake_df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    n_folds: int          = 5,
    mmd_max_samples: int  = 2000,
    random_state: int     = 42,
    verbose: bool         = True,
) -> EvaluationReport:
    """
    一站式评估函数。

    参数
    ----
    real_df / fake_df : 真实 / 生成数据，pd.DataFrame，列名相同
    num_cols          : 连续列名列表
    cat_cols          : 类别列名列表
    n_folds           : C2ST 交叉验证折数
    mmd_max_samples   : MMD 计算的最大采样数（控制计算量）
    random_state      : 随机种子
    verbose           : 是否打印 EvaluationReport

    返回
    ----
    EvaluationReport dataclass，包含全部指标。
    """
    report = EvaluationReport()

    # ── Level 1：边缘分布 KL ──────────────────────────────────────────
    report.marginal_kl = compute_marginal_kl(real_df, fake_df, num_cols, cat_cols)
    report.avg_kl      = float(np.mean(list(report.marginal_kl.values())))

    # ── Level 2：相关矩阵差异 ─────────────────────────────────────────
    (report.corr_real,
     report.corr_fake,
     report.corr_diff,
     report.corr_diff_mean,
     report.corr_diff_fro) = compute_correlation_diff(real_df, fake_df, num_cols, cat_cols)

    # ── Level 3-a：C2ST ───────────────────────────────────────────────
    report.c2st_auc, report.c2st_feature_importance = compute_c2st(
        real_df, fake_df, cat_cols,
        n_folds=n_folds, random_state=random_state,
    )

    # ── Level 3-b：MMD ────────────────────────────────────────────────
    report.mmd = compute_mmd(
        real_df, fake_df, cat_cols,
        max_samples=mmd_max_samples, random_state=random_state,
    )

    if verbose:
        print(report)

    return report