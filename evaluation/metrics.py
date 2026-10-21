from __future__ import annotations
from dataclasses import dataclass, field
import time
from typing import Any, Literal, Optional
import warnings

from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from evaluation.utils import (_as_str, _cpt_scores, _ks_stat, _mixed_corr, _score_clf, _tvd_cat,
                              compute_shap_preservation)

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



@dataclass
class EvalConfig:
    default_mode: Literal['all', 'minority'] = 'all'
    minority_col: Optional[str] = None
    minority_val: Any = None
    metric_modes: dict[str, str] = field(default_factory=dict)
    # pre_filtered=True 时，default_mode / metric_modes 对 _subset() 不再有意义，
    # 但 minority_col 仍供 _active_cols() 使用
    pre_filtered: bool = False
    def get_mode(self, metric: str) -> str:
        return self.metric_modes.get(metric, self.default_mode)


def _subset(df: pd.DataFrame, mode: str, config: EvalConfig) -> pd.DataFrame:
    if config.pre_filtered:
        return df.copy() # 数据已预过滤，直接返回 
    """Return minority subset or full DataFrame based on mode."""
    if mode == 'minority':
        if config.minority_col is None:
            raise ValueError(
                "EvalConfig.minority_col must be set when using mode='minority'."
            )
        sub = df[df[config.minority_col] == config.minority_val]
        if len(sub) == 0:
            warnings.warn(
                f"Minority subset is empty "
                f"(col='{config.minority_col}', val={config.minority_val!r})."
            )
        return sub.copy()
    return df.copy()


def _active_cols(
    num_cols: list[str],
    cat_cols: list[str],
    mode: str,
    config: EvalConfig,
) -> tuple[list[str], list[str]]:
    """
    In minority mode, minority_col is constant across the subset → exclude it.
    Constant columns corrupt correlation matrices and distance computations.
    """
    should_exclude = (
        config.minority_col is not None
        and (mode == 'minority' or config.pre_filtered)
    )
    if should_exclude:
        n = [c for c in num_cols if c != config.minority_col]
        c = [c for c in cat_cols if c != config.minority_col]
        return n, c
    return list(num_cols), list(cat_cols)


# ==========================================================
# 参照型： （ref_df, fake_df, ...)
# shape, cpt, corr_diff, c2st, prdc, alpha_beta, shap
# ==========================================================

def metric_shape(
    ref_df:   pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    config:   EvalConfig,
    **kwargs,
) -> dict:
    """
    SDMetrics Column Shapes：数值列 KS 统计量、类别列 TVD（均为误差，0 = 相同）。
    Outputs : shape_err（列均值，越低越好）、shape_score = 1 − shape_err（SDMetrics 原始得分）、
              colshape_<col>（逐列得分 = KSComplement / TVComplement，进 detail）
    """
    mode = config.get_mode('shape')
    ref  = _subset(ref_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)

    per_col: dict[str, float] = {}
    for col in n_cols:
        per_col[col] = _ks_stat(ref[col].to_numpy(), fake[col].to_numpy())
    for col in c_cols:
        per_col[col] = _tvd_cat(ref[col].to_numpy(), fake[col].to_numpy())

    err = float(np.mean(list(per_col.values()))) if per_col else float('nan')
    result = {'shape_err': err, 'shape_score': 1.0 - err}
    for col, v in per_col.items():
        result[f'colshape_{col}'] = 1.0 - v
    return result


def metric_cpt(
    ref_df:   pd.DataFrame,
    fake_df:  pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    config:   EvalConfig,
    n_bins:   int = 10,
    **kwargs,
) -> dict:
    """
    SDMetrics Column Pair Trends（自实现，向量化）：
      num–num : 1 − |ρ_ref − ρ_fake| / 2（Pearson）
      含类别列 : 联合分布列联相似度 1 − ½ Σ|p_ref − p_fake|；数值列按 ref 取值范围等宽分 n_bins 箱
    Outputs : cpt_score_{nn,nc,cc,all}（原始得分，越高越好）、cpt_err_{...} = 1 − 得分（越低越好）、
              cpt_pair_<a>__<b>（逐对得分，进 detail）。all = 全部列对的平均。
    """
    mode = config.get_mode('cpt')
    ref  = _subset(ref_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)

    by: dict[str, list[float]] = {'nn': [], 'nc': [], 'cc': []}
    result: dict[str, float] = {}
    for a, b, kind, s in _cpt_scores(ref, fake, n_cols, c_cols, n_bins):
        by[kind].append(s)
        result[f'cpt_pair_{a}__{b}'] = s

    def _mean(v):
        return float(np.mean(v)) if len(v) else float('nan')

    groups = {**by, 'all': [s for v in by.values() for s in v]}
    for k, v in groups.items():
        sc = _mean(v)
        result[f'cpt_score_{k}'] = sc
        result[f'cpt_err_{k}'] = 1.0 - sc
    return result


_REF_CORR = {}

def metric_corr_diff(ref_df, fake_df, num_cols, cat_cols, config, **kwargs) -> dict:
    """
    分块关联强度差（|C_ref − C_fake| 的块内均值，越低越好）：
      nn: Spearman；nc: η²；cc: bias-corrected Cramér's V
    """
    mode = config.get_mode('corr_diff')
    ref  = _subset(ref_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)
    if len(n_cols) + len(c_cols) < 2:
        warnings.warn("corr_diff: fewer than 2 active feature columns, skipping.")
        return {'corr_diff_nn': float('nan'), 'corr_diff_nc': float('nan'), 'corr_diff_cc': float('nan')}
    key = (id(ref_df), mode, tuple(n_cols), tuple(c_cols))
    if key not in _REF_CORR:
        _REF_CORR[key] = _mixed_corr(ref, n_cols, c_cols)
    C_ref  = _REF_CORR[key]
    C_fake = _mixed_corr(fake, n_cols, c_cols)

    diff = (C_ref - C_fake).abs().values
    p = len(n_cols)

    def _upper(m):
        return m[np.triu_indices(m.shape[0], k=1)]

    def _mean(v):
        return float(np.mean(v)) if np.size(v) else float('nan')

    return {'corr_diff_nn': _mean(_upper(diff[:p, :p])),
            'corr_diff_nc': _mean(diff[:p, p:]),
            'corr_diff_cc': _mean(_upper(diff[p:, p:]))}


def metric_c2st(
    ref_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    config:       EvalConfig,
    n_folds:      int = 5,
    random_state: int = 42,
    **kwargs,
) -> dict:
    if not _HAS_XGB and not _HAS_LGB:
        warnings.warn("c2st: xgboost/lightgbm not installed, skipping.")
        return {'c2st_auc': float('nan')}

    mode = config.get_mode('c2st')
    ref  = _subset(ref_df, mode, config)
    fake = _subset(fake_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)
    all_feat = n_cols + c_cols

    r = ref[all_feat].copy();  r['__y__'] = 0
    f = fake[all_feat].copy(); f['__y__'] = 1
    combined = pd.concat([r, f], ignore_index=True)
    for col in c_cols:
        combined[col] = LabelEncoder().fit_transform(combined[col].astype(str))
    y = combined.pop('__y__').values
    X = combined.values.astype(float)

    clf = (
        XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', verbosity=0, random_state=random_state,
        )
        if _HAS_XGB else
        LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            verbose=-1, random_state=random_state,
        )
    )

    cv  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    auc = float(cross_val_score(clf, X, y, cv=cv, scoring='roc_auc').mean())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)

    result = {'c2st_auc': auc}
    if hasattr(clf, 'feature_importances_'):
        for col, imp in zip(all_feat, clf.feature_importances_):
            result[f'c2st_fimp_{col}'] = float(imp)
    return result


def _matrix(real, fake, num_cols, cat_cols):
    """类别列 one-hot、数值列按 real 的均值/方差标准化；返回 (X_real, X_fake)"""
    cols = num_cols + cat_cols
    both = pd.concat([real[cols], fake[cols]], keys=['r', 'f'])
    for c in cat_cols:
        both[c] = both[c].astype(str)
    X = pd.get_dummies(both, columns=cat_cols, dtype=float)
    mu, sd = X.loc['r'].mean(), X.loc['r'].std(ddof=0).replace(0, 1.0)
    X = (X - mu) / sd
    return X.loc['r'].to_numpy(), X.loc['f'].to_numpy()


import torch

def _dev():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

_MM = 'donot_use_mm_for_euclid_dist'   # 精确距离，避免矩阵乘法带来的数值误差

def _kth_nn_dist(X, k, bs=2048):
    """X 内每点到其第 k 近邻的距离（排除自身），与 prdc 包定义一致"""
    Xt = torch.as_tensor(X, dtype=torch.float32, device=_dev())
    out = []
    for i in range(0, len(Xt), bs):
        d = torch.cdist(Xt[i:i+bs], Xt, compute_mode=_MM)
        out.append(d.topk(k + 1, dim=1, largest=False).values[:, -1])
    return torch.cat(out)

def _prdc_gpu(xr, xf, k, bs=2048):
    dev = _dev()
    R = torch.as_tensor(xr, dtype=torch.float32, device=dev)
    F = torch.as_tensor(xf, dtype=torch.float32, device=dev)
    r_nn, f_nn = _kth_nn_dist(xr, k, bs), _kth_nn_dist(xf, k, bs)
    prec_hit = torch.zeros(len(F), dtype=torch.bool, device=dev)
    dens_cnt = torch.zeros(len(F), device=dev)
    rec_hit = cov_hit = 0
    for i in range(0, len(R), bs):
        d = torch.cdist(R[i:i+bs], F, compute_mode=_MM)      # (b, n_fake)
        inside = d < r_nn[i:i+bs, None]
        prec_hit |= inside.any(0)
        dens_cnt += inside.sum(0)
        rec_hit += (d < f_nn[None, :]).any(1).sum().item()
        cov_hit += (d.min(1).values < r_nn[i:i+bs]).sum().item()
    return dict(precision=prec_hit.float().mean().item(),
                recall=rec_hit / len(R),
                density=(dens_cnt.mean() / k).item(),
                coverage=cov_hit / len(R))

def _min_dist(Q, X, bs=1024):
    """Q 中每行到 X 的最近距离（等价于 NearestNeighbors(n_neighbors=1)）"""
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=_dev())
    out = []
    for i in range(0, len(Q), bs):
        q = torch.as_tensor(np.asarray(Q[i:i+bs]), dtype=torch.float32, device=_dev())
        out.append(torch.cdist(q, Xt, compute_mode=_MM).min(1).values.cpu())
    return torch.cat(out).numpy().astype(np.float64)

def metric_prdc(ref_df, fake_df, num_cols, cat_cols, config, k: int = 5, **kw) -> dict:
    #from prdc import compute_prdc
    n, c = _active_cols(num_cols, cat_cols, config.get_mode('prdc'), config)
    xr, xf = _matrix(ref_df, fake_df, n, c)

    names = ('precision', 'recall', 'density', 'coverage')
    if min(len(xr), len(xf)) <= k:
        return {f'prdc_{x}': float('nan') for x in names}
    #r = compute_prdc(real_features=xr, fake_features=xf, nearest_k=k)
    r = _prdc_gpu(xr, xf, k)
    return {f'prdc_{x}': float(r[x]) for x in names}


def _alpha_beta_naive_gpu(xr, xf, bs=2048, return_curve=False):
    dev = _dev()
    xr = np.asarray(xr, dtype=np.float64); xf = np.asarray(xf, dtype=np.float64)
    mn = xr.min(0); rg = xr.max(0) - mn; rg[rg == 0] = 1.0   # 同 MinMaxScaler 对常数列的处理
    X, S = (xr - mn) / rg, (xf - mn) / rg
    Xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
    St = torch.as_tensor(S, dtype=torch.float32, device=dev)
    r2r, r2s, arg = [], [], []
    for i in range(0, len(Xt), bs):
        a = Xt[i:i+bs]
        r2r.append(torch.cdist(a, Xt, compute_mode=_MM).topk(2, dim=1, largest=False).values[:, 1])
        v, ix = torch.cdist(a, St, compute_mode=_MM).min(1)
        r2s.append(v); arg.append(ix)
    r2r = torch.cat(r2r).cpu().numpy().astype(np.float64)
    r2s = torch.cat(r2s).cpu().numpy().astype(np.float64)
    arg = torch.cat(arg).cpu().numpy()

    alphas = np.linspace(0, 1, 30)
    emb = X.mean(0)
    Radii = np.quantile(np.sqrt(((X - emb) ** 2).sum(1)), alphas)
    sc = S.mean(0)
    s2c = np.sqrt(((S - emb) ** 2).sum(1))
    cd = np.sqrt(((S[arg] - sc) ** 2).sum(1))
    cR = np.quantile(cd, alphas)
    ap = np.array([(s2c <= R).mean() for R in Radii])
    near = r2s <= r2r
    bc = np.array([(near & (cd <= R)).mean() for R in cR])
    dp = 1 - np.abs(alphas - ap).sum() / alphas.sum()
    db = 1 - np.abs(alphas - bc).sum() / alphas.sum()
    if return_curve:
        return float(dp), float(db), ap, alphas
    return float(dp), float(db)


def alpha_diag(ref_df, fake_df, num_cols, cat_cols, random_state=42):
    """signed = ap - alpha：>0 合成样本更集中，<0 更分散；lo/mid/hi 为 alpha 的前/中/后三分之一"""
    rng = np.random.default_rng(random_state)
    out = {}
    for name, (n, c) in {'all': (num_cols, cat_cols), 'num_only': (num_cols, []),
                         'cat_only': ([], cat_cols)}.items():
        if not (n or c):
            continue
        xr, xf = _matrix(ref_df, fake_df, n, c)
        m = min(len(xr), len(xf))
        if len(xr) > m: xr = xr[rng.choice(len(xr), m, replace=False)]
        if len(xf) > m: xf = xf[rng.choice(len(xf), m, replace=False)]
        dp, _, ap, al = _alpha_beta_naive_gpu(xr, xf, return_curve=True)
        s = ap - al
        out[name] = dict(alpha_precision=dp, signed_mean=s.mean(),
                         lo=s[:10].mean(), mid=s[10:20].mean(), hi=s[20:].mean())
    return pd.DataFrame(out).T.round(4)


def metric_alpha_beta(ref_df, fake_df, num_cols, cat_cols, config,
                      random_state: int = 42, **kw) -> dict:
    from synthcity.metrics.eval_statistical import AlphaPrecision
    from synthcity.plugins.core.dataloader import GenericDataLoader
    n, c = _active_cols(num_cols, cat_cols, config.get_mode('alpha_beta'), config)
    xr, xf = _matrix(ref_df, fake_df, n, c)          # 标准化在下采样之前完成

    m = min(len(xr), len(xf))                        # synthcity 要求两边等长
    rng = np.random.default_rng(random_state)
    if len(xr) > m:
        xr = xr[rng.choice(len(xr), m, replace=False)]
    if len(xf) > m:
        xf = xf[rng.choice(len(xf), m, replace=False)]

    dp, db = _alpha_beta_naive_gpu(xr, xf)
    return {'alpha_precision': dp, 'beta_recall': db}


_NAN_SHAP = {'shap_corr': float('nan'), 'shap_cos': float('nan')}


def _y_str(s: pd.Series) -> pd.Series:
    return s.round().astype(int).astype(str) if s.dtype.kind in 'if' else s.astype(str)


def _prep(ref_df, fake_df, num_cols, cat_cols, label_col):
    """全部转为数值；标签放最后一列（correlation_utils 的约定）。
    类别/标签编码以 ref_df 为参照；fake 中未见过的类别 -> -1，未见过的标签行丢弃。"""
    feats = [c for c in num_cols + cat_cols if c != label_col]
    cats = [c for c in cat_cols if c != label_col]
    cmaps = {c: {v: i for i, v in enumerate(sorted(ref_df[c].astype(str).unique()))} for c in cats}
    ymap = {v: i for i, v in enumerate(sorted(_y_str(ref_df[label_col]).unique()))}

    def enc(df):
        X = df[feats].copy().reset_index(drop=True)
        for c in cats:
            X[c] = X[c].astype(str).map(cmaps[c]).fillna(-1)
        y = _y_str(df[label_col]).map(ymap).reset_index(drop=True)
        keep = y.notna()
        out = X[keep].astype(float)
        out[label_col] = y[keep].astype(int)
        return out.reset_index(drop=True)

    return enc(ref_df), enc(fake_df)


def metric_shap(ref_df, fake_df, num_cols, cat_cols, config: EvalConfig,
                **kwargs) -> dict:
    if config.minority_col is None:
        warnings.warn("shap: EvalConfig.minority_col (label) must be set, skipping.")
        return dict(_NAN_SHAP)
    tr, fk = _prep(ref_df, fake_df, num_cols, cat_cols, config.minority_col)
    if tr.iloc[:, -1].nunique() < 2 or fk.iloc[:, -1].nunique() < 2:
        warnings.warn("shap: both ref_df and fake_df need >=2 classes "
                      "(not applicable to minority-only fake data), skipping.")
        return dict(_NAN_SHAP)
    c, s = compute_shap_preservation(tr, fk)
    return {'shap_corr': float(c), 'shap_cos': float(s)}

# ==========================================================
# 下游型： （train_df, test_df, fake_df, ...)
# dcr, tstr
# ==========================================================

def gower_min_dist(train, queries, num_cols, cat_cols, bs: int = 1024):
    """
    每个 query 行到 train 的最近 Gower 距离（对全部列取平均）：
      数值列 |Δ| / range(train)（clip 到 [0,1]），类别列 0/1 失配（train 中未见类别视为失配）。
    实现：数值部分用 L1 cdist；类别失配数 = 类别列数 − one-hot 内积（矩阵乘法，避免逐列比较）。
    queries 为 DataFrame 列表，返回对应的距离数组列表（train one-hot 只构造一次）。
    """
    dev = _dev()
    q = len(cat_cols)
    d = len(num_cols) + q

    if num_cols:
        X = train[num_cols].to_numpy(dtype=float)
        lo = X.min(axis=0)
        rg = X.max(axis=0) - lo
        rg[rg == 0] = 1.0

        def _num(df):
            x = np.clip((df[num_cols].to_numpy(dtype=float) - lo) / rg, 0.0, 1.0)
            return torch.as_tensor(x, dtype=torch.float32, device=dev)

        Nt = _num(train)

    if q:
        cats = [np.unique(_as_str(train[c].to_numpy())) for c in cat_cols]
        offs = np.concatenate([[0], np.cumsum([len(k) + 1 for k in cats])])   # 每列多留 1 位给"未见类别"

        def _oh(df):
            idx = np.empty((len(df), q), dtype=np.int64)
            for j, c in enumerate(cat_cols):
                cd = pd.Categorical(_as_str(df[c].to_numpy()), categories=cats[j]).codes.astype(np.int64)
                idx[:, j] = np.where(cd < 0, len(cats[j]), cd) + offs[j]
            t = torch.zeros(len(df), int(offs[-1]), device=dev)
            return t.scatter_(1, torch.as_tensor(idx, device=dev), 1.0)

        Ot = _oh(train)

    outs = []
    for Q in queries:
        res = []
        for i in range(0, len(Q), bs):
            chunk = Q.iloc[i:i + bs]
            dist = torch.zeros(len(chunk), len(train), device=dev)
            if num_cols:
                dist += torch.cdist(_num(chunk), Nt, p=1.0)
            if q:
                dist += q - _oh(chunk) @ Ot.T
            res.append((dist.min(dim=1).values / d).cpu())
        outs.append(torch.cat(res).numpy().astype(np.float64))
    return outs


def holdout_dcr(train_df, test_df, num_cols, cat_cols, config) -> np.ndarray:
    """holdout real → train 的最近 Gower 距离，作为 DCR 基线（每折只需算一次）。"""
    mode  = config.get_mode('dcr')
    train = _subset(train_df, mode, config)
    test  = _subset(test_df,  mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)
    return gower_min_dist(train, [test], n_cols, c_cols)[0]


def metric_dcr(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    fake_df:   pd.DataFrame,
    num_cols:  list[str],
    cat_cols:  list[str],
    config:    EvalConfig,
    dcr_real:  np.ndarray | None = None,
    **kwargs,
) -> dict:
    """
    DCR（Gower 距离）：fake→train 与基线 holdout→train 比较。
    Outputs : dcr_mean（fake→train 均值）、dcr_ratio（dcr_mean / holdout 均值，≈1 无系统性记忆，明显 <1 疑似记忆）、
              dcr_ks（两分布的 KS 统计量）、dcr_exact（fake 中与某训练行完全相同的比例）
    dcr_real 传入则复用（run.py 每折算一次）。
    """
    mode  = config.get_mode('dcr')
    fake  = _subset(fake_df,  mode, config)
    train = _subset(train_df, mode, config)
    n_cols, c_cols = _active_cols(num_cols, cat_cols, mode, config)

    dcr_fake = gower_min_dist(train, [fake], n_cols, c_cols)[0]
    if dcr_real is None:
        dcr_real = holdout_dcr(train_df, test_df, num_cols, cat_cols, config)

    return {
        'dcr_mean':  float(dcr_fake.mean()),
        'dcr_ratio': float(dcr_fake.mean() / (dcr_real.mean() + 1e-12)),
        'dcr_ks':    _ks_stat(dcr_fake, dcr_real),
        'dcr_exact': float((dcr_fake <= 1e-9).mean()),
    }


def metric_tstr(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    fake_df:      pd.DataFrame,
    num_cols:     list[str],
    cat_cols:     list[str],
    config:       EvalConfig,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """
    TSTR evaluation with robust handling of unknown categories and label rounding.
    """

    _SFXS = ('roc_auc', 'pr_auc', 'f1', 'gmean')
    _nan  = {f'tstr_{s}': float('nan') for s in _SFXS}

    if not (_HAS_XGB or _HAS_LGB):
        warnings.warn("tstr: xgboost/lightgbm not installed, skipping.")
        return _nan
    if config.minority_col is None:
        warnings.warn("tstr: EvalConfig.minority_col must be set, skipping.")
        return _nan

    label_col = config.minority_col
    if label_col not in fake_df.columns:
        warnings.warn(f"tstr: fake_df missing '{label_col}'. Skipping.")
        return _nan

    # 特征列（排除标签）
    cat_feats = [c for c in cat_cols if c != label_col]
    num_feats = [c for c in num_cols if c != label_col]
    feat_cols = num_feats + cat_feats
    if not feat_cols:
        warnings.warn("tstr: no feature columns after excluding label, skipping.")
        return _nan

    # ----- 准备编码器及未知类别 fallback -----
    # 为每个类别特征计算众数（用于填充未知值）
    cat_modes = {}
    for col in cat_feats:
        # 转字符串，找到出现最多的类别
        mode_val = train_df[col].astype(str).mode()
        if len(mode_val) > 0:
            cat_modes[col] = mode_val.iloc[0]
        else:
            cat_modes[col] = "UNKNOWN"

    # Label encoder for features (fit on train)
    feat_encoders = {}
    for col in cat_feats:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str))
        feat_encoders[col] = le

    # Label encoder for target
    le_label = LabelEncoder()
    # 将标签转为字符串，同时做舍入（如果标签是数值）
    train_labels = train_df[label_col].copy()
    if train_labels.dtype.kind in 'if':  # 浮点或整数
        train_labels = train_labels.round().astype(int).astype(str)
    else:
        train_labels = train_labels.astype(str)
    le_label.fit(train_labels)
    # minority 编码
    try:
        minority_enc = le_label.transform([str(config.minority_val)])[0]
    except Exception:
        # 如果 minority_val 不在标签中，尝试使用二分类的 pos_label 推断
        uniq = le_label.classes_
        if len(uniq) == 2:
            # 默认将较小的类作为负类，较大的作为正类？不确定，但至少不报错
            minority_enc = 1  # 假定第二个类为正
            warnings.warn(f"tstr: minority_val '{config.minority_val}' not found; using class '{uniq[1]}' as positive.")
        else:
            warnings.warn("tstr: cannot determine minority encoding; returning NaN.")
            return _nan

    def _enc_X(df: pd.DataFrame) -> np.ndarray:
        """编码特征矩阵，未知类别用众数填充后编码"""
        df_enc = df[feat_cols].copy().reset_index(drop=True)
        for col in cat_feats:
            # 转换为字符串
            col_vals = df[col].astype(str)
            # 未知类别 -> 众数
            unknown_mask = ~col_vals.isin(feat_encoders[col].classes_)
            if unknown_mask.any():
                col_vals[unknown_mask] = cat_modes[col]
            # 编码
            df_enc[col] = feat_encoders[col].transform(col_vals)
        # 数值列直接保留
        return df_enc.values.astype(float)

    def _enc_y(df: pd.DataFrame) -> np.ndarray:
        """编码标签，并做舍入（如果是数值）"""
        y = df[label_col].copy()
        if y.dtype.kind in 'if':
            y = y.round().astype(int).astype(str)
        else:
            y = y.astype(str)
        return le_label.transform(y)

    X_fake = _enc_X(fake_df);  y_fake = _enc_y(fake_df)
    X_test = _enc_X(test_df);  y_test = _enc_y(test_df)

    def _build() -> object:
        return (
            XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', verbosity=0, random_state=random_state,
            ) if _HAS_XGB else
            LGBMClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                verbose=-1, random_state=random_state,
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        clf = _build(); clf.fit(X_fake, y_fake)

    s = _score_clf(clf, X_test, y_test, pos_label=minority_enc)
    return {f'tstr_{k}': s[k] for k in _SFXS}


def metric_real(train_df, test_df, num_cols, cat_cols, config, dcr_real=None, **kwargs) -> dict:
    """
    method='real' 行（ref=both，每折一行）：
      tstr_* = TRTR（用真实训练集训练，即 metric_tstr(train, test, train)）
      dcr_mean / dcr_exact = holdout→train 的 DCR 基线
    """
    if dcr_real is None:
        dcr_real = holdout_dcr(train_df, test_df, num_cols, cat_cols, config)
    row = {'dcr_mean': float(dcr_real.mean()), 'dcr_exact': float((dcr_real <= 1e-9).mean())}
    row.update(metric_tstr(train_df=train_df, test_df=test_df, fake_df=train_df,
                           num_cols=num_cols, cat_cols=cat_cols, config=config))
    return row
    