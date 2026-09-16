import shap
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier


def compute_crps(train: pd.DataFrame, syn: pd.DataFrame, top_k: int = 10):
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    x_syn = syn.iloc[:, :-1]
    y_syn = syn.iloc[:, -1]

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    clf.fit(x_train, y_train)
    importance = clf.feature_importances_
    
    top_idx = np.argsort(importance)[::-1][:top_k]
    top_features = x_train.columns[top_idx]

    corr_real = (x_train[top_features].corr(method='spearman').values)
    corr_syn  = (x_syn[top_features].corr(method='spearman').values)

    fro_diff = np.linalg.norm(corr_real - corr_syn, ord='fro')
    fro_real = np.linalg.norm(corr_real, ord='fro')

    crps_F = 1 - fro_diff / (fro_real + 1e-8)

    idx = np.triu_indices_from(corr_real, k=1)
    r_real = corr_real[idx]
    r_syn  = corr_syn[idx]

    crps_S = spearmanr(r_real, r_syn).correlation

    return crps_F, crps_S


def compute_shap_preservation(train: pd.DataFrame, syn: pd.DataFrame):
    x_real = train.iloc[:, :-1]
    y_real = train.iloc[:, -1]

    x_syn = syn.iloc[:, :-1]
    y_syn = syn.iloc[:, -1]

    clf_real = XGBClassifier(
        n_estimators=300,
        max_depath=5,
        learning_rate=0.05,
        random_state=42
    )
    clf_real.fit(x_real, y_real)

    clf_syn = XGBClassifier(
        n_estimators=300,
        max_depath=5,
        learning_rate=0.05,
        random_state=42
    )
    clf_syn.fit(x_syn, y_syn)

    explainer_real = shap.TreeExplainer(clf_real)
    explainer_syn = shap.TreeExplainer(clf_syn)

    shap_real = explainer_real.shap_values(x_real)
    shap_syn = explainer_syn.shap_values(x_real)

    imp_real = np.abs(shap_real).mean(axis=0)
    imp_syn  = np.abs(shap_syn).mean(axis=0)

    shap_corr = spearmanr(imp_real, imp_syn).correlation
    #mae = np.mean(np.abs(imp_real - imp_syn))
    cos = cosine_similarity(imp_real.reshape(1, -1), imp_syn.reshape(1, -1))[0,0]

    return shap_corr, cos

    
