# evaluation/eva_single.py
"""单折评估器： 统计保真度、隐私、DCR、ML效用、类别占比"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, recall_score
)
import sklearn.base
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from core.data_processor import TabularDataProcessor


class TabularEvaluator:
    def __init__(
        self,
        df_real_train: pd.DataFrame,
        df_fake: pd.DataFrame,
        df_real_test: pd.DataFrame,
        target_col: str,
        cat_cols: list,
        ml_mode: str = None,
        classifiers: dict = None,
        num_transform: str = 'quantile',
        privacy_sample_size: int = 1000,
        random_state: int = 42
    ):
        """
        :param ml_mode: 'augmentation' or 'tstr'
        :param classifiers: 自定义分类器字典
        :param num_transform: 分类器的预处理方法
        :param privacy_sample_size: DCR计算时的下采样数量
        """
        self.target_col = target_col
        self.cat_cols = cat_cols
        self.num_cols = [c for c in df_real_train.columns
                         if c not in cat_cols and c != target_col]
        self.features = self.num_cols + self.cat_cols
        self.ml_mode = ml_mode
        self.privacy_sample_size = privacy_sample_size
        self.random_state = random_state

        # ---------- 剥离特征与标签 ----------
        self.X_real_train_raw = df_real_train[self.features].copy()
        self.y_real_train = df_real_train[target_col]

        self.has_fake_label = (target_col in df_fake.columns)
        self.X_fake_raw = df_fake[self.features].copy()
        self.y_fake = df_fake[target_col] if self.has_fake_label else pd.Series(1, index=self.X_fake_raw.index)

        self.X_real_test_raw = df_real_test[self.features].copy()
        self.y_real_test = df_real_test[target_col]

        # 内部分类用预处理（与生成模型训练时的预处理独立）
        self.ml_processor = TabularDataProcessor(
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            num_transform_method=num_transform
        )
        self.ml_processor.fit(self.X_real_train_raw)

        self.X_real_train_proc = self.ml_processor.transform(self.X_real_train_raw)
        self.X_fake_proc = self.ml_processor.transform(self.X_fake_raw)
        self.X_real_test_proc = self.ml_processor.transform(self.X_real_test_raw)

        # 分类器（可外部定制）
        self.classifiers = classifiers or {
            'RandomForest'  : RandomForestClassifier(n_estimators=100, random_state=42),
            'MLP'           : MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
            'XGBoost'       : XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'LightGBM'      : LGBMClassifier(random_state=42, verbose=-1)
        }

    # ---------- 1. 统计保真度 ----------
    def evaluate_statistical_fidelity(self) -> dict:
        """评估边缘分布和特征关联矩阵，越低越好"""
        results = {}

        ks_list = []
        for c in self.num_cols:
            real_c = self.X_real_train_raw[c].dropna()
            fake_c = self.X_fake_raw[c].dropna()
            if len(real_c) > 0 and len(fake_c) > 0:
                stat, _ = ks_2samp(real_c, fake_c)
                ks_list.append(stat if not np.isnan(stat) else 1.0)
            else:
                ks_list.append(1.0)

        results['Avg_KL_Distance'] = np.mean(ks_list) if ks_list else 0.0

        if len(self.num_cols) > 1:
            real_corr = self.X_real_train_raw[self.num_cols].corr(method='pearson').fillna(0).values
            fake_corr = self.X_fake_raw[self.num_cols].corr(method='pearson').fillna(0).values
            results['Correlation_MSE'] = float(np.mean((real_corr - fake_corr) ** 2))
        else:
            results['Correlation_MSE'] = 0.0

        return results

    # ---------- 2. 隐私评估（DCR）----------
    def evaluate_privacy(self) -> dict:

        sample_size = self.privacy_sample_size

        valid_real = self.X_real_train_raw.dropna()
        valid_fake = self.X_fake_raw.dropna()

        n_real = min(len(valid_real), sample_size)
        n_fake = min(len(valid_fake), sample_size)
        X_real_s = valid_real.sample(n_real, random_state = self.random_state) # short for x real sample
        X_fake_s = valid_fake.sample(n_fake, random_state = self.random_state)

        real_num = X_real_s[self.num_cols].values
        fake_num = X_fake_s[self.num_cols].values
        real_cat = X_real_s[self.cat_cols].values
        fake_cat = X_fake_s[self.cat_cols].values

        ranges = np.ptp(valid_real[self.num_cols].values, axis=0)
        ranges[ranges == 0] = 1e-5
        num_dist = cdist(fake_num / ranges, real_num / ranges, metric='cityblock')

        if len(self.cat_cols) > 0:
            cat_dist = cdist(fake_cat, real_cat, metric='hamming') * len(self.cat_cols)
        else:
            cat_dist = np.zeros_like(num_dist)

        gower = (num_dist + cat_dist) / len(self.features)
        min_dcr = np.min(gower, axis=1)

        return {
            'Mean_Gower_DCR': float(np.mean(min_dcr)),
            '5th_Min_Gower_DCR': float(np.percentile(min_dcr, 5))
        }

    # ---------- 3. ML效能评估 ----------
    def evaluate_ml_utility(self, mode: str = None) -> dict:
        """
        下游分类效用评估
        :param mode: 覆盖构造时传入的ml_mode
        """
        mode = mode or self.ml_mode

        if mode == 'augmentation':
            X_train = pd.concat([self.X_real_train_proc, self.X_fake_proc], axis=0)
            y_train = pd.concat([self.y_real_train, self.y_fake], axis=0)
            shuffle_idx = np.random.permuation(len(X_train))
            X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
            y_train = y_train.iloc[shuffle_idx]
        elif mode == 'tstr':
            X_train = self.X_fake_proc
            y_train = self.y_fake
        else:
            raise ValueError(f"不支持的评估模式：{mode}")

        ml_results = {}
        for clf_name, clf_base in self.classifiers.items():
            clf_aug = sklearn.base.clone(clf_base)

            # Baseline: 仅真实训练集
            clf_base.fit(self.X_real_train_proc, self.y_real_train)
            preds_base = clf_base.predict(self.X_real_test_proc)
            probs_base = (clf_base.predict_proba(self.X_real_test_proc)[:, 1] if hasattr(clf_base, "predict_proba") else preds_base)
            

            # Augmented / TSTR
            clf_aug.fit(X_train, y_train)
            preds_aug = clf_aug.predict(self.X_real_test_proc)
            probs_aug = (clf_aug.predict_proba(self.X_real_test_proc)[:, 1] if hasattr(clf_aug, "predict_proba") else preds_aug)
            
            ml_results[clf_name] = {
                'Base_ROC-AUC': roc_auc_score(self.y_real_test, probs_base),
                'Aug_ROC-AUC': roc_auc_score(self.y_real_test, probs_aug),
                'Base_PR-AUC': average_precision_score(self.y_real_test, probs_base),
                'Aug_PR-AUC': average_precision_score(self.y_real_test, probs_aug),
                'Base_F1': f1_score(self.y_real_test, preds_base),
                'Aug_F1': f1_score(self.y_real_test, preds_aug),
                'Base_Recall': recall_score(self.y_real_test, preds_base),
                'Aug_Recall': recall_score(self.y_real_test, preds_aug)
            }
        return ml_results
    
    # ---------- 4. 类别占比（单折版本） ----------
    def evaluate_categorical_proportions(self) -> pd.DataFrame:
        """
        输出各类别特征在真实训练集与合成集中的占比对比
        :return: 以MultiIndex(feature, category)为索引的DataFrame
        """
        cols = list(self.cat_cols)
        
        if self.ml_mode == 'tstr':
            cols.append(self.target_col)

        records = []
        for col in cols:
            if col == self.target_col:
                real_prob = self.y_real_train.value_counts(normalize=True) * 100
                fake_prob = self.y_fake.value_counts(normalize=True) * 100
            else:
                real_prob = self.X_real_train_raw[col].value_counts(normalize=True) * 100
                fake_prob = self.X_fake_raw[col].value_counts(normalize=True) * 100
            all_cats = real_prob.index.union(fake_prob.index)
            for cat in all_cats:
                records.append({
                    'Feature': col,
                    'Category': cat,
                    'Real_Train(%)': round(real_prob.get(cat, 0.0), 2),
                    'Fake(%)': round(fake_prob.get(cat, 0.0), 2),
                })
        
        return pd.DataFrame(records).set_index(['Feature', 'Category'])

    # ---------- 完整单折报告 ----------
    def run_full_evaluation(self):
        stats = self.evaluate_statistical_fidelity()
        privacy = self.evaluate_privacy()
        ml_res = self.evaluate_ml_utility()
        cat_proportion = self.evaluate_categorical_proportions() # DataFrame
        return {**stats, **privacy, 'ml_utility': ml_res, 'categorical_proportions': cat_proportion}