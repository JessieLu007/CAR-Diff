import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class TabularEvaluator:
    def __init__(self, X_real_train, y_real_train, X_fake, y_fake, 
                 X_real_test, y_real_test, num_cols, cat_cols):
        """
        :param X_real_train: 真实的训练特征 (DataFrame)
        :param y_real_train: 真实的训练标签 (Series 或 DataFrame)
        :param X_fake: 生成的特征 (DataFrame)
        :param y_fake: 生成的标签 (Series 或 list, 例如全是 1)
        :param X_real_test: 真实的测试特征 (DataFrame)
        :param y_real_test: 真实的测试标签 (Series 或 DataFrame)
        :param num_cols: 数值特征列名列表
        :param cat_cols: 类别特征列名列表 (不含标签)
        """
        self.X_real_train = X_real_train.copy()
        # 统一转为一维numpy数组，防止sklearn报错
        self.y_real_train = np.ravel(y_real_train)

        self.X_fake = X_fake.copy()
        self.y_fake = np.ravel(y_fake)

        self.X_real_test = X_real_test.copy()
        self.y_real_test = np.ravel(y_real_test)

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.features = self.num_cols + self.cat_cols

    def evaluate_statistical_fidelity(self):
        """评估分布拟合度（越低越好）"""
        results = {}

        # A. 边缘分布拟合（Marginal Distribution)
        # 使用KS检验（Kolmogorov-Smirnov）计算连续数值特征的分布距离
        ks_distances = [ks_2samp(self.X_real_train[c], self.X_fake[c])[0] for c in self.num_cols]
        results['Avg_KS_Distance'] = np.mean(ks_distances)

        # B. 联合分布关联度（Joint Distribution / Correlation)
        # 解决属性间关联度丢失问题，计算真实数据与生成数据的特征相关性矩阵，并求它们之间的均方误差
        real_corr = self.X_real_train[self.num_cols].corr(method='pearson').fillna(0).values
        fake_corr = self.X_fake[self.num_cols].corr(method='pearson').fillna(0).values
        results['Correlation_MSE'] = np.mean((real_corr - fake_corr) ** 2)

        return results

    def evaluate_privacy(self, sample_size=1000):
        # 评估隐私保护程度（DCR），为避免计算过慢，对数据进行下采样并标准化处理
        # 提取纯特征，并对齐顺序
        X_real = self.X_real_train[self.features].copy()
        X_fake = self.X_fake[self.features].copy()

        n_real, n_fake = min(len(X_real), sample_size), min(len(X_fake), sample_size)
        X_real_sample = X_real.sample(n_real, random_state=42)
        X_fake_sample = X_fake.sample(n_fake, random_state=42)

        # 分离数值和类别特征
        real_num = X_real_sample[self.num_cols].values
        fake_num = X_fake_sample[self.num_cols].values
        real_cat = X_real_sample[self.cat_cols].values
        fake_cat = X_fake_sample[self.cat_cols].values

        # 1. 计算数值特征的距离
        ranges = np.ptp(self.X_real_train[self.num_cols].values, axis=0)
        ranges[ranges == 0] = 1e-5 # 防止除零
        # cdist(metric='cityblock')计算的是绝对值之和，需要除以ranges
        num_dist = cdist(fake_num / ranges, real_num / ranges, metric='cityblock')

        # 2. 计算类别特征的距离（汉明距离*类别特征总数）
        # cdist(metric='hamming')返回的是不匹配的比例，乘以长度即为不匹配的个数
        if len(self.cat_cols) > 0:
            cat_dist = cdist(fake_cat, real_cat, metric='hamming') * len(self.cat_cols)
        else:
            cat_dist = 0

        # 3. Gower距离 = （数值距离之和 + 类别距离之和）/ 总特征数
        # 范围在[0,1]之间，0代表完全一致
        gower_matrix = (num_dist + cat_dist) / len(self.features)

        min_distances = np.min(gower_matrix, axis=1)

        return {
            'Mean_Gower_DCR': np.mean(min_distances),
            '5th_Min_Glower_DCR': np.percentile(min_distances, 5)
        }


    def evaluate_ml_utility(self, mode='augmentation'):
        '''
        评估下游分类效用
        :param mode: 'augmentation' (真实+生成 混合训练) 或 'tstr' (Train on Synthetic, Test on Real)
        '''
        # 使用随机森林，随机且不需要调参，适合作为分类基准
        clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_base.fit(self.X_real_train, self.y_real_train)
        preds_base = clf_base.predict(self.X_real_test)
        probs_base = clf_base.predict_proba(self.X_real_test)[:, 1]
        base_auc = roc_auc_score(self.y_real_test, probs_base)
        base_f1 = f1_score(self.y_real_test, preds_base) # , average='macro'

        # 准备增强模型（Augmented: 真实训练集 + 生成集）
        if mode == 'augmentation':
            X_aug_train = pd.concat([self.X_real_train, self.X_fake], axis=0)
            y_aug_train = np.concatenate([self.y_real_train, self.y_fake], axis=0)
        elif mode == 'tstr': # TSTR: 只用合成数据训练，测试真实数据（考验生成数据的独立可用性）
            X_aug_train = self.X_fake
            y_aug_train = self.y_fake
        else:
            raise ValueError(f"不支持的评估模式：{mode}")

        clf_aug = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_aug.fit(X_aug_train, y_aug_train)

        preds_aug = clf_aug.predict(self.X_real_test)
        probs_aug = clf_aug.predict_proba(self.X_real_test)[:, 1]

        aug_auc = roc_auc_score(self.y_real_test, probs_aug)
        aug_f1 = f1_score(self.y_real_test, preds_aug) #, average='macro'

        return {
            'Baseline_AUC': base_auc,
            'Augmented_AUC': aug_auc,
            'AUC_Delta': aug_auc - base_auc,
            'Baseline_Macro_F1': base_f1,
            'Augmented_Macro_F1': aug_f1,
            'F1_Delta':aug_f1 - base_f1
        }
    
    def run_full_evaluation(self, test_mode='augmentation'):
        print('\n' + '='*50)
        print("📊 开始数据质量与效用全面评估 (Tabular Evaluation)")
        print("="*50)

        stats_res = self.evaluate_statistical_fidelity()
        print(f"📉 [分布拟合度] (越低越好):")
        print(f"   - Avg KS Distance (数值边缘分布差异): {stats_res['Avg_KS_Distance']:.4f}")
        print(f"   - Correlation MSE (特征关联矩阵误差): {stats_res['Correlation_MSE']:.4f}")

        privacy = self.evaluate_privacy()
        print(f"\n🛡️ [隐私保护度 (DCR)]:")
        print(f"   - Mean Gower DCR: {privacy['Mean_Gower_DCR']:.4f} (适中为佳，防止过拟合)")
        print(f"   - 5% Min Gower DCR: {privacy['5th_Min_Glower_DCR']:.4f} (太低有复读机嫌疑)")

        ml_res = self.evaluate_ml_utility(mode=test_mode)
        print(f"\n🚀 [下游分类效用 - 数据增强模式] (越高越好):")
        print(f"   - Baseline AUC: {ml_res['Baseline_AUC']:.4f}  | Baseline F1: {ml_res['Baseline_Macro_F1']:.4f}")
        print(f"   - Augment  AUC: {ml_res['Augmented_AUC']:.4f}  | Augment  F1: {ml_res['Augmented_Macro_F1']:.4f}")
        
        # 用直观的符号表示提升还是下降
        auc_trend = "🟢 提升" if ml_res['AUC_Delta'] > 0 else "🔴 下降"
        f1_trend = "🟢 提升" if ml_res['F1_Delta'] > 0 else "🔴 下降"
        
        print(f"\n   >> AUC 差异: {ml_res['AUC_Delta']:+.4f} ({auc_trend})")
        print(f"   >> F1  差异: {ml_res['F1_Delta']:+.4f} ({f1_trend})")
        print("="*50 + "\n")
        
        return {**stats_res, **ml_res}
    


from scipy.spatial.distance import jensenshannon

class StabilityEvaluator:
    def __init__(self, df_real, list_of_df_fakes, cat_cols):
        """
        :param df_real: 真实数据集 
        :param list_of_df_fakes: 包含多次生成结果的DataFrame列表
        :param cat_cols: 类别特征列名
        """
        self.df_real = df_real
        self.list_of_df_fakes = list_of_df_fakes
        self.cat_cols = cat_cols
        self.num_runs = len(list_of_df_fakes)

    def _get_prob_dist(self, series, categories):
        """获取对齐的概率分布向量"""
        counts = series.value_counts(normalize=True).reindex(categories).fillna(0)
        return counts.values
        
    def evaluate_categorical_stability(self):
        report = {}
        for col in self.cat_cols:
            # 锁定该列的所有可能取值，保证分布向量维度一致
            categories = sorted(self.df_real[col].dropna().unique())
            real_dist = self._get_prob_dist(self.df_real[col], categories)

            run_tvd_list = []
            run_jsd_list = []
            run_dists = []

            # 计算每一次生成与真实数据的差异
            for df_fake in self.list_of_df_fakes:
                fake_dist = self._get_prob_dist(df_fake[col], categories)
                run_dists.append(fake_dist)

                # 计算TVD
                tvd = 0.5 * np.sum(np.abs(real_dist - fake_dist))
                run_tvd_list.append(tvd)

                # 计算JSD (平方后为距离)
                jsd = jensenshannon(real_dist, fake_dist) ** 2
                run_jsd_list.append(jsd)

            run_dists = np.array(run_dists) # [num_runs, num_categories]

            # 计算少数类的变异系数（CV), 找到占比最小的类别（少数类）
            minority_idx = np.argmin(real_dist)
            minority_props_across_runs = run_dists[:, minority_idx]
            minority_mean = np.mean(minority_props_across_runs)
            minority_std = np.std(minority_props_across_runs)
            cv_minority = minority_std / minority_mean if minority_mean > 0 else 0

            report[col]={
                'TVD_Mean': np.mean(run_tvd_list),
                'TVD_Std': np.std(run_tvd_list),
                'JSD_Mean': np.mean(run_jsd_list),
                'Minority_CV': cv_minority # 越小少数类生成数量越稳定
            }
        print(f"\n{'='*50}")
        print(f"🎲 生成分布稳定性报告 (基于 {self.num_runs} 次独立生成)")
        print(f"{'='*50}")
        print(f"{'Feature':<15} | {'Mean TVD':<10} | {'TVD Std (波动)':<15} | {'Minority CV':<12}")
        print("-" * 55)
        for col, metrics in report.items():
            print(f"{col:<15} | {metrics['TVD_Mean']:.4f}   | ± {metrics['TVD_Std']:.4f}      | {metrics['Minority_CV']:.4f}")
            
        return report
