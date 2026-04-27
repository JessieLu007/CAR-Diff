# evaluation/evaluator.py
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import cdist, jensenshannon
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from core.data_processor import TabularDataProcessor


class TabularEvaluator:
    def __init__(self, df_real_train, df_fake, df_real_test, target_col, cat_cols):
        self.target_col = target_col
        self.cat_cols = cat_cols
        self.num_cols = [c for c in df_real_train.columns if c not in cat_cols and c != target_col]
        self.features = self.num_cols + self.cat_cols

        # 剥离特征与标签
        self.X_real_train_raw = df_real_train[self.features].copy()
        self.y_real_train = df_real_train[target_col]#.values.ravel()

        self.has_fake_label = (target_col in df_fake.columns)
        self.X_fake_raw = df_fake[self.features].copy()
        if self.has_fake_label:
            self.y_fake = df_fake[target_col]#.values.ravel()
        else:
            self.y_fake = pd.Series(1, index=self.X_fake_raw.index)

        self.X_real_test_raw = df_real_test[self.features].copy()
        self.y_real_test = df_real_test[target_col]#.values.ravel()

        # 预处理操作
        self.ml_processor = TabularDataProcessor(
            num_cols=self.num_cols,
            cat_cols=self.cat_cols,
            num_transform_method='quantile'
        )

        self.ml_processor.fit(self.X_real_train_raw)

        # 转换特征空间
        self.X_real_train_proc = self.ml_processor.transform(self.X_real_train_raw)
        self.X_fake_proc = self.ml_processor.transform(self.X_fake_raw)
        self.X_real_test_proc = self.ml_processor.transform(self.X_real_test_raw)

        # 初始化分类器字典
        self.classifiers = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
            'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
        }

    def evaluate_statistical_fidelity(self):
        """评估分布拟合度（越低越好）"""
        results = {}
        ks_distances = []
        for c in self.num_cols:
            real_col = self.X_real_train_raw[c].dropna()
            fake_col = self.X_fake_raw[c].dropna()

            if len(real_col) > 0 and len(fake_col) > 0:
                stat, pval = ks_2samp(real_col,fake_col)
                ks_distances.append(stat if not np.isnan(stat) else 1.0)
            else:
                ks_distances.append(1.0)
        # A. 边缘分布拟合（Marginal Distribution)
        # 使用KS检验（Kolmogorov-Smirnov）计算连续数值特征的分布距离
        #ks_distances = [ks_2samp(self.X_real_train_raw[c], self.X_fake_raw[c])[0] for c in self.num_cols]
        results['Avg_KS_Distance'] = np.mean(ks_distances) if ks_distances else 0.0

        # B. 联合分布关联度（Joint Distribution / Correlation)
        # 解决属性间关联度丢失问题，计算真实数据与生成数据的特征相关性矩阵，并求它们之间的均方误差
        if len(self.num_cols) > 1:
            real_corr = self.X_real_train_raw[self.num_cols].corr(method='pearson').fillna(0).values
            fake_corr = self.X_fake_raw[self.num_cols].corr(method='pearson').fillna(0).values
            results['Correlation_MSE'] = np.mean((real_corr - fake_corr) ** 2)
        else:
            results['Correlation_MSE'] = 0.0

        return results

    def evaluate_privacy(self, sample_size=1000):
        # 评估隐私保护程度（DCR），为避免计算过慢，对数据进行下采样并标准化处理
        # 提取纯特征，并对齐顺序
        valid_real_raw = self.X_real_train_raw.dropna()
        valid_fake_raw = self.X_fake_raw.dropna()

        n_real, n_fake = min(len(valid_real_raw), sample_size), min(len(valid_fake_raw), sample_size)
        X_real_sample = valid_real_raw.sample(n_real, random_state=42)
        X_fake_sample = valid_fake_raw.sample(n_fake, random_state=42)

        # 分离数值和类别特征
        real_num = X_real_sample[self.num_cols].values
        fake_num = X_fake_sample[self.num_cols].values
        real_cat = X_real_sample[self.cat_cols].values
        fake_cat = X_fake_sample[self.cat_cols].values

        # 1. 计算数值特征的距离
        ranges = np.ptp(valid_real_raw[self.num_cols].values, axis=0)
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
        ml_results = {}

        # 准备增强模型（Augmented: 真实训练集 + 生成集）
        if mode == 'augmentation':
            X_aug_train = pd.concat([self.X_real_train_proc, self.X_fake_proc], axis=0)
            y_aug_train = pd.concat([self.y_real_train, self.y_fake], axis=0)
            # shuffle
            shuffle_idx = np.random.permutation(len(X_aug_train))
            X_aug_train = X_aug_train.iloc[shuffle_idx].reset_index(drop=True)
            y_aug_train = y_aug_train.iloc[shuffle_idx]
        elif mode == 'tstr': # TSTR: 只用合成数据训练，测试真实数据（考验生成数据的独立可用性）
            X_aug_train = self.X_fake_proc
            y_aug_train = self.y_fake
        else:
            raise ValueError(f"不支持的评估模式：{mode}")
        
        for clf_name, clf_base in self.classifiers.items():
            import sklearn
            clf_aug = sklearn.base.clone(clf_base)

            # ------ 训练baseline -------
            clf_base.fit(self.X_real_train_proc, self.y_real_train)
            preds_base = clf_base.predict(self.X_real_test_proc)
            probs_base = clf_base.predict_proba(self.X_real_test_proc)[:, 1] if hasattr(clf_base, "predict_proba") else preds_base
            #base_pr_auc = average_precision_score(self.y_real_test, probs_base)
            #base_f1 = f1_score(self.y_real_test, preds_base, average='macro')

            # ------- 训练 Augmented -------
            clf_aug.fit(X_aug_train, y_aug_train)
            preds_aug = clf_aug.predict(self.X_real_test_proc)
            probs_aug = clf_aug.predict_proba(self.X_real_test_proc)[:, 1] if hasattr(clf_aug, "precict_proba") else preds_aug
            #aug_pr_auc = average_precision_score(self.y_real_test, probs_aug)
            #aug_f1 = f1_score(self.y_real_test, preds_aug, average='macro')

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
        print(f"   - Baseline AUC: {ml_res['Base_AUC']:.4f}  | Base F1: {ml_res['Base_F1']:.4f}")
        print(f"   - Augment  AUC: {ml_res['Aug_AUC']:.4f}  | Aug F1: {ml_res['Aug_F1']:.4f}")
        
        # 用直观的符号表示提升还是下降
        auc_trend = "🟢 提升" if ml_res['AUC_Delta'] > 0 else "🔴 下降"
        f1_trend = "🟢 提升" if ml_res['F1_Delta'] > 0 else "🔴 下降"
        
        print(f"\n   >> AUC 差异: {ml_res['AUC_Delta']:+.4f} ({auc_trend})")
        print(f"   >> F1  差异: {ml_res['F1_Delta']:+.4f} ({f1_trend})")
        print("="*50 + "\n")
        
        return {**stats_res, **ml_res}
    
class CrossValidationEvaluator:
    def __init__(self, fold_results, target_col, cat_cols, mode='augmentation',
                 training_scope: str = 'all', modeling_task: str = 'joint'):
        self.fold_results = fold_results
        self.target_col = target_col
        self.cat_cols = cat_cols
        self.mode = mode
        self.training_scope = training_scope
        self.modeling_task = modeling_task
        self.cat_compare_report = ""

    def run(self):
        assert self.fold_results, "fold_results 不能为空! 请先传入数据或使用load_from_disk()读档"
        stats_agg = defaultdict(list)
        ml_agg = defaultdict(lambda: defaultdict(list))

        for fold_data in self.fold_results:
            evaluator = TabularEvaluator(
                df_real_train=fold_data['df_real_train'],
                df_fake=fold_data['df_fake'],
                df_real_test=fold_data['df_real_test'],
                target_col = self.target_col,
                cat_cols=self.cat_cols
            )

            # 1. 统计与隐私
            stats = {**evaluator.evaluate_statistical_fidelity(), **evaluator.evaluate_privacy()}
            for k, v in stats.items():
                stats_agg[k].append(v)

            # 2. 分类器
            test_clf = False
            if test_clf:
                ml_res = evaluator.evaluate_ml_utility(mode=self.mode)
                for clf_name, metrics in ml_res.items():
                    for m_name, m_val in metrics.items():
                        ml_agg[clf_name][m_name].append(m_val)
        
        self._print_compact_report(stats_agg, ml_agg, self.mode)
        self.cat_compare_report = self.evaluate_categorical_proportions()
        return stats_agg, ml_agg
    
    def evaluate_categorical_proportions(self):
        """Fold-by-Fold 精细化分类属性占比对比"""
        assert self.fold_results, "请先提供fold_results数据!"
        report_str = "\n" + "="*80 + "\n"
        report_str += "📊 分类属性占比 Fold-by-Fold 对比 (Real vs Fake)\n"
        report_str += "="*80 + "\n"

        # 预先计算fake列名列表
        num_folds = len(self.fold_results)
        fake_cols = [f'Fake_F{i+1}(%)' for i in range(num_folds)]
        header_cols = ['Real_Global(%)'] + fake_cols + ['Fake_Mean(%)']

        # 如果数据中有'label'列，将其加入cat_cols中并比较
        if 'label' in self.fold_results[0]["df_real_train"].columns:
            self.cat_cols.append('label')

        # 输出一次表头
        report_str += "\t".join(header_cols) + "\n"

        for col in self.cat_cols:
            if self.training_scope == 'all' and self.modeling_task == 'joint':
                all_real = pd.concat([f['df_real_train'][col] for f in self.fold_results])
            else:
                all_real_series = []
                for f in self.fold_results:
                    df_tr = f['df_real_train']
                    mask = (df_tr[self.target_col] == 1)
                    all_real_series.append(df_tr.loc[mask, col])
                all_real = pd.concat(all_real_series)
            real_global_prop = all_real.value_counts(normalize=True) * 100

            # 2. 初始化对比DataFrame
            comp_df = pd.DataFrame({'Real_Global(%)': real_global_prop})

            # 3. 逐折计算生成数据的分布
            for i, fold_data in enumerate(self.fold_results):
                fake_col = fold_data['df_fake'][col]
                prop_fake = fake_col.value_counts(normalize=True) * 100
                comp_df[f'Fake_F{i+1}(%)'] = prop_fake
                
            # 4. 填充NaN为0（对应某一折某个类别完全没生成的情况)
            comp_df = comp_df.fillna(0.0)

            # 5. 计算Fake的折均值
            #fake_cols = [f'Fake_F{i+1}(%)' for i in range(len(self.fold_results))]
            comp_df['Fake_Mean(%)'] = comp_df[fake_cols].mean(axis=1)

            # 6. 按真实全局分布降序，保留 2 位小数
            comp_df = comp_df.sort_values(by='Real_Global(%)', ascending=False).round(2)
            
            #report_str += f"\n--- Feature: {col} ---\n"
            report_str += comp_df.to_string(header=False) + "\n"

        if 'label' in self.cat_cols:
            self.cat_cols.remove('label')

        print(report_str)
        return report_str
    

    @classmethod
    def load_from_disk(cls, exp_dir: str, target_col: str, cat_cols: list, mode='augmentation'):
        """
        从硬盘中的pkl或csv文件夹加载历史数据，并实例化Evaluator
        """
        print(f"📂 正在从 {exp_dir} 读取历史数据...")
        pkl_path = os.path.join(exp_dir, "fold_results.pkl")
        
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                fold_results = pickle.load(f)
            print(f"✅ 成功读取 {len(fold_results)} 折 Pkl 数据！")
        else:
            fold_results = []
            data_dir = os.path.join(exp_dir, "fold_data")
            for i in range(5):
                fold_results.append({
                    'fold': i+1,
                    'df_real_train': pd.read_csv(os.path.join(data_dir, f"fold_{i}_real_train.csv")),
                    'df_real_test': pd.read_csv(os.path.join(data_dir, f"fold_{i}_real_test.csv")),
                    'df_fake': pd.read_csv(os.path.join(data_dir, f"fold_{i}_fake.csv"))
                })
            print(f"✅ 成功拼装 5 折 CSV 数据！")

        # 将读出的数据喂给类本身并返回实例化对象
        return cls(fold_results, target_col, cat_cols, mode)
    
    def _print_compact_report(self, stats_agg, ml_agg, mode):
        print(f"\n[{mode.upper()} MODE] 5-Fold Evaluation Results")
        print("-" * 80)
        
        print("[1] Data Fidelity & Privacy:")
        for k, v in stats_agg.items():
            print(f"    - {k:<20}: {np.mean(v):.4f} ± {np.std(v):.4f}")

        print_clf = False
        if print_clf:            
            print("-" * 80)
            print("[2] ML Classification Utility:\n")
            
            # 宽版精美 Markdown 表头
            header = f"| {'Model':<14} | {'Metric':<10} | {'Baseline':<18} | {'Augmented':<18} | {'Delta':<18} |"
            separator = f"|{'-'*16}|{'-'*12}|{'-'*20}|{'-'*20}|{'-'*20}|"
            print(header)
            print(separator)
        
            metrics_list = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall']
            
            for clf_name, metrics in ml_agg.items():
                for i, m in enumerate(metrics_list):
                    b_vals, a_vals = metrics[f'Base_{m}'], metrics[f'Aug_{m}']
                    d_vals = [a - b for a, b in zip(a_vals, b_vals)]
                    
                    b_str = f"{np.mean(b_vals):.4f} ± {np.std(b_vals):.4f}"
                    a_str = f"{np.mean(a_vals):.4f} ± {np.std(a_vals):.4f}"
                    d_str = f"{np.mean(d_vals):+.4f} ± {np.std(d_vals):.4f}"
                    
                    # 只有第一行打印模型名称，其余留空，视觉更紧凑
                    model_col = clf_name if i == 0 else ""
                    print(f"| {model_col:<14} | {m:<10} | {b_str:<18} | {a_str:<18} | {d_str:<18} |")
                    
            print("=" * 80 + "\n")

    def save_results(self, stats_agg, ml_agg, save_dir="cv_results"):
        """
        将五折交叉验证的所有产物（数据、评估指标、可视化表格）结构化落盘
        """
        import os
        import json
        import datetime
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        exp_dir = os.path.join(save_dir, f"exp_{timestamp}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # ==========================================
        # 1. 保存每一折的数据集 (CSV)
        # ==========================================
        data_dir = os.path.join(exp_dir, "fold_data")
        os.makedirs(data_dir, exist_ok=True)
        
        for i, fold_data in enumerate(self.fold_results):
            fold_data['df_real_train'].to_csv(os.path.join(data_dir, f"fold_{i}_real_train.csv"), index=False)
            fold_data['df_real_test'].to_csv(os.path.join(data_dir, f"fold_{i}_real_test.csv"), index=False)
            fold_data['df_fake'].to_csv(os.path.join(data_dir, f"fold_{i}_fake.csv"), index=False)

        # ==========================================
        # 2. 保存详细的机器可读评估指标 (JSON)
        # ==========================================
        # 将 numpy 数组转为 list 以便 json 序列化
        json_metrics = {
            "stats_and_privacy": {k: [float(x) for x in v] for k, v in stats_agg.items()},
            "ml_utility": {
                clf: {metric: [float(x) for x in vals] for metric, vals in clf_metrics.items()}
                for clf, clf_metrics in ml_agg.items()
            }
        }
        with open(os.path.join(exp_dir, "metrics_detailed.json"), "w") as f:
            json.dump(json_metrics, f, indent=4)

        # ==========================================
        # 3. 保存方便人类查看/贴论文的汇总表 (CSV)
        # ==========================================
        records = []
        metrics_list = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall']
        
        for clf_name, metrics in ml_agg.items():
            for m in metrics_list:
                b_vals, a_vals = metrics[f'Base_{m}'], metrics[f'Aug_{m}']
                d_vals = [a - b for a, b in zip(a_vals, b_vals)]
                
                records.append({
                    'Model': clf_name,
                    'Metric': m,
                    'Base_Mean': np.mean(b_vals), 'Base_Std': np.std(b_vals),
                    'Aug_Mean': np.mean(a_vals),  'Aug_Std': np.std(a_vals),
                    'Delta_Mean': np.mean(d_vals), 'Delta_Std': np.std(d_vals)
                })
                
        df_summary = pd.DataFrame(records)
        df_summary.to_csv(os.path.join(exp_dir, "metrics_summary_table.csv"), index=False)

        if hasattr(self, 'cat_compare_report'):
            with open(os.path.join(exp_dir, "categorical_comparison.txt"), "w", encoding='utf-8') as f:
                f.write(self.cat_compare_report)
        
        print(f"💾 所有五折实验数据与评估报告已安全保存至: {exp_dir}")
        return exp_dir
    
    def _print_compact_report_old(self, stats_agg, ml_agg, mode):
        print(f"\n[{mode.upper()} MODE] 5-Fold Evaluation Results")
        print("-" * 100)
        
        # 1. 紧凑打印统计与隐私指标
        print("[1] Data Fidelity & Privacy:")
        for k, v in stats_agg.items():
            print(f"    - {k:<20}: {np.mean(v):.4f} (±{np.std(v):.4f})")
            
        print("-" * 100)
        
        # 2. 使用 Pandas 组合高度紧凑的对比表格
        print("[2] ML Classification Utility:")

        header = f"| {'Model':<14} | {'Metric':<6} | {'Baseline':<18} | {'Augmented':<18} | {'Delta':<18} |"
        separator = f"{'-'*16} | {'-'*8} | {'-'*20} | {'-'*20} | {'-'*20} |"
        print(header)
        print(separator)

        for clf_name, metrics in ml_agg.items():
            b_auc, a_auc = metrics['Base_AUC'], metrics['Aug_AUC']
            d_auc = [a - b for a, b in zip(a_auc, b_auc)] # paired difference

            b_auc_str = f"{np.mean(b_auc):.4f} ± {np.std(b_auc):.4f}"
            a_auc_str = f"{np.mean(a_auc):.4f} ± {np.std(a_auc):.4f}"
            d_auc_str = f"{np.mean(d_auc):.4f} ± {np.std(d_auc):.4f}"

            print(f"| {clf_name:<14} | {'AUC':<6} | {b_auc_str:<18} | {a_auc_str:<18} | {d_auc_str:<18}")

            b_f1, a_f1 = metrics['Base_F1'], metrics['Aug_F1']
            d_f1 = [a - b for a, b in zip(a_f1, b_f1)]

            b_f1_str = f"{np.mean(b_f1):.4f} ± {np.std(b_f1):.4f}"
            a_f1_str = f"{np.mean(a_f1):.4f} ± {np.std(a_f1):.4f}"
            d_f1_str = f"{np.mean(d_f1):.4f} ± {np.std(d_f1):.4f}"

            print(f"| {'':<14} | {'F1':<6} | {b_f1_str:<18} | {a_f1_str:<18} | {d_f1_str:<18} |")

        print("="*100 + "\n")


from scipy.spatial.distance import jensenshannon

class StabilityEvaluator:
    def __init__(self, df_real, list_of_df_fakes, cat_cols):
        """
        :param df_real: 真实数据集 
        :param list_of_df_fakes: 包含多次生成结果的DataFrame列表
        :param cat_cols: 类别特征列名
        """
        self.df_real = df_real
        if isinstance(list_of_df_fakes, pd.DataFrame):
            list_of_df_fakes = [list_of_df_fakes]
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
