# evaluation/eva_crossval.py
"""多折交叉验证评估器：调用TabularEvaluator并聚合结果"""
import numpy as np
import pandas as pd
import os, pickle, json, datetime
from collections import defaultdict
from .core_evaluator import TabularEvaluator


class CrossValidationEvaluator:
    def __init__(
            self, 
            fold_results: list, 
            target_col: str, 
            cat_cols: list,
            ml_mode: str = 'tstr',
            training_scope: str = 'all',
            modeling_task: str = 'joint'
            ):
        """
        :param fold_results: 每折数据列表，每项含df_real_train / df_fake / df_real_test
        :param ml_mode： 'augmentation' or 'tstr' or infer_ml_mode_from_task()
        :param modeling_task: 模型任务标识
        """
        self.fold_results = fold_results
        self.target_col = target_col
        self.cat_cols = list(cat_cols)
        self.ml_mode = ml_mode          # 统一使用 ml_mode
        self.training_scope = training_scope
        self.modeling_task = modeling_task
        self.cat_compare_report = ""

    
    # ===================================================
    # 主入口
    # ===================================================
    def run(self, run_clf=False):
        """
        执行全部折的评估并聚合
        :param run_clf: 是否启用分类器评估与打印
        :return: (stats_agg, ml_agg)
        """
        assert self.fold_results, "fold_results 不能为空！ 请先传入数据或使用load_from_disk()读档"

        stats_agg = defaultdict(list)
        ml_agg = defaultdict(lambda: defaultdict(list))

        for fold_data in self.fold_results:
            eva = TabularEvaluator(
                df_real_train=fold_data['df_real_train'],
                df_fake=fold_data['df_fake'],
                df_real_test=fold_data['df_real_test'],
                target_col=self.target_col,
                cat_cols=self.cat_cols,
                ml_mode=self.ml_mode      # 直接传入固定模式
            )
            stats = {**eva.evaluate_statistical_fidelity(),
                     **eva.evaluate_privacy()}
            for k, v in stats.items():
                stats_agg[k].append(v)

            if run_clf:
                ml_res = eva.evaluate_ml_utility()
                for clf_name, metrics in ml_res.items():
                    for m_name, m_val in metrics.items():
                        ml_agg[clf_name][m_name].append(m_val)

        self._print_compact_report(stats_agg, ml_agg, print_clf = run_clf)
        self.cat_compare_report = self._build_cat_report()
        print(self.cat_compare_report)
        return stats_agg, ml_agg
    
    # ====================================================================
    # 类别占比对比（多折版本）
    # ====================================================================
    def _build_cat_report(self) -> str:
        """生成Fold by Fold 类别占比对比"""
        assert self.fold_results, "请先提供fold_results数据"

        # 是否将target列纳入对比
        cols = list(self.cat_cols)
        if self.target_col in self.fold_results[0]['df_real_train'].columns and self.target_col not in cols:
            cols.append(self.target_col)

        num_folds = len(self.fold_results)
        fake_cols = [f'Fake_F{i+1}(%)' for i in range(num_folds)]
        header_col = ['Real_Global(%)'] + fake_cols + ['Fake_Mean(%)']

        report = "\n" + "="*80 + "\n"
        report += "📊 分类属性占比 Fold-by-Fold 对比 (Real vs Fake)\n"
        report += "=" * 80 + "\n"

        report += "\t".join(header_col) + "\n"

        for col in cols:
            # 全局真实分布
            if self.training_scope == 'all' and self.modeling_task == 'joint':
                all_real = pd.concat([f['df_real_train'][col] for f in self.fold_results])
            else:
                all_real = pd.concat([
                    f['df_real_train'].loc[f['df_real_train'][self.target_col] == 1, col]
                    for f in self.fold_results
                ])
            real_global_prop = all_real.value_counts(normalize=True) * 100

            comp_df = pd.DataFrame({'Real_Global(%)': real_global_prop})

            for i, fold_data in enumerate(self.fold_results):
                comp_df[f'Fake_F{i+1}(%)'] = fold_data['df_fake'][col].value_counts(normalize=True) * 100

            comp_df = comp_df.fillna(0.0)
            comp_df['Fake_Mean(%)'] = comp_df[fake_cols].mean(axis=1)
            comp_df = comp_df.sort_values('Real_Global(%)', ascending=False).round(2)

            report += comp_df.to_string(header=False) + "\n"

        return report

    # =================================================================
    # 内部打印
    # =================================================================
    def _print_compact_report(self, stats_agg: dict, ml_agg: dict, print_clf: bool = False):
        n = len(self.fold_results)
        print(f"\n[{self.ml_mode.upper()} MODE] {n}-Fold Evaluation Results")
        print("-" * 80)

        print("[1] Data Fidelity & Privacy:")
        for k, v in stats_agg.items():
            print(f"    - {k:<20}: {np.mean(v):.4f} ± {np.std(v):.4f}")
        
        if not print_clf:
            print("=" * 80 + "\n")
            return
        
        print("-" * 80)
        print("[2] ML Classification Utility")
        header = f"| {'Model':<14} | {'Metric':<10} | {'Baseline':<18} | {'Augmented':<18} | {'Delta':<18} |"
        separator = f"|{'-'*16}|{'-'*12}|{'-'*20}|{'-'*20}|{'-'*20}|"
        print(header)
        print(separator)

        metrics_list = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall']
        for clf_name, metrics in ml_agg.items():
            for i, m in enumerate(metrics_list):
                b_vals = metrics.get(f'Base_{m}', [])
                a_vals = metrics.get(f'Aug_{m}', [])
                if not b_vals:
                    continue
                d_vals = [a - b for a, b in zip(a_vals, b_vals)]
                b_str = f"{np.mean(b_vals):.4f ± {np.std(b_vals):.4f}}"
                a_str = f"{np.mean(a_vals):.4f ± {np.std(a_vals):.4f}}"
                d_str = f"{np.mean(d_vals):.4f} ± {np.std(d_vals):.4f}"
                model_col = clf_name if i == 0 else ""
                print(f"| {model_col:<14} | {m:<10} | {b_str:<18} | {a_str:<18} | {d_str:<18} |")
        print("=" * 80 + "\n")


    # ===============================================================
    # 从磁盘加载历史结果
    # ===============================================================
    @classmethod
    def load_from_disk(cls, exp_dir: str, target_col: str, cat_cols: list, ml_mode: str = 'augmentation'):
        """从pkl或CSV目录中加载历史数据并实例化"""
        print(f"📂 正在从 {exp_dir} 读取历史数据...")
        pkl_path = os.path.join(exp_dir, "fold_results.pkl")

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                fold_results = pickle.load(f)
            print(f"✅ 成功读取 {len(fold_results)} 折 Pkl 数据！")
        else:
            data_dir = os.path.join(exp_dir, "fold_data")
            csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('_real_train.csv')])
            n_folds = len(csv_files)
            fold_results = []
            for i in range(n_folds):
                fold_results.append({
                    'fold': i + 1,
                    'df_real_train': pd.read_csv(os.path.join(data_dir, f"fold_{i}_real_train.csv")),
                    'df_real_test': pd.read_csv(os.path.join(data_dir, f"fold_{i}_real_test.csv")),
                    'df_fake': pd.read_csv(os.path.join(data_dir, f"fold_{i}_fake.csv"))
                })
            print(f"✅ 自动检测并拼装 {len(fold_results)} 折 CSV 数据")
        return cls(fold_results, target_col, cat_cols, ml_mode=ml_mode)
    

    # ===================================================================
    # 保存结果
    # ===================================================================
    def save_results(self, stats_agg: dict, ml_agg: dict, save_dir: str = "cv_results", exp_name: str = None) -> str:
        """
        将五折交叉验证的所有产物（数据、评估指标、可视化表格）结构化落盘
        """
        import os, json, datetime

        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        exp_name = exp_name or f"exp_{timestamp}"
        exp_dir = os.path.join(save_dir, f"exp_{exp_name}")
        os.makedirs(exp_dir, exist_ok=True)
        
        # --- 1. 保存每一折的数据集 (CSV) ---
        data_dir = os.path.join(exp_dir, "fold_data")
        os.makedirs(data_dir, exist_ok=True)
        
        for i, fold_data in enumerate(self.fold_results):
            fold_data['df_real_train'].to_csv(os.path.join(data_dir, f"fold_{i}_real_train.csv"), index=False)
            fold_data['df_real_test'].to_csv(os.path.join(data_dir, f"fold_{i}_real_test.csv"), index=False)
            fold_data['df_fake'].to_csv(os.path.join(data_dir, f"fold_{i}_fake.csv"), index=False)

        # --- 2. 保存详细的机器可读评估指标 (JSON) ---
        # 将 numpy 数组转为 list 以便 json 序列化
        json_metrics = {
            "stats_and_privacy": {k: [float(x) for x in v] for k, v in stats_agg.items()},
            "ml_utility": {
                clf: {metric: [float(x) for x in vals] for metric, vals in clf_metrics.items()}
                for clf, clf_metrics in ml_agg.items()
            },
        }
        with open(os.path.join(exp_dir, "metrics_detailed.json"), "w") as f:
            json.dump(json_metrics, f, indent=4)

        # --- 3. 汇总表 CSV（均值±标准差）---
        metrics_list = ['ROC-AUC', 'PR-AUC', 'F1', 'Recall']
        records = []
        
        for clf_name, clf_metrics in ml_agg.items():
            for m in metrics_list:
                b_vals = clf_metrics.get(f'Base_{m}', [])
                a_vals = clf_metrics.get(f'Aug_{m}', [])
                if not b_vals:
                    continue
                d_vals = [a - b for a, b in zip(a_vals, b_vals)]
                records.append({
                    'Model': clf_name, 'Metric': m,
                    'Base_Mean':  np.mean(b_vals), 'Base_Std':  np.std(b_vals),
                    'Aug_Mean':   np.mean(a_vals),  'Aug_Std':   np.std(a_vals),
                    'Delta_Mean': np.mean(d_vals),  'Delta_Std': np.std(d_vals),
                })
        pd.DataFrame(records).to_csv(os.path.join(exp_dir, "metrics_summary.csv"), index=False)

        # --- 4. 类别占比报告 ---
        if self.cat_compare_report:
            with open(os.path.join(exp_dir, "categorical_comparison.txt"), "w", encoding='utf-8') as f:
                f.write(self.cat_compare_report)
 
        print(f"💾 所有实验数据与评估报告已保存至: {exp_dir}")
        return exp_dir