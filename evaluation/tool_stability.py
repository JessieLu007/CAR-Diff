# evaluation/eva_stability.py
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

class StabilityEvaluator:
    def __init__(self, df_real:pd.DataFrame, list_of_df_fakes, cat_cols: list):
        """
        :param df_real: 真实数据集
        :param list_of_df_fakes: 多次生成结果的 DataFrame 列表（或单个 DataFrame）
        :param cat_cols: 类别特征列名
        """
        self.df_real = df_real
        self.list_of_df_fakes = ([list_of_df_fakes] if isinstance(list_of_df_fakes, pd.DataFrame) 
                                 else list_of_df_fakes)
        self.cat_cols = cat_cols
        self.num_runs = len(self.list_of_df_fakes)

    def _prob_dist(self, series: pd.Series, categories: list) -> np.ndarray: 
        return series.value_counts(normalize=True).reindex(categories).fillna(0).values

    def evaluate_categorical_stability(self) -> dict:
        """
        计算每个类别特征的 TVD、JSD 及少数类变异系数（CV）。
        TVD / JSD 越低越好；Minority_CV 越小代表少数类生成数量越稳定。
        """
        report = {}
        for col in self.cat_cols:
            categories = sorted(self.df_real[col].dropna().unique())
            real_dist = self._prob_dist(self.df_real[col], categories)

            tvd_list, jsd_list, run_dists = [], [], []
            for df_fake in self.list_of_df_fakes:
                fake_dist = self._prob_dist(df_fake[col], categories)
                run_dists.append(fake_dist)
                tvd_list.append(0.5 * np.sum(np.abs(real_dist - fake_dist)))
                jsd_list.append(jensenshannon(real_dist, fake_dist) ** 2)

            run_dists       = np.array(run_dists)
            minority_idx    = np.argmin(real_dist)
            minority_props  = run_dists[:, minority_idx]
            minority_mean   = float(np.mean(minority_props)) 
            cv_minority     = float(np.std(minority_props) / minority_mean) if minority_mean > 0 else 0.0

            report[col] = {
                'TVD_Mean':     float(np.mean(tvd_list)),
                'TVD_Std':      float(np.std(tvd_list)),
                'JSD_Mean':     float(np.mean(jsd_list)),
                'Minority_CV':  cv_minority
            }
        self._print_report(report)
        return report

    def _print_report(self, report: dict):
        print(f"\n{'='*50}")
        print(f"生成分布稳定性报告 (基于 {self.num_runs} 次独立生成)")
        print(f"{'Feature':<15} | {'Mean TVD':<10} | {'TVD Std':<15} | {'Minority CV':<12}")
        print("-" * 55)
        for col, m in report.items():
            print(f"{col:<15} | {m['TVD_Mean']:.4f}   | ± {m['TVD_Std']:.4f}      | {m['Minority_CV']:.4f}")