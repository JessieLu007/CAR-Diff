# 文件位置: core/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class TabularDataProcessor:
    """支持 MICE 插补 + 分布映射 + 数值结构约束恢复 的表格数据双向处理器"""

    def __init__(self, num_cols, cat_cols, num_transform_method='quantile', random_state=42):
        assert isinstance(num_cols, list) and isinstance(cat_cols, list)

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_transform_method = num_transform_method
        self.random_state = random_state

        self.meta = {}
        self.precisions = {}
        self.original_columns = None

        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=random_state,
            initial_strategy='median'
        )

    # =========================================================
    # 统一数值结构检测
    # =========================================================
    def _analyze_numeric_structure(self, values, tolerance=1e-9):
        values = pd.Series(values).dropna().values
        unique = np.sort(np.unique(values))

        result = {
            'step': None,
            'valid_set': unique,
            'decimal_pattern': None
        }

        if len(unique) <= 1:
            return result

        # ---------- 1. 检测等间距 ----------
        diffs = np.diff(unique)
        if np.all(np.abs(diffs - diffs[0]) < tolerance):
            result['step'] = diffs[0]

        # ---------- 2. 检测小数模式 ----------
        frac = np.abs(unique - np.round(unique))
        frac = np.round(frac, 3)
        unique_frac = np.unique(frac)

        if len(unique_frac) <= 3:
            result['decimal_pattern'] = unique_frac.tolist()

        return result

    # =========================================================
    def fit(self, df: pd.DataFrame):
        self.original_columns = df.columns.tolist()
        df_temp = pd.DataFrame(index=df.index)

        # =========================
        # 1. 类别特征
        # =========================
        for col in self.cat_cols:
            series = df[col]
            encoder = LabelEncoder()
            encoder.fit(series.dropna().astype(str))

            self.meta[col] = {
                'type': 'cat',
                'encoder': encoder,
                'classes': len(encoder.classes_)
            }

            df_temp[col] = series.apply(
                lambda x: encoder.transform([str(x)])[0] if pd.notna(x) else np.nan
            )

        # =========================
        # 2. 数值特征（结构+精度）
        # =========================
        for col in self.num_cols:
            series = df[col]
            valid_data = series.dropna()

            is_int = np.allclose(valid_data, np.round(valid_data), equal_nan=True)

            # ---------- 结构分析 ----------
            structure = self._analyze_numeric_structure(valid_data)
            step = structure['step']
            valid_set = structure['valid_set']
            decimal_pattern = structure['decimal_pattern']

            # ---------- 精度 ----------
            if step is not None:
                step_str = f"{step:.10f}".rstrip('0').rstrip('.')
                prec = len(step_str.split('.')[1]) if '.' in step_str else 0
            else:
                sample = valid_data.sample(min(1000, len(valid_data)))
                decimals = sample.astype(str).apply(
                    lambda x: len(x.split('.')[1]) if '.' in x else 0
                )
                prec = int(decimals.mode()[0])

            self.precisions[col] = prec

            # ---------- 编码器 ----------
            if self.num_transform_method == 'quantile':
                encoder = QuantileTransformer(
                    n_quantiles=min(100, len(valid_data)),
                    output_distribution='normal',
                    random_state=self.random_state
                )
            elif self.num_transform_method == 'standard':
                encoder = StandardScaler()
            else:
                raise ValueError(f"Unsupported method: {self.num_transform_method}")

            self.meta[col] = {
                'type': 'num',
                'is_int': is_int,
                'min': valid_data.min(),
                'max': valid_data.max(),
                'encoder': encoder,
                'precision_step': step,
                'precision_valid_set': valid_set,
                'decimal_pattern': decimal_pattern
            }

            df_temp[col] = series

        # =========================
        # 3. MICE 插补
        # =========================
        ordered_cols = self.num_cols + self.cat_cols
        df_temp = df_temp[ordered_cols]

        imputed_array = self.imputer.fit_transform(df_temp)
        df_imputed = pd.DataFrame(imputed_array, columns=ordered_cols, index=df.index)

        # =========================
        # 4. 分布映射拟合
        # =========================
        for col in self.num_cols:
            self.meta[col]['encoder'].fit(
                df_imputed[col].values.reshape(-1, 1)
            )

        return self

    # =========================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_temp = pd.DataFrame(index=df.index)

        # ---------- 类别 ----------
        for col in self.cat_cols:
            encoder = self.meta[col]['encoder']
            df_temp[col] = df[col].apply(
                lambda x: encoder.transform([str(x)])[0] if pd.notna(x) else np.nan
            )

        # ---------- 数值 ----------
        for col in self.num_cols:
            df_temp[col] = df[col]

        # ---------- MICE ----------
        ordered_cols = self.num_cols + self.cat_cols
        df_imputed = pd.DataFrame(
            self.imputer.transform(df_temp[ordered_cols]),
            columns=ordered_cols,
            index=df.index
        )

        df_out = pd.DataFrame(index=df.index)

        # ---------- 数值映射 ----------
        for col in self.num_cols:
            df_out[col] = self.meta[col]['encoder'].transform(
                df_imputed[col].values.reshape(-1, 1)
            ).flatten()

        # ---------- 类别 ----------
        for col in self.cat_cols:
            df_out[col] = np.round(df_imputed[col]).astype(int)

        return df_out[ordered_cols].astype('float32')

    # =========================================================
    def inverse_transform(self, df_or_tensor):
        ordered_cols = self.num_cols + self.cat_cols

        if not isinstance(df_or_tensor, pd.DataFrame):
            df = pd.DataFrame(df_or_tensor, columns=ordered_cols)
        else:
            df = df_or_tensor.copy()

        df_out = pd.DataFrame(index=df.index)

        # =========================
        # 数值恢复（核心）
        # =========================
        for col in self.num_cols:
            meta = self.meta[col]
            vals = df[col].values.reshape(-1, 1)      
            inversed = meta['encoder'].inverse_transform(vals).flatten()

            # ---------- 边界 ----------
            inversed = np.clip(inversed, meta['min'], meta['max'])

            step = meta['precision_step']
            valid_set = meta['precision_valid_set']
            pattern = meta['decimal_pattern']

            # ---------- step ----------
            if step is not None and step > 0:
                inversed = np.round(inversed / step) * step
                inversed = np.round(inversed, 6)

            # ---------- valid set ----------
            elif valid_set is not None and len(valid_set) > 0:
                valid_arr = np.asarray(valid_set)
                inversed = np.array([
                    valid_arr[np.argmin(np.abs(valid_arr - v))]
                    for v in inversed
                ])

            # ---------- decimal pattern ----------
            if pattern is not None:
                base = np.floor(inversed)
                frac = inversed - base
                frac_new = np.array([
                    min(pattern, key=lambda p: abs(f - p))
                    for f in frac
                ])
                inversed = base + frac_new

            # ---------- 精度 ----------
            prec = self.precisions.get(col, 2)
            inversed = np.round(inversed, prec)

            if prec == 0 or meta['is_int']:
                inversed = inversed.astype(int)

            df_out[col] = inversed

        # =========================
        # 类别恢复
        # =========================
        for col in self.cat_cols:
            vals = np.clip(
                np.round(df[col].values).astype(int),
                0,
                self.meta[col]['classes'] - 1
            )

            labels = self.meta[col]['encoder'].inverse_transform(vals)

            try:
                df_out[col] = pd.to_numeric(labels)
            except ValueError:
                df_out[col] = labels

        # =========================
        if self.original_columns is not None:
            return df_out[self.original_columns]
        return df_out

    # =========================================================
    def get_model_info(self):
        return {
            'num_numerical': len(self.num_cols),
            'num_classes': [self.meta[col]['classes'] for col in self.cat_cols],
            'cat_cols': self.cat_cols,
            'num_cols': self.num_cols
        }