# core/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class TabularDataProcessor:
    """
    表格数据双向处理器：MICE 插补 + 分布映射 + 数值结构约束恢复

    num_transform_method:
        'quantile' : QuantileTransformer → 正态分布（默认，配合标准 MLP 使用）
        'standard' : StandardScaler → 零均值单位方差
        'none'     : 不做分布变换，保留原始数值（配合 Periodic Embedding 使用）
    """

    def __init__(self, num_cols, cat_cols, num_transform_method='quantile', random_state=42):
        assert isinstance(num_cols, list) and isinstance(cat_cols, list)
        assert num_transform_method in ('quantile', 'standard', 'none'), (
            f"num_transform_method 必须是 'quantile'/'standard'/'none'，got '{num_transform_method}'"
        )

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_transform_method = num_transform_method
        self.random_state = random_state

        self.meta = {}
        self.precisions = {}
        self.num_stats = {}
        self.original_columns = None

        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=random_state,
            initial_strategy='median',
        )

    # =========================================================
    # 数值结构检测（私有）
    # =========================================================
    def _detect_quantize_step(self, unique_values, candidates=(0.5,), tol=1e-6):
        """
        检测最小公共步长,仅用于非整数列
        候选只保留 0.5：整数列由 is_int 分支处理
        """
        for q in candidates:
            remainders = np.abs(unique_values % q)
            if np.all(np.minimum(remainders, np.abs(remainders - q)) < tol):
                return q
        return None

    def _analyze_numeric_structure(self, values, is_int: bool, tolerance=1e-9):
        """
        返回数值列的 quantize_step：
          - 整数列（is_int=True）：直接返回 None，逆变换时统一用 round+astype(int) 处理
          - 非整数列：检测是否存在 0.5 步长（半整数列如 sysBP/diaBP）
          - 连续列：返回 None，仅做精度舍入
        """
        result = {'step': None, 'quantize_step': None}

        # 跳过整数列， 避免等差训练集导致的误判
        if is_int:
            return result
        
        unique = np.sort(np.unique(pd.Series(values).dropna().values))
        if len(unique) <= 1:
            return result
        
        result['quantize_step'] = self._detect_quantize_step(unique)
        return result

    @staticmethod
    def _infer_precision(series: pd.Series) -> int:
        """
        推断小数位数。用 95 分位数而非众数：
        众数会被大量整数/一位小数值拉低，导致少数高精度值被过度截断。
        例如 BMI = [18.4, 22.7, 25.13, ...] 众数=1，25.13 会被 round 成 25.1。
        95 分位数更保守，上限设为 4（超过 4 位通常是浮点噪声）。
        """
        sample = series.dropna().sample(min(1000, len(series)))
        decimals = [
            len(f"{v:.10f}".rstrip('0').split('.')[1]) if '.' in f"{v:.10f}".rstrip('0') else 0
            for v in sample
        ]
        return int(min(np.percentile(decimals, 95), 4))

    # =========================================================
    def fit(self, df: pd.DataFrame):
        self.original_columns = df.columns.tolist()
        df_temp = pd.DataFrame(index=df.index)

        # ---------- 类别特征 ----------
        for col in self.cat_cols:
            series = df[col]
            encoder = LabelEncoder()
            encoder.fit(series.dropna().astype(str))
            self.meta[col] = {
                'type': 'cat',
                'encoder': encoder,
                'classes': len(encoder.classes_),
            }
            df_temp[col] = series.apply(
                lambda x: encoder.transform([str(x)])[0] if pd.notna(x) else np.nan
            )

        # ---------- 数值特征 ----------
        for col in self.num_cols:
            valid_data = df[col].dropna()

            # Bug fix: 去掉不存在的 equal_nan 参数
            is_int = bool(np.allclose(valid_data, np.round(valid_data)))

            structure = self._analyze_numeric_structure(valid_data, is_int=is_int)
            quantize_step = structure['quantize_step']

            # 精度：优先从 quantize_step 推导，更稳定
            if quantize_step is not None:
                q_str = f"{quantize_step:.10f}".rstrip('0')
                prec = len(q_str.split('.')[1]) if '.' in q_str else 0
            else:
                prec = self._infer_precision(valid_data)
            self.precisions[col] = prec

            # 分布编码器
            if self.num_transform_method == 'quantile':
                encoder = QuantileTransformer(
                    n_quantiles=min(100, len(valid_data)),
                    output_distribution='normal',
                    random_state=self.random_state,
                )
            elif self.num_transform_method == 'standard':
                encoder = StandardScaler()
            else:
                encoder = None  # 'none'：直接透传

            # 为还原后的数据增加5%的余量
            value_range = valid_data.max() - valid_data.min()

            self.meta[col] = {
                'type': 'num',
                'is_int': is_int,
                'min': float(valid_data.min() - 0.05 * value_range),
                'max': float(valid_data.max() + 0.05 * value_range),
                'encoder': encoder,
                'quantize_step': quantize_step,
            }
            df_temp[col] = df[col]

        # ---------- MICE 插补 ----------
        ordered_cols = self.num_cols + self.cat_cols
        imputed_array = self.imputer.fit_transform(df_temp[ordered_cols])
        df_imputed = pd.DataFrame(imputed_array, columns=ordered_cols, index=df.index)

        # ---------- 分布映射拟合 ----------
        for col in self.num_cols:
            enc = self.meta[col]['encoder']
            if enc is not None:
                enc.fit(df_imputed[col].values.reshape(-1, 1))

        # ---------- 'none' 模式：记录统计量供 Periodic Embedding 使用 ----------
        if self.num_transform_method == 'none':
            for col in self.num_cols:
                col_data = df_imputed[col].values
                self.num_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std':  float(col_data.std()) + 1e-8,
                }

        return self

    # =========================================================
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ordered_cols = self.num_cols + self.cat_cols
        df_temp = pd.DataFrame(index=df.index)

        for col in self.cat_cols:
            enc = self.meta[col]['encoder']
            df_temp[col] = df[col].apply(
                lambda x: enc.transform([str(x)])[0] if pd.notna(x) else np.nan
            )
        for col in self.num_cols:
            df_temp[col] = df[col]

        df_imputed = pd.DataFrame(
            self.imputer.transform(df_temp[ordered_cols]),
            columns=ordered_cols,
            index=df.index,
        )

        df_out = pd.DataFrame(index=df.index)
        for col in self.num_cols:
            enc = self.meta[col]['encoder']
            if enc is not None:
                df_out[col] = enc.transform(df_imputed[col].values.reshape(-1, 1)).flatten()
            else:
                df_out[col] = df_imputed[col].values  # 'none'：直接透传

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

        # ---------- 数值恢复 ----------
        for col in self.num_cols:
            meta = self.meta[col]
            vals = df[col].values.reshape(-1, 1)

            inversed = (
                meta['encoder'].inverse_transform(vals).flatten()
                if meta['encoder'] is not None
                else vals.flatten()
            )

            inversed = np.clip(inversed, meta['min'], meta['max'])

            q = meta['quantize_step']
            if q is not None and q > 0:
                # 对齐到最近合法值：round(x/q)*q
                inversed = np.round(inversed / q) * q
                inversed = np.round(inversed, 6)  # 消除浮点误差

            prec = self.precisions.get(col, 2)
            inversed = np.round(inversed, prec)

            # 用 pd.array 做整数转换：能保留 NaN，不会像 numpy astype(int) 一样溢出
            if prec == 0 or meta['is_int']:
                df_out[col] = pd.array(inversed, dtype=pd.Int64Dtype())
            else:
                df_out[col] = inversed

        # ---------- 类别恢复 ----------
        for col in self.cat_cols:
            raw = df[col].values
            # Bug fix: NaN 会被 astype(int) 变成极大负数，先替换为 0
            raw = np.where(np.isnan(raw), 0.0, np.round(raw))
            vals = np.clip(raw.astype(int), 0, self.meta[col]['classes'] - 1)
            labels = self.meta[col]['encoder'].inverse_transform(vals)
            try:
                df_out[col] = pd.to_numeric(labels)
            except ValueError:
                df_out[col] = labels

        if self.original_columns is not None:
            return df_out[self.original_columns]
        return df_out

    # =========================================================
    def get_model_info(self):
        return {
            'num_numerical': len(self.num_cols),
            'num_classes':   [self.meta[col]['classes'] for col in self.cat_cols],
            'cat_cols':      self.cat_cols,
            'num_cols':      self.num_cols,
            'num_stats':     self.num_stats,  # 'none' 模式下才有值，其余为 {}
        }