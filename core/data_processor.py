# core/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class TabularDataProcessor:
    """
    表格数据双向处理器：MICE 插补 + 分布映射 + 数值结构约束恢复
    num_transform_method: 'quantile',  'standard', 'none'
    cat_encoding_method: 'label', 'ordinal', 'none'
    """

    def __init__(self, num_cols, cat_cols, num_transform_method='quantile', cat_encoding_method='label', random_state=42):
        assert isinstance(num_cols, list) and isinstance(cat_cols, list)
        assert num_transform_method in ('quantile', 'standard', 'minmax', 'none'), (
            f"num_transform_method 必须是 'quantile'/'standard'/'minmax'/'none'，got '{num_transform_method}'"
        )
        assert cat_encoding_method in ('label', 'ordinal', 'none'), (
            f"cat_encoding_method 必须是 'label'/'ordinal'/'none'，got '{cat_encoding_method}'"
        )

        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.num_transform_method = num_transform_method
        self.cat_encoding_method = cat_encoding_method
        self.random_state = random_state

        self.meta = {}
        self.precisions = {}
        self.num_stats = {}

        self.imputer = IterativeImputer(
            max_iter=10,
            random_state=random_state,
            initial_strategy='median',
        )

    # =========================================================
    # 类别编码辅助方法（私有）
    # =========================================================
    def _cat_encode(self, enc, x_str: str):
        """将单个字符串值编码为整数，屏蔽 LabelEncoder / OrdinalEncoder 的 API 差异。"""
        if isinstance(enc, OrdinalEncoder):
            # OrdinalEncoder.transform 接受二维输入，unknown → -1 → 后续转 NaN
            return enc.transform([[x_str]])[0, 0]
        else:
            # LabelEncoder.transform 接受一维输入
            return enc.transform([x_str])[0]

    def _cat_inverse_transform(self, col: str, vals_1d: np.ndarray) -> np.ndarray:
        """将整数数组反编码为原始标签，屏蔽 LabelEncoder / OrdinalEncoder 的维度差异。"""
        enc = self.meta[col]['encoder']
        if isinstance(enc, OrdinalEncoder):
            # OrdinalEncoder.inverse_transform 需要二维输入，输出也是二维
            return enc.inverse_transform(vals_1d.reshape(-1, 1)).flatten()
        else:
            # LabelEncoder.inverse_transform 接受一维输入
            return enc.inverse_transform(vals_1d)


    def _encode_cat_series_for_mice(self, col: str, series: pd.Series) -> pd.Series:
        """
        将类别列编码为浮点整数供 MICE 使用。
        关键：OrdinalEncoder 的 unknown_value=-1 在此统一转为 NaN，
        确保 MICE 将其视为缺失值而非合法编码。
        """
        enc = self.meta[col]['encoder']
        encoded = series.apply(
            lambda x: self._cat_encode(enc, str(x)) if pd.notna(x) else np.nan
        )
        if isinstance(enc, OrdinalEncoder):
            # -1 是 unknown_value，对 MICE 而言应当作缺失值
            encoded = encoded.replace(-1.0, np.nan)
        return encoded

    # =========================================================
    # 数值结构检测（私有）
    # =========================================================
    def _detect_quantize_step(self, unique_values, candidates=(0.5,), tol=1e-6):
        for q in candidates:
            remainders = np.abs(unique_values % q)
            if np.all(np.minimum(remainders, np.abs(remainders - q)) < tol):
                return q
        return None

    def _analyze_numeric_structure(self, values, is_int: bool, tolerance=1e-9):
        result = {'step': None, 'quantize_step': None}
        if is_int:
            return result
        unique = np.sort(np.unique(pd.Series(values).dropna().values))
        if len(unique) <= 1:
            return result
        result['quantize_step'] = self._detect_quantize_step(unique)
        return result

    #@staticmethod
    def _infer_precision(self, series: pd.Series) -> int:
        sample = series.dropna().sample(min(1000, len(series)), random_state=self.random_state)
        decimals = [
            len(f"{v:.10f}".rstrip('0').split('.')[1]) if '.' in f"{v:.10f}".rstrip('0') else 0
            for v in sample
        ]
        return int(min(np.percentile(decimals, 95), 4))

    # =========================================================
    def fit(self, df: pd.DataFrame):
        df_temp = pd.DataFrame(index=df.index)

        # ---------- 类别特征 ----------
        for col in self.cat_cols:
            series_raw = df[col]
            # Fix: dropna() 在 astype(str) 之前，避免 'nan' 字符串被编入 encoder
            valid_str = series_raw.dropna().astype(str)

            if self.cat_encoding_method == 'label':
                # LabelEncoder不支持未知类别，仅当确信测试集无新类别时使用
                encoder = LabelEncoder()
                encoder.fit(valid_str)
                n_classes = len(encoder.classes_)

            else: # 'ordinal' 或 'none'都需要处理未知类别
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
                encoder.fit(valid_str.values.reshape(-1, 1))
                n_classes = len(encoder.categories_[0])

            self.meta[col] = {
                'type': 'cat',
                'encoder': encoder,
                'classes': n_classes,
            }

            # Fix: 传入原始 series（含真正的 NaN），由辅助方法内部处理
            df_temp[col] = self._encode_cat_series_for_mice(col, series_raw)

        # ---------- 数值特征 ----------
        for col in self.num_cols:
            valid_data = df[col].dropna()
            is_int = bool(np.allclose(valid_data, np.round(valid_data)))
            structure = self._analyze_numeric_structure(valid_data, is_int=is_int)
            quantize_step = structure['quantize_step']

            if quantize_step is not None:
                q_str = f"{quantize_step:.10f}".rstrip('0')
                prec = len(q_str.split('.')[1]) if '.' in q_str else 0
            else:
                prec = self._infer_precision(valid_data)
            self.precisions[col] = prec

            if self.num_transform_method == 'quantile':
                encoder = QuantileTransformer(
                    n_quantiles=min(1000, len(valid_data)),
                    output_distribution='normal',
                    random_state=self.random_state,
                )
            elif self.num_transform_method == 'standard':
                encoder = StandardScaler()
            elif self.num_transform_method == 'minmax':
                encoder = MinMaxScaler()
            else:
                encoder = None

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

        # Bug 2 Fix: 统一通过辅助方法编码，OrdinalEncoder unknown → NaN 由辅助方法处理
        for col in self.cat_cols:
            df_temp[col] = self._encode_cat_series_for_mice(col, df[col])
        for col in self.num_cols:
            df_temp[col] = df[col]

        if df_temp[ordered_cols].isnull().any().any():
            imputed_array = self.imputer.transform(df_temp[ordered_cols])
            df_imputed = pd.DataFrame(imputed_array, columns=ordered_cols, index=df.index)
        else:
            df_imputed = df_temp[ordered_cols].copy()

        df_out = pd.DataFrame(index=df.index)

        for col in self.num_cols:
            enc = self.meta[col]['encoder']
            if enc is not None:
                df_out[col] = enc.transform(df_imputed[col].values.reshape(-1, 1)).flatten()
            else:
                df_out[col] = df_imputed[col].values

        for col in self.cat_cols:
            raw_vals = np.clip(
                np.round(df_imputed[col].values).astype(int),
                0, self.meta[col]['classes'] - 1
            )
            df_out[col] = raw_vals

        return df_out[ordered_cols].astype('float32')


    # =========================================================
    def transform_readable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        仅做 MICE 插补，不做分布变换，返回原始量纲的可读 DataFrame。
        插补器与 fit() 时完全共享，保证一致性。
        """
        assert hasattr(self, 'meta') and self.meta, "必须先调用 fit() 方法"
        ordered_cols = self.num_cols + self.cat_cols
        df_temp = pd.DataFrame(index=df.index)

        # Bug 2 Fix: 统一通过辅助方法编码
        for col in self.cat_cols:
            df_temp[col] = self._encode_cat_series_for_mice(col, df[col])
        for col in self.num_cols:
            df_temp[col] = df[col]

        df_imputed = pd.DataFrame(
            self.imputer.transform(df_temp[ordered_cols]),
            columns=ordered_cols,
            index=df.index,
        )

        df_out = pd.DataFrame(index=df.index)

        for col in self.num_cols:
            meta = self.meta[col]
            vals = np.clip(df_imputed[col].values, meta['min'], meta['max'])
            q = meta['quantize_step']
            if q is not None and q > 0:
                vals = np.round(vals / q) * q
                vals = np.round(vals, 6)
            prec = self.precisions.get(col, 2)
            vals = np.round(vals, prec)
            if prec == 0 or meta['is_int']:
                df_out[col] = vals.astype(np.int64)
            else:
                df_out[col] = vals

        for col in self.cat_cols:
            raw = df_imputed[col].values
            raw = np.where(np.isnan(raw), 0.0, np.round(raw))
            vals = np.clip(raw.astype(int), 0, self.meta[col]['classes'] - 1)
            # Bug 3 Fix: 调用统一辅助方法，内部处理 LabelEncoder/OrdinalEncoder 的维度差异
            labels = self._cat_inverse_transform(col, vals)
            try:
                df_out[col] = pd.to_numeric(labels)
            except (ValueError, TypeError):
                df_out[col] = labels

        extra_cols = [c for c in df.columns if c not in ordered_cols]
        for col in extra_cols:
            df_out[col] = df[col].values

        return df_out[df.columns.tolist()]
    
    
    def canonicalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        对已经做过缺失值处理的（预填充）数据，
        仅对数值列执行 clipping/rounding/类型恢复。
        cat 列已经是原始标签，直接透传。
        """
        df_out = pd.DataFrame(index=df.index)

        for col in self.num_cols:
            meta = self.meta[col]
            vals = np.clip(df[col].values.astype(float), meta['min'], meta['max'])
            q = meta['quantize_step']
            if q is not None and q > 0:
                vals = np.round(vals / q) * q
                vals = np.round(vals, 6)
            prec = self.precisions.get(col, 2)
            vals = np.round(vals, prec)
            df_out[col] = vals.astype(np.int64) if (prec == 0 or meta['is_int']) else vals

        for col in self.cat_cols:
            df_out[col] = df[col].values

        extra_cols = [c for c in df.columns if c not in self.num_cols + self.cat_cols]
        for col in extra_cols:
            df_out[col] = df[col].values

        return df_out[df.columns.tolist()]

    # =========================================================
    def inverse_transform(self, df_or_tensor):
        ordered_cols = self.num_cols + self.cat_cols

        if not isinstance(df_or_tensor, pd.DataFrame):
            df = pd.DataFrame(df_or_tensor, columns=ordered_cols)
        else:
            df = df_or_tensor.copy()

        df_out = pd.DataFrame(index=df.index)

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
                inversed = np.round(inversed / q) * q
                inversed = np.round(inversed, 6)
            prec = self.precisions.get(col, 2)
            inversed = np.round(inversed, prec)
            if prec == 0 or meta['is_int']:
                df_out[col] = pd.array(inversed, dtype=pd.Int64Dtype())
            else:
                df_out[col] = inversed

        for col in self.cat_cols:
            series = df[col]
            if series.dtype == object:
                df_out[col] = series
                continue
            raw = series.values
            raw = np.where(np.isnan(raw), 0.0, np.round(raw))
            vals = np.clip(raw.astype(int), 0, self.meta[col]['classes'] - 1)
            # Bug 3 Fix: 统一辅助方法
            labels = self._cat_inverse_transform(col, vals)
            try:
                df_out[col] = pd.to_numeric(labels)
            except (ValueError, TypeError):
                df_out[col] = labels

        return df_out

    # =========================================================
    def get_model_info(self):
        return {
            'num_numerical': len(self.num_cols),
            'num_classes':   [self.meta[col]['classes'] for col in self.cat_cols],
            'cat_cols':      self.cat_cols,
            'num_cols':      self.num_cols,
            'num_stats':     self.num_stats,
        }