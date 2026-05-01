# 文件位置: core/data_processor.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


class TabularDataProcessor:
    """
    支持 MICE 插补 + 分布映射 + 数值结构约束恢复 的表格数据双向处理器

    num_transform_method:
        'quantile' : QuantileTransformer → 正态分布（默认，配合标准MLP使用）
        'standard' : StandardScaler → 零均值单位方差
        'none'     : 不做分布变换，保留原始数值（配合 Periodic Embedding 使用）
    """

    def __init__(self, num_cols, cat_cols, num_transform_method='quantile', random_state=42):
        assert isinstance(num_cols, list) and isinstance(cat_cols, list)
        assert num_transform_method in ('quantile', 'standard', 'none'), \
            f"num_transform_method must be 'quantile', 'standard', or 'none', got '{num_transform_method}'"

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
    def _detect_quantize_step(self, unique_values, candidates=(1.0, 0.5, 0.25, 0.1, 0.01), tol=1e-6):
        """
        检测"最小公共步长"：验证所有值是否都是候选步长 q 的整数倍。
        例如 sysBP = [106, 127.5, 150, ...] → q=0.5，因为所有值 % 0.5 ≈ 0。

        返回满足条件的最大候选步长（即最粗粒度），如 q=1.0 满足则返回 1.0，
        否则尝试 0.5，以此类推。如果都不满足则返回 None（视为连续列）。
        """
        for q in candidates:
            remainders = np.abs(unique_values % q)
            # 余数接近 0 或接近 q（处理浮点边界）
            if np.all(np.minimum(remainders, np.abs(remainders - q)) < tol):
                return q
        return None

    def _analyze_numeric_structure(self, values, tolerance=1e-9):
        values = pd.Series(values).dropna().values
        unique = np.sort(np.unique(values))

        result = {
            'step': None,           # 严格等间距步长（如 education: 1,2,3,4 → step=1）
            'quantize_step': None,  # 最小公共步长（如 sysBP 的 0.5）
            'valid_set': unique,    # 保留原始唯一值（仅供参考，不再用于 snap）
        }

        if len(unique) <= 1:
            return result

        # ---------- 层1：检测严格等间距 ----------
        diffs = np.diff(unique)
        if np.all(np.abs(diffs - diffs[0]) < tolerance):
            result['step'] = float(diffs[0])
            # 等间距列同时也满足 quantize_step，不需要再单独检测
            result['quantize_step'] = float(diffs[0])
            return result

        # ---------- 层2：检测最小公共步长（非等间距但有规则精度）----------
        # 只在非等间距时才跑，避免重复
        q = self._detect_quantize_step(unique)
        if q is not None:
            result['quantize_step'] = q

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
            quantize_step = structure['quantize_step']  # None 表示真正连续列

            # ---------- 精度（小数位数）----------
            # 优先从 quantize_step 推导精度，确保与四舍五入步长一致
            if quantize_step is not None:
                q_str = f"{quantize_step:.10f}".rstrip('0').rstrip('.')
                prec = len(q_str.split('.')[1]) if '.' in q_str else 0
            else:
                sample = valid_data.sample(min(1000, len(valid_data)), random_state=self.random_state)
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
                # 'none'：不做分布变换，逆变换时也直接透传
                encoder = None

            self.meta[col] = {
                'type': 'num',
                'is_int': is_int,
                'min': valid_data.min(),
                'max': valid_data.max(),
                'encoder': encoder,
                # quantize_step: 逆变换时用 round(x/q)*q 对齐到合法值
                # None 表示连续列，逆变换只做精度舍入，不做离散化
                'quantize_step': quantize_step,
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
        # 4. 分布映射拟合（'none' 时跳过）
        # =========================
        for col in self.num_cols:
            if self.meta[col]['encoder'] is not None:
                self.meta[col]['encoder'].fit(
                    df_imputed[col].values.reshape(-1, 1)
                )

        # =========================
        # 5. 记录 'none' 模式下的归一化参数（供 Periodic Embedding 的 scale 参考）
        # =========================
        if self.num_transform_method == 'none':
            self.num_stats = {}
            for col in self.num_cols:
                col_data = df_imputed[col].values
                self.num_stats[col] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()) + 1e-8
                }
        else:
            self.num_stats = {}

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
            if self.meta[col]['encoder'] is not None:
                df_out[col] = self.meta[col]['encoder'].transform(
                    df_imputed[col].values.reshape(-1, 1)
                ).flatten()
            else:
                # 'none' 模式：直接透传（保留原始值，供 Periodic Embedding 使用）
                df_out[col] = df_imputed[col].values

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
        # 数值恢复
        # =========================
        for col in self.num_cols:
            meta = self.meta[col]
            vals = df[col].values.reshape(-1, 1)

            # ---------- 逆分布映射 ----------
            if meta['encoder'] is not None:
                inversed = meta['encoder'].inverse_transform(vals).flatten()
            else:
                # 'none' 模式：直接透传
                inversed = vals.flatten()

            # ---------- 边界裁剪 ----------
            inversed = np.clip(inversed, meta['min'], meta['max'])

            quantize_step = meta['quantize_step']

            # ---------- 量化对齐（仅对有 quantize_step 的列）----------
            # 逻辑：round(x / q) * q，将生成值四舍五入到最近的合法值
            # 例：sysBP q=0.5 → 127.3 → round(127.3/0.5)*0.5 = 127.5
            #     education q=1.0 → 2.3 → round(2.3/1.0)*1.0 = 2.0
            # 连续列（q=None）跳过，只做后续精度舍入
            if quantize_step is not None and quantize_step > 0:
                inversed = np.round(inversed / quantize_step) * quantize_step
                inversed = np.round(inversed, 6)  # 消除浮点误差

            # ---------- 精度舍入 ----------
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
            'num_cols': self.num_cols,
            # 供 Periodic Embedding 使用的统计信息（'none' 模式下才有值）
            'num_stats': self.num_stats if hasattr(self, 'num_stats') else {}
        }
