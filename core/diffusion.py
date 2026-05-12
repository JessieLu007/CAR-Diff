# 文件位置: core/diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy

# 导入底层工具与配置类
from .utils import extract, index_to_log_onehot, log_1_min_a, log_add_exp, ohe_to_categories
from .data_processor import TabularDataProcessor
from configs.experiment_configs import ModelConfig, TrainingConfig, SamplingConfig, DataConfig

from .diff_losses import compute_numerical_loss #compute_categorical_loss, 
from .cat_loss import compute_categorical_loss
from .diff_schedules import DiffusionScheduleMixin
from .diff_sampling import sample_core
from .diff_training import train_loop


# ---------------------------------------------------------------------------
# task 取值说明（对应 TrainingConfig.task）
#
#   "full_data_joint"
#       训练数据：全量（含 target_col）
#       建模范式：Joint 联合分布 P(X, Y)，target_col 作为普通分类特征一起学
#       y 条件：全部置为 null_class_idx（无监督引导）
#       典型用途：TSTR、替代数据集生成
#
#   "full_data_cond"
#       训练数据：全量（target_col 从特征中移除，作为条件标签）
#       建模范式：Conditional P(X | Y)，用真实标签做 CFG 引导
#       y 条件：真实 target_col 值，训练时按 uncond_p 随机丢弃为 null
#       典型用途：按类别精细控制生成质量
#
#   "minority_uncond"
#       训练数据：仅 label=1 的少数类子集
#       建模范式：无条件 P(X | Y=1)，target_col 不参与特征也不作条件
#       y 条件：全部置为 null_class_idx
#       典型用途：少数类拟真扩增（简单场景）
#
#   "minority_cond"
#       训练数据：全量（target_col 从特征中移除，作为条件标签）
#       建模范式：Conditional P(X | Y)，但生成时只采样 Y=1
#       y 条件：真实 target_col 值，训练时按 uncond_p 随机丢弃
#       典型用途：CFG 引导的少数类增强（分类器提升场景）
#
# ---------------------------------------------------------------------------


class MedAGDiffusion(DiffusionScheduleMixin, nn.Module):
    """
    统一医学表格生成扩散模型 (Unified Controllable Diffusion Framework)
    为 SCI 科研论文设计，支持 Joint (联合分布) 与 Conditional (条件分布) 双范式。

    调用方只需始终传入全量训练数据（df_train），
    内部根据 TrainingConfig.task 自动完成数据筛选、标签路由与特征剥离。
    """

    def __init__(self, model_config: ModelConfig, denoise_class):
        super().__init__()
        self.cfg = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_timesteps = self.cfg.num_timesteps
        self.denoise_class = denoise_class
        self._denoise_fn = None
        self.data_processor = None

        # 保存原始训练集的分布信息，用于后续的 n_samples 自动推断
        self.target_dist_info = {}

        # 初始化扩散 Schedule Buffers
        self._init_diffusion_buffers(self.num_timesteps)

        # 索引0为num，1为cat，用来调节损失权重
        self.log_vars = nn.Parameter(torch.zeros(2))

    # =========================================================
    # 数据路由：根据 task 决定"送什么数据给 processor"
    # =========================================================

    def _route_training_data(
        self,
        df_train: pd.DataFrame,
        target_col: str,
        cat_cols: list,
    ) -> tuple[pd.DataFrame, list, torch.Tensor]:
        """
        根据 task 对训练数据做筛选、标签提取与特征剥离。

        返回：
            df_for_proc   —— 送入 TabularDataProcessor 的 DataFrame（不含 target_col，除非是 joint 模式）
            actual_cat_cols —— 实际传给 processor 的分类列列表
            y_train_tensor  —— 每条样本对应的条件标签 tensor
        """
        task = self.train_cfg.task
        actual_cat_cols = deepcopy(cat_cols)

        # ------------------------------------------------------------------
        # Joint 联合分布：target_col 作为普通分类特征保留在 df_for_proc 中
        # 所有样本 y 均设为 null（模型不接受外部条件引导）
        # ------------------------------------------------------------------
        if task == "full_data_joint":
            if target_col not in actual_cat_cols:
                actual_cat_cols.append(target_col)
            df_for_proc = df_train.copy()
            y_train = torch.full((len(df_train),), self.cfg.null_class_idx, dtype=torch.long)

        # ------------------------------------------------------------------
        # 条件全量：target_col 剥离为条件标签，模型学 P(X | Y)
        # ------------------------------------------------------------------
        elif task == "full_data_cond":
            if target_col in actual_cat_cols:
                actual_cat_cols.remove(target_col)
            df_for_proc = df_train.drop(columns=[target_col])
            y_train = torch.tensor(df_train[target_col].values, dtype=torch.long)

        # ------------------------------------------------------------------
        # 无条件少数类：仅使用 label=1 子集，target_col 完全丢弃
        # ------------------------------------------------------------------
        elif task == "minority_uncond":
            df_minority = df_train[df_train[target_col] == 1].copy()
            if target_col in actual_cat_cols:
                actual_cat_cols.remove(target_col)
            df_for_proc = df_minority.drop(columns=[target_col])
            y_train = torch.full((len(df_minority),), self.cfg.null_class_idx, dtype=torch.long)

        # ------------------------------------------------------------------
        # 条件少数类：全量训练学 P(X | Y)，生成时只采样 Y=1
        # ------------------------------------------------------------------
        elif task == "minority_cond":
            if target_col in actual_cat_cols:
                actual_cat_cols.remove(target_col)
            df_for_proc = df_train.drop(columns=[target_col])
            y_train = torch.tensor(df_train[target_col].values, dtype=torch.long)

        else:
            raise ValueError(
                f"未知 task='{task}'，合法值: "
                "'full_data_joint' | 'full_data_cond' | 'minority_uncond' | 'minority_cond'"
            )

        return df_for_proc, actual_cat_cols, y_train

    # =========================================================
    # 模型拟合
    # =========================================================

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str,
        num_cols: list,
        cat_cols: list,
        train_config: TrainingConfig,
        data_config: DataConfig,
    ):
        self.train_cfg = train_config
        self.target_col = target_col
        self.original_df_columns = df.columns.tolist()
        df_train = df.copy()

        # 记录目标列分布信息（始终基于全量数据统计）
        counts = df_train[target_col].value_counts()
        self.target_dist_info = {
            'counts': counts.to_dict(),
            'majority_class': counts.idxmax(),
            'majority_count': counts.max(),
            'minority_class': counts.idxmin(),
            'minority_count': counts.min(),
            'original_total': len(df_train),
        }

        # 数据路由：根据 task 决定送给 processor 的数据与标签
        df_train_for_proc, self.actual_cat_cols, y_train_tensor = \
            self._route_training_data(df_train, target_col, cat_cols)

        # 构建 & 拟合数据处理器
        self.data_processor = TabularDataProcessor(
            num_cols=num_cols,
            cat_cols=self.actual_cat_cols,
            num_transform_method=data_config.num_transform_method,
            random_state=data_config.random_state,
        )
        self.data_processor.fit(df_train_for_proc)
        df_trans = self.data_processor.transform(df_train_for_proc)

        self.meta_info = self.data_processor.get_model_info()
        self.num_numerical_features = self.meta_info['num_numerical']
        self.num_classes = self.meta_info['num_classes']

        # 记录全局真实的分类边缘先验分布（Marginal Priors），修正分布趋向均值的问题
        self.true_marginals = []
        cat_start_idx = self.num_numerical_features

        # 确定用哪些行来统计边缘分布
        if self.train_cfg.task in ('minority_cond', 'minority_uncond'):
            # 用原始标签筛选少数类进行，再对应到df_trans的索引
            minority_mask = (df_train_for_proc.index.isin(
                df_train[df_train[target_col] == 1].index
            ))
            df_trans_ref = df_trans[minority_mask]
        else:
            df_trans_ref = df_trans # full_data任务用全量

        for i, n_classes in enumerate(self.num_classes):
            col_data = df_trans_ref.iloc[:, cat_start_idx + i]
            val_counts = col_data.value_counts(normalize=True).sort_index()
            marginal = torch.zeros(n_classes)
            for k, v in val_counts.items():
                marginal[int(k)] = v
            self.true_marginals.append(torch.clamp(marginal.to(self.device), min=1e-6))

        self._init_class_expanded()

        # 将true_marginals加入到meta_info中传递
        self.meta_info['true_marginals'] = self.true_marginals

        self._denoise_fn = self.denoise_class(
            info=self.meta_info, # 当前info字典中包含了true_marginals
            dim_t=self.cfg.dim_t,
            dim_hidden=self.cfg.dim_hidden,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(df_trans.values, dtype=torch.float32),
            y_train_tensor,
        )
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

        train_loop(self, dataloader, train_config)

        return self

    # =========================================================
    # 生成后处理：根据 task 决定如何补回 target_col
    # =========================================================

    def _postprocess_generated(
        self,
        raw_tensor: torch.Tensor,
        y_cond: int | None,
    ) -> pd.DataFrame:
        """
        将模型输出的原始 tensor 逆变换为 DataFrame，并按 task 补回 target_col。

        full_data_joint  : target_col 已在特征内，processor 负责还原，无需额外补充
        full_data_cond   : 调用方通过 y_cond 指定要补的标签值
        minority_uncond  : 恒补 label=1
        minority_cond    : 由 y_cond 指定（通常为 1）
        """
        task = self.train_cfg.task
        df_gen = self.data_processor.inverse_transform(raw_tensor.cpu().numpy())

        if task == "full_data_joint":
            # target_col 由 processor 一并还原，不需要额外处理
            pass

        elif task == "full_data_cond":
            assert y_cond is not None, \
                "task='full_data_cond' 时，generate() 必须传入 y_cond 指定目标类别"
            df_gen[self.target_col] = y_cond

        elif task == "minority_uncond":
            df_gen[self.target_col] = 1

        elif task == "minority_cond":
            df_gen[self.target_col] = y_cond if y_cond is not None else 1

        else:
            raise ValueError(f"未知 task='{task}'")

        return df_gen

    # =========================================================
    # 上帝类 API：模型生成
    # =========================================================

    @torch.no_grad()
    def generate(
        self,
        sampling_config: SamplingConfig,
        y_cond: int = None,
        return_latent: bool = False,
    ) -> pd.DataFrame:
        self._denoise_fn.eval()
        assert sampling_config.num_samples is not None, "SamplingConfig.num_samples 未设置"
        n_samples = sampling_config.num_samples

        # joint 模式下模型无视外部条件，一律使用 null
        if self.train_cfg.task == "full_data_joint":
            y_for_sampling = self.cfg.null_class_idx
        else:
            y_for_sampling = y_cond if y_cond is not None else self.cfg.null_class_idx

        raw = sample_core(self, n_samples, y_for_sampling, cfg=sampling_config)

        if return_latent:
            x_num_latent = raw[:, :self.num_numerical_features].clone()
            self._denoise_fn.train()
            return x_num_latent.cpu(), raw.cpu()

        df_gen = self._postprocess_generated(raw, y_cond)
        self._denoise_fn.train()

        if self.data_processor.original_columns is not None:
            return df_gen[self.data_processor.original_columns]
        return df_gen

    # =========================================================
    # 核心数学模块
    # =========================================================

    def compute_loss(self, batch_x, batch_y):
        b = batch_x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()

        y_cond = batch_y.clone()

        # conditional 范式：按 uncond_p 随机丢弃标签，实现 CFG 训练
        # joint / uncond 范式：标签已在路由阶段全部置为 null，此处保持原值即可
        if self.train_cfg.task in ("full_data_cond", "minority_cond"):
            drop_mask = torch.rand((b,), device=self.device) < self.train_cfg.uncond_p
            y_cond[drop_mask] = self.cfg.null_class_idx

        x_num = batch_x[:, :self.num_numerical_features].float()
        x_cat = batch_x[:, self.num_numerical_features:].long()

        noise = torch.randn_like(x_num)
        x_num_t = (
            extract(self.sqrt_alphas_cumprod, t, x_num.shape) * x_num
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_num.shape) * noise
        )

        log_x_cat = index_to_log_onehot(x_cat, self.num_classes)
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_cat.shape)
        log_1_min_cumprod_alpha_t = extract(self.log_1_min_cumprod_alpha, t, log_x_cat.shape)

        log_x_cat_t = log_add_exp(
            log_x_cat + log_cumprod_alpha_t,
            log_1_min_cumprod_alpha_t - torch.log(self.num_classes_expanded),
        )

        x_in = torch.cat([x_num_t, log_x_cat_t.exp()], dim=1)
        model_out = self._denoise_fn(x_in, t, y_cond)

        loss_num, loss_corr = compute_numerical_loss(self, x_num, x_num_t, model_out, t, noise)
        loss_cat = compute_categorical_loss(self, x_cat, model_out, use_basic=self.train_cfg.use_basic)
        
        # 使用不确定性加权公式：Loss = (1 / e^log_var) * L + log_var
        prec_num = torch.exp(-self.log_vars[0])
        prec_cat = torch.exp(-self.log_vars[1])

        total_loss = (prec_num * loss_num + self.log_vars[0]) + \
                     (prec_cat * loss_cat + self.log_vars[1])

        return total_loss, loss_num, loss_cat, loss_corr

    def _init_class_expanded(self):
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([np.full(c, c) for c in self.num_classes])
        ).float().to(self.device)

        offsets = np.append([0], np.cumsum(self.num_classes))
        self.slices_for_classes = [
            np.arange(offsets[i], offsets[i + 1]) for i in range(len(self.num_classes))
        ]