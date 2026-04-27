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

from .diff_losses import compute_categorical_loss, compute_numerical_loss
from .diff_schedules import DiffusionScheduleMixin
from .diff_sampling import sample_core
from .diff_training import train_loop

class  MedAGDiffusion(DiffusionScheduleMixin, nn.Module):
    """
    统一医学表格生成扩散模型 (Unified Controllable Diffusion Framework)
    为 SCI 科研论文设计，支持 Joint (联合分布) 与 Conditional (条件分布) 双范式。
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

    def _setup_training_for_task(self, df_train, target_col, cat_cols, data_config):
        """
        输入原始 df_train、target_col、原始 cat_cols，
        输出: df_train_for_proc, actual_cat_cols, y_train_tensor
        """
        task = self.train_cfg.task
        self.actual_cat_cols = deepcopy(cat_cols)

        if task == "minority_uncond":
            df_task = df_train[df_train[target_col] == 1]
            if target_col in self.actual_cat_cols:
                self.actual_cat_cols.remove(target_col)
            y_train = torch.full((len(df_task),), self.cfg.null_class_idx, dtype=torch.long)
            df_for_proc = df_task.drop(columns=[target_col])

        elif task == "minority_cond":
            df_task = df_train  # 用全量
            if target_col in self.actual_cat_cols:
                self.actual_cat_cols.remove(target_col)
            y_train = torch.tensor(df_task[target_col].values, dtype=torch.long)
            df_for_proc = df_task.drop(columns=[target_col])

        elif task == "full_data":
            use_joint = True
            df_task = df_train
            if use_joint:
                if target_col not in self.actual_cat_cols:
                    self.actual_cat_cols.append(target_col)
                y_train = torch.full((len(df_task),), self.cfg.null_class_idx, dtype=torch.long)
                df_for_proc = df_task
            else:
                if target_col in self.actual_cat_cols:
                    self.actual_cat_cols.remove(target_col)
                y_train = torch.tensor(df_task[target_col].values, dtype=torch.long)
                df_for_proc = df_task.drop(columns=[target_col])
        else:
            raise ValueError(f"Unknown task: {task}")

        return df_for_proc, self.actual_cat_cols, y_train

    # =========================================================
    # 模型拟合 
    # =========================================================
    def fit(self, 
            df: pd.DataFrame, 
            target_col: str, 
            num_cols: list, 
            cat_cols: list, 
            train_config: TrainingConfig,
            data_config: DataConfig):
        
        self.train_cfg = train_config
        self.target_col = target_col
        self.original_df_columns = df.columns.tolist() # 记录最原始的列顺序
        df_train = df.copy()

        counts = df_train[target_col].value_counts()
        self.target_dist_info = {
            'counts': counts.to_dict(),
            'majority_class': counts.idxmax(),
            'majority_count': counts.max(),
            'minority_class': counts.idxmin(),
            'minority_count': counts.min(),
            'original_total': len(df_train)
        }

        df_train_for_proc, self.actual_cat_cols, y_train_tensor = \
            self._setup_training_for_task(df_train, target_col, cat_cols, data_config)

        self.data_processor = TabularDataProcessor(
            num_cols=num_cols, 
            cat_cols=self.actual_cat_cols, 
            num_transform_method=data_config.num_transform_method,
            random_state=data_config.random_state
        )
        self.data_processor.fit(df_train_for_proc)
        df_trans = self.data_processor.transform(df_train_for_proc)

        self.meta_info = self.data_processor.get_model_info()
        self.num_numerical_features = self.meta_info['num_numerical']
        self.num_classes = self.meta_info['num_classes']

        # [修正分布趋向均值的问题]记录全局真实的分类边缘先验分布（Marginal Priors)
        self.true_marginals = []
        cat_start_idx = self.num_numerical_features

        for i, n_classes in enumerate(self.num_classes):
            # 获取该分类列的频数统计并归一化
            col_data = df_trans.iloc[:, cat_start_idx + i]
            val_counts = col_data.value_counts(normalize=True).sort_index()
            marginal = torch.zeros(n_classes)
            for k, v in val_counts.items():
                marginal[int(k)] = v
            # 保存并加入微小偏移防止Log(0)
            self.true_marginals.append(torch.clamp(marginal.to(self.device), min=1e-6))
        
        self._init_class_expanded()
        
        self._denoise_fn = self.denoise_class(
            info=self.meta_info,
            dim_t=self.cfg.dim_t,
            dim_hidden=self.cfg.dim_hidden,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout
        ).to(self.device)

        dataset = TensorDataset(
            torch.tensor(df_trans.values, dtype=torch.float32),
            y_train_tensor
        )
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

        train_loop(self, dataloader, train_config)

        return self

    # =========================================================
    # 上帝类 API: 模型生成
    # =========================================================
    def _postprocess_generated_for_task(self, raw_tensor, n_samples, y_cond, sampling_config):
        task = self.train_cfg.task

        if task == "minority_uncond":
            df_gen = self.data_processor.inverse_transform(raw_tensor.cpu().numpy())
            df_gen[self.target_col] = 1

        elif task == "minority_cond":
            if y_cond is None:
                y_cond = 1
            df_gen = self.data_processor.inverse_transform(raw_tensor.cpu().numpy())
            df_gen[self.target_col] = y_cond

        elif task == "full_data":
            use_joint = True
            df_gen = self.data_processor.inverse_transform(raw_tensor.cpu().numpy())
            if not use_joint:
                # conditional full_data 情况下需要补 label
                assert y_cond is not None
                df_gen[self.target_col] = y_cond
        else:
            raise ValueError(f"Unknown task: {task}")
        return df_gen


    @torch.no_grad()
    def generate(self, sampling_config: SamplingConfig, y_cond: int = None, return_latent:bool = False) -> pd.DataFrame:
        self._denoise_fn.eval()
        assert sampling_config.num_samples is not None, "SamplingConfig.num_samples 未设置"
        n_samples = sampling_config.num_samples

        # ----- full_data任务下，一律使用Null条件-----
        if self.train_cfg.task == "full_data":
            y_for_sampling = self.cfg.null_class_idx
        else:
            y_for_sampling = (y_cond if y_cond is not None else self.cfg.null_class_idx)

        raw = sample_core(self, n_samples, y_for_sampling, cfg=sampling_config)

        if return_latent:
            # 直接把latent数值部分和完整raw 一起返回，用于debug
            x_num_latent = raw[:, :self.num_numerical_features].clone()
            self._denoise_fn.train()
            return x_num_latent.cpu(), raw.cpu()

        df_gen = self._postprocess_generated_for_task(raw, n_samples, y_cond, sampling_config)
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
        if self.train_cfg.task == 'minority_cond':
            drop_mask = torch.rand((b,), device=self.device) < self.train_cfg.uncond_p
            y_cond[drop_mask] = self.cfg.null_class_idx
        else:
            y_cond.fill_(self.cfg.null_class_idx) 

        x_num = batch_x[:, :self.num_numerical_features].float()
        x_cat = batch_x[:, self.num_numerical_features:].long()

        noise = torch.randn_like(x_num)
        x_num_t = extract(self.sqrt_alphas_cumprod, t, x_num.shape) * x_num + \
                  extract(self.sqrt_one_minus_alphas_cumprod, t, x_num.shape) * noise

        log_x_cat = index_to_log_onehot(x_cat, self.num_classes)
        log_cumprod_alpha_t = extract(self.log_cumprod_alpha, t, log_x_cat.shape)
        log_1_min_cumprod_alpha_t = extract(self.log_1_min_cumprod_alpha, t, log_x_cat.shape)
        
        log_x_cat_t = log_add_exp(
            log_x_cat + log_cumprod_alpha_t, #原始类别 x alpha_bar_t
            log_1_min_cumprod_alpha_t - torch.log(self.num_classes_expanded)
        )
        
        x_in = torch.cat([x_num_t, log_x_cat_t.exp()], dim=1)
        model_out = self._denoise_fn(x_in, t, y_cond)

        loss_num, loss_corr = compute_numerical_loss(self, x_num, x_num_t, model_out, t, noise)
        loss_cat = compute_categorical_loss(self, x_cat, model_out, use_basic=True)

        total_loss = loss_num + loss_cat + self.cfg.corr_weight * loss_corr
        return total_loss, loss_num, loss_cat, loss_corr        


    def _init_class_expanded(self):
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([np.full(c, c) for c in self.num_classes])
        ).float().to(self.device)

        offsets = np.append([0], np.cumsum(self.num_classes))
        self.slices_for_classes = [
            np.arange(offsets[i], offsets[i + 1]) for i in range(len(self.num_classes))
        ]
