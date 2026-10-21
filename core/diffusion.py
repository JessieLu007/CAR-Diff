# 文件位置: core/diffusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

# 导入底层工具与配置类
from .utils import extract, index_to_log_onehot, log_1_min_a, log_add_exp, ohe_to_categories
from configs.experiment_configs import ModelConfig, TrainingConfig, SamplingConfig

from .loss import compute_numerical_loss, compute_categorical_loss
from .diff_schedules import DiffusionScheduleMixin
from .diff_sampling import sample_core
from .diff_training import train_loop, smooth_labels


# ---------------------------------------------------------------------------
# task 取值说明（对应 TrainingConfig.task）
#
#   "full_data"
#       训练数据：全量（含 target_col）
#       建模范式：Joint 联合分布 P(X, Y)，target_col 作为普通分类特征一起学
#       y 条件：全部置为 null_class_idx（无监督引导）
#       典型用途：TSTR、替代数据集生成
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

    职责边界：
        - 接收已准备好的 DataLoader 和 meta_info，不感知原始 DataFrame
        - 负责扩散过程的前向加噪、损失计算、去噪采样
        - generate() 返回原始 tensor，逆变换与列补全由调用方负责
    """

    def __init__(
            self, 
            model_cfg: ModelConfig,
            train_cfg: TrainingConfig,
            sampling_cfg: SamplingConfig,
            denoise_class
            ):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.sampling_cfg = sampling_cfg

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.denoise_class = denoise_class
        self._denoise_fn = None

        # 初始化扩散 Schedule Buffers
        self._init_diffusion_buffers(self.model_cfg.num_timesteps)

        # 索引0为num，1为cat，用来调节损失权重
        self.log_vars = nn.Parameter(torch.zeros(2))


    # diffusion.py 中，_count_parameters 改为 staticmethod
    @staticmethod
    def _count_parameters(model: nn.Module) -> None:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{'-'*45}")
        print(f"    模型参数规模")
        print(f"{'-'*45}")
        for name, module in model.named_children():
            n = sum(p.numel() for p in module.parameters())
            print(f"    {name:<30} {n:>8,}")
        print(f"{'-'*45}")
        print(f"    {'Total':<30} {total:>8,}")
        print(f"    {'Trainable':<30} {trainable:>8,}")
        print(f"{'-'*45}\n")

    def build_network(self, meta_info: dict)-> "MedAGDiffusion":
        """
        仅初始化网络结构（不训练)
        """
        self.meta_info              = meta_info
        self.num_numerical_features = meta_info['num_numerical']
        self.num_classes            = [int(n) for n in meta_info['num_classes']]

        self._init_class_expanded()

        # minority 模式下提高 dropout, 一直过拟合
        is_minority = self.train_cfg.task in ('minority_uncond', 'minority_cond')
        dropout = 0.2 if is_minority else self.model_cfg.dropout

        net_kwargs = dict(
            task=self.train_cfg.task,
            info=self.meta_info,
            num_classes_cond=self.model_cfg.null_class_idx,   # 嵌入表大小 = null_class_idx + 1
            dim_t=self.model_cfg.dim_t,
            dim_hidden=self.model_cfg.dim_hidden,
            num_layers=self.model_cfg.num_layers,
            dropout=dropout,
        )
        if self.model_cfg.share_decoder is not True:
            net_kwargs['share_decoder'] = False
        self._denoise_fn = self.denoise_class(**net_kwargs).to(self.device)

        self.to(self.device)

        return self

    # =========================================================
    # 模型拟合
    # 调用方负责：数据路由、预处理、DataLoader 构建、meta_info 准备
    # =========================================================
    def fit(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        meta_info: dict,
    ):
        """
        参数：
            train_dataloader  —— 每个 batch 为 (x_tensor, y_tensor)
            val_dataloader    —— 同上，可为 None
            meta_info         —— 包含 num_numerical、num_classes、true_marginals 等
                                 由调用方（MedSynth）负责构建
            train_config      —— 训练超参数
        """
        self.build_network(meta_info)
        self._count_parameters(self._denoise_fn)
        
        train_loop(self, train_dataloader, val_dataloader, self.train_cfg)

        return self

    # =========================================================
    # 采样生成
    # 返回原始 tensor，不做逆变换，不补 target_col
    # =========================================================
    @torch.no_grad()
    def generate(
        self,
        sampling_cfg: SamplingConfig,
        y_cond: int = None,
        return_latent: bool = False,
    ) -> torch.Tensor:
        """
        返回 shape=(n_samples, n_features) 的原始 tensor。
        逆变换与列补全由调用方（MedSynth）负责。
        """
        self._denoise_fn.eval()
        assert sampling_cfg.num_samples is not None, "SamplingConfig.num_samples 未设置"

        # full_data 模式下模型无视外部条件，一律使用 null
        if self.train_cfg.task in ("full_data", "minority_uncond"):
            y_for_sampling = self.model_cfg.null_class_idx
        else:
            y_for_sampling = y_cond if y_cond is not None else self.model_cfg.null_class_idx

        # 将sampling_cfg传递给sample相关函数
        raw = sample_core(self, sampling_cfg.num_samples, y_for_sampling)

        self._denoise_fn.train()

        if return_latent:
            x_num_latent = raw[:, :self.num_numerical_features].clone()
            return x_num_latent, raw

        return raw


    def compute_loss(self, batch_x, batch_y):
        b = batch_x.shape[0]
        t = torch.randint(0, self.model_cfg.num_timesteps, (b,), device=self.device).long()

        y_cond = batch_y.clone()

        # conditional 范式：按 uncond_p 随机丢弃标签，实现 CFG 训练
        # joint / uncond 范式：标签已在路由阶段全部置为 null，此处保持原值即可
        if self.train_cfg.task == "minority_cond":
            drop_mask = torch.rand((b,), device=self.device) < self.train_cfg.uncond_p
            y_cond[drop_mask] = self.model_cfg.null_class_idx

            if getattr(self.train_cfg, "use_label_smoothing", False):
                keep_mask = ~drop_mask
                if keep_mask.any():
                    y_cond[keep_mask] = smooth_labels(
                        y_cond[keep_mask],
                        num_classes=self.model_cfg.null_class_idx,
                        smooth_prob=self.train_cfg.label_smooth_prob,
                    )

        x_num = batch_x[:, :self.num_numerical_features].float()

        x_cat = batch_x[:, self.num_numerical_features:].long()

        noise = torch.randn_like(x_num)
        x_num_t = (
            extract(self.sqrt_alphas_cumprod, t, x_num.shape) * x_num
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_num.shape) * noise
        )

        # 类别列不参与扩散：网络只接收加噪后的数值列，类别由分类头预测
        model_out = self._denoise_fn(x_num_t, t, y_cond)

        loss_num = compute_numerical_loss(self, x_num, x_num_t, model_out, t, noise)
        loss_cat = compute_categorical_loss(self, x_cat, model_out)

        # 使用不确定性加权公式：Loss = (1 / e^log_var) * L + log_var
        prec_num = torch.exp(-self.log_vars[0])
        prec_cat = torch.exp(-self.log_vars[1])

        total_loss = (prec_num * loss_num + self.log_vars[0]) + \
                     (prec_cat * loss_cat + self.log_vars[1])

        return total_loss, loss_num, loss_cat

    def _init_class_expanded(self):
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([np.full(c, c) for c in self.num_classes])
        ).float().to(self.device)

        offsets = np.append([0], np.cumsum(self.num_classes))
        self.slices_for_classes = [
            torch.arange(int(offsets[i]), int(offsets[i+1]), device=self.device)
            for i in range(len(self.num_classes))
        ]
