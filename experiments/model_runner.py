# 文件位置: experiments/model_runner.py
"""
模型调用层：DataLoader / meta_info 构建、训练、生成、保存与加载。
只接收 data_prep 处理好的数据，不感知 task 路由细节。
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from dataclasses import asdict, replace
from torch.utils.data import DataLoader, TensorDataset

from core.diffusion import MedAGDiffusion
from configs.experiment_configs import ExperimentConfig


class InMemoryTensorLoader:
    """替代 DataLoader：数据整份常驻显存，直接张量索引取 batch。
    last_batch_strategy 控制"凑不满一个 batch 的尾巴"怎么处理：
        "drop" —— 直接丢弃（默认，原来的行为）
        "pad"  —— 从整份数据里随机重复采样补满到 batch_size，不丢数据，
                   代价是这一个 batch 里会有部分样本重复
    """
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int, 
                 device, shuffle: bool = True, last_batch_strategy: str = "drop"):
        assert last_batch_strategy in ("drop", "pad"), \
            f"last_batch_strategy 只能是 'drop' 或 'pad'，收到: {last_batch_strategy!r}"
        if batch_size > x.shape[0] and last_batch_strategy == "drop":
            print(f"⚠️  batch_size({batch_size}) > 样本数({x.shape[0]})，drop 策略下每个 epoch 没有可用 batch，建议改用 'pad'")
        self.x = x.to(device)
        self.y = y.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.last_batch_strategy = last_batch_strategy
        self.dataset = self.x

    def __iter__(self):
        n = self.x.shape[0]
        order = torch.randperm(n, device=self.x.device) if self.shuffle else torch.arange(n, device=self.x.device)
        usable_n = (n // self.batch_size) * self.batch_size

        for start in range(0, usable_n, self.batch_size):
            idx = order[start:start + self.batch_size]
            yield self.x[idx], self.y[idx]

        remainder = n - usable_n
        if remainder > 0 and self.last_batch_strategy == "pad":
            tail_idx = order[usable_n:]
            pad_idx = torch.randint(0, n, (self.batch_size - remainder,), device=self.x.device)
            idx = torch.cat([tail_idx, pad_idx])
            yield self.x[idx], self.y[idx]

    def __len__(self):
        n = self.x.shape[0]
        full_batches = n // self.batch_size
        if self.last_batch_strategy == "pad" and n % self.batch_size > 0:
            return full_batches + 1
        return full_batches

def build_dataloader(processor, df_features: pd.DataFrame, y: torch.Tensor,batch_size: int, 
                     shuffle: bool, device, last_batch_strategy: str = "drop") -> InMemoryTensorLoader:
    x_encoded = processor.transform(df_features)
    ordered_cols = processor.num_cols + processor.cat_cols
    x_tensor = torch.tensor(x_encoded[ordered_cols].values, dtype=torch.float32)
    return InMemoryTensorLoader(x_tensor, y, batch_size=batch_size, device=device, 
                                shuffle=shuffle, last_batch_strategy=last_batch_strategy)


def build_meta_info(processor, df_features: pd.DataFrame) -> dict:
    """按列名取值，不依赖 processor.transform() 输出的列顺序。"""
    info = processor.get_data_info()
    x_encoded = processor.transform(df_features)

    true_marginals = []
    for i, col in enumerate(processor.cat_cols):
        cat_col_data = x_encoded[col].values.astype(int)
        n_classes = info['num_classes'][i]
        counts = np.bincount(cat_col_data, minlength=int(n_classes))
        true_marginals.append(torch.tensor(counts / counts.sum(), dtype=torch.float32))

    info['num_classes'] = [int(n) for n in info['num_classes']]
    info['true_marginals'] = true_marginals
    return info


def train_model(denoise_class, model_cfg, train_cfg, sampling_cfg, train_loader, val_loader, meta_info) -> MedAGDiffusion:
    diffusion = MedAGDiffusion(
        model_cfg=model_cfg, train_cfg=train_cfg,
        sampling_cfg=sampling_cfg, denoise_class=denoise_class,
    )
    diffusion.fit(train_dataloader=train_loader, val_dataloader=val_loader, meta_info=meta_info)
    return diffusion


def decide_n_samples(target_dist: dict, task: str) -> int:
    t_count, b_count, total, ir = (
        target_dist["target_count"], target_dist["background_count"],
        target_dist["total"], target_dist["IR"],
    )
    if task == "full_data":
        return total
    if task in ("minority_uncond", "minority_cond"):
        if t_count > 0:
            return t_count
        if t_count == 0:
            return 1
        if ir >= 2:
            return max(b_count - t_count, 1)
        return 2 * t_count
    return total


def generate_samples(
    diffusion: MedAGDiffusion, processor, sampling_cfg,
    task: str, target_class: int, target_dist: dict,
    target_col: str, original_columns: list,
) -> pd.DataFrame:
    n_samples = sampling_cfg.num_samples if sampling_cfg.num_samples is not None \
        else decide_n_samples(target_dist, task)

    temp_cfg = replace(sampling_cfg, num_samples=n_samples)
    raw_tensor = diffusion.generate(sampling_cfg=temp_cfg, y_cond=target_class)

    ordered_cols = processor.num_cols + processor.cat_cols
    df_for_inverse = pd.DataFrame(raw_tensor.detach().cpu().numpy(), columns=ordered_cols)
    df_gen = processor.inverse_transform(df_for_inverse)

    if task != "full_data":
        df_gen[target_col] = target_class

    for col in original_columns:
        if col not in df_gen.columns:
            df_gen[col] = np.nan
    return df_gen[original_columns]


def save_run(
    save_path: str, processor, meta_info: dict, diffusion: MedAGDiffusion,
    target_dist: dict, target_col: str, cat_cols: list, model_feature_cols: list,
    original_columns: list, exp_cfg: ExperimentConfig, denoise_class,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    torch.save(diffusion.state_dict(), os.path.join(save_path, "model.pt"))

    bundle = {
        "processor": processor, "meta_info": meta_info, "target_dist": target_dist,
        "target_col": target_col, "cat_cols": cat_cols, "model_feature_cols": model_feature_cols,
        "original_columns": original_columns, "exp_cfg": exp_cfg, "denoise_class": denoise_class,
    }
    with open(os.path.join(save_path, "run_bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(exp_cfg), f, indent=4, ensure_ascii=False)


def load_run(save_path: str, denoise_class=None) -> dict:
    """
    denoise_class 现在可选：优先用 bundle 里存的那份，
    只有加载旧存档（没存 denoise_class）时才需要显式传入。
    """
    bundle_path = os.path.join(save_path, "run_bundle.pkl")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"{bundle_path} 不存在。")

    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    resolved_denoise_class = denoise_class or bundle.get("denoise_class")
    assert resolved_denoise_class is not None, \
        "run_bundle.pkl 中没有 denoise_class 记录（旧存档），请显式传入 load_run(save_path, denoise_class=...)"

    exp_cfg = bundle["exp_cfg"]
    if "cat_ar" not in vars(exp_cfg.model): # 旧存档 (w/o AR) 没有该字段
        exp_cfg.model.cat_ar = False
    diffusion = MedAGDiffusion(
        model_cfg=exp_cfg.model, train_cfg=exp_cfg.training,
        sampling_cfg=exp_cfg.sampling, denoise_class=resolved_denoise_class,
    )
    diffusion.build_network(bundle["meta_info"])
    state = torch.load(os.path.join(save_path, "model.pt"), map_location="cpu")
    diffusion.load_state_dict(state)

    bundle["diffusion"] = diffusion
    return bundle