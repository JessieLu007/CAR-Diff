import os
import sys


def init_environment():
    """一次性设置环境变量和路径"""
    #os.environ["OMP_NUM_THREADS"] = "1"
    #os.environ["OPENBLAS_NUM_THREADS"] = "1"
    #os.environ["MKL_NUM_THREADS"] = "1"
    #os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    #os.environ["NUMEXPR_NUM_THREADS"] = "1"
    sys.path.append("..")
    # 也可以在这里导入 env_config 并执行其初始化
    import env_config
    #return 0

import sys
import torch
import json
import pickle
import gc
import importlib
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from configs.experiment_configs import DataConfig, ModelConfig, TrainingConfig, SamplingConfig, ExperimentConfig, make_default_experiment_config

'''
# 2. (可选) 医学物理先验回调函数
def inject_prior(df): 
    if 'sysBP' in df.columns and 'diaBP' in df.columns:
        df['pulse_pressure'] = df['sysBP'] - df['diaBP']
    if TARGET_COL not in df.columns: df[TARGET_COL] = 1 # 填补原本缺失的标签
    return df

def drop_prior(df):
    if 'pulse_pressure' in df.columns:
        df = df.drop(columns=['pulse_pressure'])
    return df
'''