# core/utils.py
import torch
import torch.nn.functional as F
import numpy as np

def extract(a, t, x_shape):
    """从常数序列中提取指定时间步t的值，并调整形状以进行广播"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,)*(len(x_shape)-1))).expand(x_shape)

@torch._dynamo.disable
def index_to_log_onehot(x, num_classes):
    # 将索引转换为对数one-hot形式，用于多项式扩散
    onehots = []
    for i in range(len(num_classes)):
        onehots.append(F.one_hot(x[:, i].long(), num_classes[i]))
    x_onehot = torch.cat(onehots, dim=1).float()
    return torch.log(x_onehot.clamp(min=1e-30))

def log_1_min_a(a):
    a = torch.clamp(a, max=-1e-7)
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    res = maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum) + 1e-40)
    return res

def ohe_to_categories(ohe, num_classes):
    # 将生成的one-hot还原为原始类别索引
    indices = np.append([0], np.cumsum(num_classes))
    res = []
    for i in range(len(indices)-1):
        res.append(ohe[:, indices[i]:indices[i+1]].argmax(dim=1))
    return torch.stack(res, dim=1)