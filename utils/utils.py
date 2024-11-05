import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

def __init__weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                   **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            #m.eps = bn_eps
            #m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps=1e-5, bn_momentum=0.5, **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init__weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)
    else:
        __init__weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def count_model_parameters(model_path):
    """
    计算模型参数总量
    :param model_path: 模型文件路径（.pt或.pth），要求选手使用 torch.save(model, 'model.pth') 保存模型
    :return: 模型参数总量
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")

    # 加载模型
    model = torch.load(model_path, weights_only=False, map_location=torch.device("cuda:0"))

    # 计算参数总量
    total_params = sum(p.numel() for p in model.parameters())
    return total_params