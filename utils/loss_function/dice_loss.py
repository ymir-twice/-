"""
    此类任务一般使用 dice loss 、 混合 loss 等等 loss function
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

"""
    交叉熵损失函数，有一个非常差的地方： 对于只用于分割前景和背景的时候，当前景像素的数量远远小于背景像素的数量时，即 y = 0 的数量远大于 y = 1 的数量时， 损失函数中 y = 0 的成分就会占据主导，使模型严重偏向背景，导致效果不好
"""

class DiceLoss(nn.Module):
    """
        DiceLoss 在这类任务上也许比交叉熵好一点儿
    """
    def __init__(self, num_classes=4, weights=[1., 1, 1., 1.], ignore_index=0, smooth=1e-5):
        """
        :param num_classes: 类别数量，包含背景类别
        :param ignore_index: 需要忽略的类别，比如背景
        :param smooth: 避免除以0的平滑系数
        """
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.weights = weights

    def forward(self, logits, true):
        """
        :param logits: 模型输出的预测值，形状为(batch_size, num_classes, H, W)
        :param true: 真实标签，形状(batch_size, H, W)
        """
        # 使用 softmax 将模型输出转换为概率
        probs = F.softmax(logits, dim=1)

        # 初始化 dice loss
        dice_loss = 0.0

        # 遍历每个类别（跳过背景类）
        for class_idx in range(1, self.num_classes):
            # 获得当前类别的预测概率和真实标签的掩码
            pred = probs[:, class_idx, :, :]  # 预测属于当前类别的概率
            true_class = (true == class_idx).float() # 真实标签种属于当前类别的掩码

            # 计算交集和并集
            intersection = torch.sum(pred * true_class)
            union = torch.sum(pred) + torch.sum(true_class)

            # 计算当前类别的 Dice 系数
            dice_class = (2.0 * intersection + self.smooth) / (union + self.smooth)

            # 累加每个类别的 Dice Loss
            dice_loss += (1 - dice_class) * self.weights[class_idx]

        # 取平均值作为最终的损失
        return dice_loss / (self.num_classes - 1)
