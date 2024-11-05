import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class ANetDiceLoss(nn.Module):
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
        probs = F.softmax(logits, dim=1)
        dice_loss = 0.0

        for class_idx in range(1, self.num_classes):
            pred = probs[:, class_idx, :, :]
            true_class = (true == class_idx).float()

            # 计算交集和并集，加入平方项
            intersection = torch.sum(pred * true_class)
            pred_sum = torch.sum(pred ** 2)
            true_sum = torch.sum(true_class ** 2)

            dice_class = (2.0 * intersection + self.smooth) / (pred_sum + true_sum + self.smooth)

            dice_loss += (1 - dice_class) * self.weights[class_idx]

        return dice_loss / (self.num_classes - 1)

class ANetLoss(nn.Module):
    def __init__(self, num_class=4, class_weights=[1, 1, 1, 1], dice_weight=1.0, bce_weight=0.5):
        super().__init__()
        class_weights = np.array(class_weights)
        self.dice_loss = ANetDiceLoss(num_class, class_weights / class_weights.sum())
        self.bce_loss = nn.BCEWithLogitsLoss(reduce='mean')
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, true):
        # Dice Loss
        dice = self.dice_loss(logits, true)
        
        # BCE Loss (注意，BCEWithLogitsLoss需要logits和true的shape一致)
        # 如果logits为多通道情况，true需要one-hot化
        true_one_hot = F.one_hot(true, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
        bce = self.bce_loss(logits, true_one_hot)
        
        # 加权损失
        return self.dice_weight * dice + self.bce_weight * bce