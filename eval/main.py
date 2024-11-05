import os
import torch
import math
import numpy as np
from models.networks import * # 导入模型

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

def calculate_iou(pred, gt, num_classes, mode='original'):
    ious = []
    for cls in range(1, 1 + num_classes):
        pred_cls = (pred == cls)
        gt_cls = (gt == cls)

        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()

        if mode == "original":
            if union == 0:
                iou = 1 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            ious.append(iou)
        elif mode == "no-count-zero":
            if union == 0:
                iou = -1 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            ious.append(iou)
        elif mode == 'whole-mean':
            ious.append(intersection)
            ious.append(union)

    return ious

def get_matrix_mean(pred_dir, gt_dir, num_classes, mod="whole-mean"):
    """
        此方法旨在使用一种基于的混淆矩阵的的方法计算平均IoU  ,  但是现在已经不用它了
    """
    if mod == 'original-matrix':
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

        assert len(pred_files) == len(gt_files), "Prediction and GT files count do not match"

        mIoUs = np.zeros(len(pred_files))

        for idx, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
            pred = np.load(os.path.join(pred_dir, pred_file))
            gt = np.load(os.path.join(gt_dir, gt_file))

            mIoU_matrix = np.zeros((4, 4))
            for i in range(num_classes + 1):
                for j in range(num_classes + 1):
                    pred_cls = (pred == i)
                    gt_cls = (gt == j)
                    mIoU_matrix[i, j] = np.logical_and(pred_cls, gt_cls).sum()
            mIoUs[idx] = np.diagonal(mIoU_matrix).sum() / (200 * 200)
        return np.mean(mIoUs)
    else:
        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
        gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

        assert len(pred_files) == len(gt_files), "Prediction and GT files count do not match"

        mIoU_matrix = np.zeros((4, 4))
        for idx, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
            pred = np.load(os.path.join(pred_dir, pred_file))
            gt = np.load(os.path.join(gt_dir, gt_file))

            for i in range(num_classes + 1):
                for j in range(num_classes + 1):
                    pred_cls = (pred == i)
                    gt_cls = (gt == j)
                    mIoU_matrix[i, j] += np.logical_and(pred_cls, gt_cls).sum()
        mIoU = np.diagonal(mIoU_matrix).sum() / mIoU_matrix.sum().sum()
        return mIoU


def seg(pred_dir, gt_dir, num_classes, mode='whole-mean'):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.npy')])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.npy')])

    assert len(pred_files) == len(gt_files), f"Prediction and GT files count do not match {len(pred_files)} {len(gt_files)} {pred_dir}"

    all_ious = np.zeros((len(pred_files), num_classes))
    if mode == "whole-mean":
        all_ious = [None for i in pred_files]

    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):
        pred = np.load(os.path.join(pred_dir, pred_file))
        gt = np.load(os.path.join(gt_dir, gt_file))

        ious = calculate_iou(pred, gt, num_classes, mode)
        all_ious[i] = ious

    mean_ious = np.zeros(3)
    if mode == 'original':
        mean_ious = np.mean(all_ious, axis=0)
    elif mode == 'no-count-zero':
        e1_ious = all_ious.T[0]
        e2_ious = all_ious.T[1]
        e3_ious = all_ious.T[2]
        mean_ious[0] = np.mean(e1_ious[e1_ious != -1])
        mean_ious[1] = np.mean(e2_ious[e2_ious != -1])
        mean_ious[2] = np.mean(e3_ious[e3_ious != -1])
    elif mode == 'whole-mean':
        inter = np.zeros(3)
        union = np.zeros(3)
        for i in range(len(all_ious)):
            inter[0] += all_ious[i][0]
            inter[1] += all_ious[i][2]
            inter[2] += all_ious[i][4]
            union[0] += all_ious[i][1]
            union[1] += all_ious[i][3]
            union[2] += all_ious[i][5]
        for i in range(3):
            mean_ious[i] = inter[i] / union[i]

    for cls, iou in enumerate(mean_ious):
        print(f"Class {cls + 1} Mean IoU: {iou}")
    return mean_ious

import argparse
parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--model', type=str,
                    help='path to model.pth')
parser.add_argument('--save_dir', type=str,
                    help='path to folder containing predict result of model')
parser.add_argument('--mode', type=str,
                    help='method to calc IoU: [original、no-count-zero、whole-mean]')

if __name__ == "__main__":
    # 训练好的模型的路径
    pred_model_name = input('请输入self模型名(不加.pth): ')
    model_path = f"models/{pred_model_name}.pth"

    score = 0
    ###计算模型参数分数###
    total_params = count_model_parameters(model_path)
    norm_params = total_params / 1_000_000
    print(f"self模型的参数总量为: {norm_params} M.")

    score_para = 0
    if norm_params > 5:
        score_para = 10
    else:
        if norm_params < 1:
            score_para = 50
        else:
            score_para = 60 - 10 * norm_params
    print(f"模型参数的分数为{score_para}")
    score += score_para
    ###################

    ####计算class IoU分数####
    pred_dir = f'eval/test_pred_{pred_model_name}/'
    gt_dir = 'eval/test_ground_truths/'

    num_classes = 3  # 异常类型数

    print(f"self-net {pred_model_name} IoU: ")
    pre_IoU = seg(pred_dir, gt_dir, num_classes)
    print(f"mIoU: {sum(pre_IoU) / len(pre_IoU)}")

    for pre in pre_IoU:
        score_class = 100 * pre
        print(f"分数：{score_class}")
        score += score_class
    print(f"最终分数：{score}")