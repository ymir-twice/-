"""
    predict.py
        此文件用于根据模型配置文件，生成分割结果
        执行此文件后，可以直接使用对应的模型获取预测成果，并计算FPS
        FPS的计算不应该包括数据的读取、预处理和结果的存入
    使用方式: 
        将待测模型的网络定义放入 models/networks.py 中
        在 vscode 下, ctrl + shift + p, 执行 predict task 即可运行.
        如果希望自行设置相关目录, 请通过 argparser 的help操作
"""
import torch
import pandas as pd
import argparse
import os
import shutil  # rm -r
from PIL import Image
import numpy as np
import torch.utils
from dataset import myTestDataset
import time
from tqdm import tqdm
from models.networks import *
import importlib
def load_all_as_globals(module_name):
    # 动态导入模块
    mod = importlib.import_module(module_name)

    # 获取模块中的所有属性名称
    module_items = dir(mod)

    # 遍历模块的所有属性
    for item_name in module_items:
        # 获取属性
        item = getattr(mod, item_name)
        globals()[item_name] = item


def get_dataLoader():
    testset = myTestDataset(idx_path=args.data_stats, 
                        img_dir=args.data_dir, imshape=args.input_shape, if_aug=args.if_aug)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
    return testloader

def main(model_path, result_path):
    # 目录工作
    model_name = input("请输入待测试模型名称(以 .pth 结尾): ")
    #load_all_as_globals(f"model-src.{model_name.split('.')[0]}")
    model_path = model_path + '/' + model_name
    assert os.path.isfile(model_path), f"{model_path} does not exist"
    print(model_name)
    result_path = result_path + f"_{model_name.split('.')[0]}/"
    print(result_path)
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    else:
        shutil.rmtree(result_path)
        os.mkdir(result_path)

    # 载入模型
    print(model_path)
    model = torch.load(model_path, weights_only=False, map_location=torch.device("cuda:0")).cuda()
    model.eval()
    torch.no_grad()

    # 装载数据
    testloader = get_dataLoader()

    # 准备计时器
    elapsed_time = np.zeros(len(testmeta))

    # 开始计算
    pbar = tqdm(testloader)
    for i, (inputs, img_name)in enumerate(pbar):
        inputs = inputs.cuda()
        img_name = img_name[0]

        begin = time.perf_counter()
        out = model(inputs)
        out = torch.argmax(out, dim=1)  # 获得类别标签
        end = time.perf_counter()
        elapsed_time[i] = (end - begin)
        out = np.array(out.cpu()).squeeze(0)

        #print(result_path + img_name[0])
        np.save(result_path + "prediction_" + img_name, out)

    total_elapsed_time = sum(elapsed_time)
    return total_elapsed_time

parser = argparse.ArgumentParser(description='predict')
parser.add_argument('--model', type=str,
                    help='path to model.pth')
parser.add_argument('--save_dir', type=str,
                    help='path to folder containing predict result of model')
parser.add_argument('--input_shape', type=int,
                    help='n for n × n')
parser.add_argument('--if_aug', type=int,
                    help='if use data autmentation')
parser.add_argument('--data_dir', type=str,
                    help='path to data folder default is data/images/test/', default='data/images/test/')
parser.add_argument('--data_stats', type=str,
                    help='path to data stats file default is stats/test-meta.csv', default='stats/test-meta.csv')


if __name__ == "__main__":

    args = parser.parse_args()
    testmeta = pd.read_csv(args.data_stats,dtype=object)
    total_calc_time = main(args.model, args.save_dir)
    fps = len(testmeta) / total_calc_time
    print(f"模型的fps为: {fps}")