# 从原始数据中生成国赛训练集

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2


def concat_duijiao(img1, img2):
    half_width = 100
    half_height = 100
    width = 200
    height = 200
    top_left = img1.crop((0, 0, half_width, half_height))
    bottom_right = img1.crop((half_width, half_height, width, height))
    top_right = img2.crop((half_width, 0, width, half_height))
    bottom_left = img2.crop((0, half_height, half_width, height))
    new_image = Image.new("RGB", (width, height))

    # 粘贴裁剪后的区域到新图像
    new_image.paste(top_left, (0, 0))
    new_image.paste(top_right, (half_width, 0))
    new_image.paste(bottom_left, (0, half_height))
    new_image.paste(bottom_right, (half_width, half_height))
    return new_image

def concat_shangxia(img1, img2):
    half_width = 100
    half_height = 100
    width = 200
    height = 200
    new_image = Image.new("RGB", (width, height))
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (0, half_height))
    return new_image

def concat_zuoyou(img1, img2):
    half_width = 100
    half_height = 100
    width = 200
    height = 200
    new_image = Image.new("RGB", (width, height))
    new_image.paste(img1, (0, 0))
    new_image.paste(img2, (half_width, 0))
    return new_image

def concat_images(img1_name, prifix='0'):
    img1 = Image.open(train_ori_dir + img1_name + impost['f'])
    img2_name = random.choice(trainmeta['id'])
    img2 = Image.open(train_ori_dir + img2_name + impost['f'])
    img1_gt = Image.open(train_ori_gt_dir + img1_name + impost['l'])
    img2_gt = Image.open(train_ori_gt_dir + img2_name + impost['l'])

    new_image = concat_duijiao(img1, img2)
    new_image_gt = concat_duijiao(img1_gt, img2_gt)

    new_image.save(train_dir + prifix + img1_name + '.jpg')
    new_image_gt.save(train_gt_dir + prifix + img1_name + '.png')

def concat_images_double(img1_name, prifix1='1', prifix2='2'):
    img1 = Image.open(train_ori_dir + img1_name + impost['f'])
    im2_name = random.choice(trainmeta['id'])
    img2 = Image.open(train_ori_dir + im2_name + impost['f'])
    img1_gt = Image.open(train_ori_gt_dir + img1_name + impost['l'])
    img2_gt = Image.open(train_ori_gt_dir + im2_name + impost['l'])

    # 上下拼接
    new_image = concat_shangxia(img1, img2)
    new_image_gt = concat_shangxia(img1_gt, img2_gt)
    new_image.save(train_dir + prifix1 + img1_name + '.jpg')
    new_image_gt.save(train_gt_dir + prifix1 + img1_name + '.png')
    
    # 左右拼接
    new_image = concat_zuoyou(img1, img2)
    new_image_gt = concat_zuoyou(img1_gt, img2_gt)
    new_image.save(train_dir + prifix2 + img1_name + '.jpg')
    new_image_gt.save(train_gt_dir + prifix2 + img1_name + '.png')


def overlay(img_name, prifix='5'):
    background = cv2.imread(f'data/images/training_ori/{img_name}.jpg', cv2.IMREAD_GRAYSCALE)
    background_gt = cv2.imread(f'data/annotations/training_ori/{img_name}.png', cv2.IMREAD_GRAYSCALE)
    random_img = random.choice(trainmeta['id'])
    overlay = cv2.imread(f'data/images/training_ori/{random_img}.jpg', cv2.IMREAD_GRAYSCALE)
    overlay_gt = cv2.imread(f'data/annotations/training_ori/{random_img}.png', cv2.IMREAD_GRAYSCALE)

    # 调整小图像的大小，若需要可以指定更小尺寸
    overlay_p1 = 0, 0
    overlay_p1 = overlay_p1[0], overlay_p1[1], overlay_p1[0] + random.randint(80, 100), overlay_p1[1] + random.randint(80, 100)
    overlay_p2 = random.randint(30, 80), random.randint(30, 80)
    overlay_p2 = overlay_p2[0], overlay_p2[1], overlay_p2[0] + random.randint(20, 100), overlay_p2[1] + random.randint(20, 100)
    positions = [overlay_p1, overlay_p2]

    # 创建一个彩色版本的背景用于合并显示
    background_color = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    background_gt = cv2.cvtColor(background_gt, cv2.COLOR_GRAY2BGR)

    # 不规则拼接：循环每个位置，将未缩放的小图像叠加到背景图上
    for pos in positions:
        start_x, start_y, end_x, end_y = pos

        # 计算叠加区域的边界，确保 overlay 不超出背景图像的边界

        # 对 overlay 进行裁剪以匹配 ROI 的尺寸
        overlay_cropped = overlay[start_y:end_y, start_x:end_x]
        overlay_color = cv2.cvtColor(overlay_cropped, cv2.COLOR_GRAY2BGR)
        overlay_gt_cropped = overlay_gt[start_y:end_y, start_x:end_x]
        overlay_gt_color = cv2.cvtColor(overlay_gt_cropped, cv2.COLOR_GRAY2BGR)

        # 提取背景中的 ROI
        roi = background_color[start_y:end_y, start_x:end_x]
        roi_gt = background_gt[start_y:end_y, start_x:end_x]

        # 创建遮罩，将非零区域的值设为 255
        mask = np.ones_like(overlay_cropped, dtype=np.uint8) * 255
        mask_gt = np.ones_like(overlay_gt_cropped, dtype=np.uint8) * 255

        # 使用遮罩将 overlay 不规则地叠加到背景图上
        np.copyto(roi, overlay_color, where=(mask[..., None] == 255))
        np.copyto(roi_gt, overlay_gt_color, where=(mask_gt[..., None] == 255))
    background_color = Image.fromarray(cv2.cvtColor(background_color, cv2.COLOR_BGR2RGB))
    background_color.save('data/images/training/' + prifix + f'{img_name}.jpg')
    background_gt = Image.fromarray(background_gt[:, :, 0])
    background_gt.save('data/annotations/training/' + prifix + f'{img_name}.png')


def add_salt_pepper_noise(img, pepper_salt_prob=0.039, salt_vs_pepper=[0.432, 8.8]):
    """
    向图像添加椒盐噪声。
    
    参数:
    - img (Image): 输入图像。
    - salt_prob (float): 椒盐噪声中盐的概率。
    - pepper_prob (float): 椒盐噪声中椒的概率。

    返回:
    - Image: 添加椒盐噪声后的图像。
    """
    img_array = np.array(img)
    total_pixels = img_array.size

    total_prob = random.uniform(pepper_salt_prob - 0.01, 0.15)
    salt_vs_pepper = random.uniform(salt_vs_pepper[0] + 0.01, salt_vs_pepper[1] - 0.01)
    salt_vs_pepper = np.array([salt_vs_pepper, 1.0])
    salt_vs_pepper = salt_vs_pepper / np.sum(salt_vs_pepper)
    salt_prob = total_prob * salt_vs_pepper[0]
    pepper_prob = total_prob * salt_vs_pepper[1]

    # 添加盐噪声
    num_salt = max(int(salt_prob * total_pixels) - np.sum(img_array == 255), 0)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in img_array.shape]
    img_array[salt_coords[0], salt_coords[1]] = 255    # 设置为白色（盐）

    # 添加椒噪声
    num_pepper = max(int(pepper_prob * total_pixels) - np.sum(img_array == 0), 0)
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in img_array.shape]
    img_array[pepper_coords[0], pepper_coords[1]] = 0  # 设置为黑色（椒）

    return Image.fromarray(img_array)

def images_add_noise(img_name, prifix='3'):
    img1 = Image.open(train_ori_dir + img_name + impost['f'])
    img1_gt = Image.open(train_ori_gt_dir + img_name + impost['l'])

    new_image = add_salt_pepper_noise(img1)
    new_image.save(train_dir + prifix + img_name + '.jpg')
    img1_gt.save(train_gt_dir + prifix + img_name + '.png')


def original(img_name, prifix='4'):
    img1 = Image.open(train_ori_dir + img_name + impost['f'])
    img1_gt = Image.open(train_ori_gt_dir + img_name + impost['l'])

    img1.save(train_dir + prifix + img_name + '.jpg')
    img1_gt.save(train_gt_dir + prifix + img_name + '.png')


def mixup(img_name, prefix='6', alpha=0.4):
    # 读取原始图像和标签
    img1 = Image.open(train_ori_dir + img_name + impost['f'])
    img1_gt = Image.open(train_ori_gt_dir + img_name + impost['l'])

    img_name2 = random.choice(trainmeta['id'])
    img2 = Image.open(train_ori_dir + img_name2 + impost['f'])
    img2_gt = Image.open(train_ori_gt_dir + img_name2 + impost['l'])

    # 将图像和标签转换为 NumPy 数组
    img1_array = np.array(img1)
    img2_array = np.array(img2)
    img1_gt_array = np.array(img1_gt)
    img2_gt_array = np.array(img2_gt)

    # 从 Beta 分布中采样 λ
    lam = np.random.beta(alpha, alpha)

    # 生成混合图像和混合标签
    mixed_img_array = (lam * img1_array + (1 - lam) * img2_array).astype(np.uint8)
    mixed_gt_array = (lam * img1_gt_array + (1 - lam) * img2_gt_array).astype(np.uint8)

    # 将混合后的数据转换为图像
    mixed_img = Image.fromarray(mixed_img_array)
    mixed_gt = Image.fromarray(mixed_gt_array)

    # 保存混合后的图像和标签
    mixed_img.save(train_dir + prefix + img_name + img_name2 + '.jpg')
    mixed_gt.save(train_gt_dir + prefix + img_name + img_name2 + '.png')


if __name__ == "__main__":
    # 读取数据，配置文本
    train_ori_dir = 'data/images/training_ori/'
    train_dir = 'data/images/training/'
    test_dir = 'data/images/test/'

    impost = {'f': '.jpg', 'l': '.png'}
    trainfiles = [_.split('.')[0] for _ in os.listdir(train_ori_dir)]
    testfiles = [_.split('.')[0] for _ in os.listdir(test_dir)]
    trainmeta = pd.DataFrame({'id': trainfiles}).sort_values('id').reset_index(drop=True)
    testmeta = pd.DataFrame({'id': testfiles}).sort_values('id').reset_index(drop=True)
    print("原始训练集: ", trainmeta.shape)
    print("测试集: ", testmeta.shape)

    train_ori_gt_dir = 'data/annotations/training_ori/'
    train_gt_dir = 'data/annotations/training/'


    print('生成对角 0')
    trainmeta['id'].apply(lambda x : concat_images(x, '0'))
    print('生成上下左右 1 2')
    trainmeta['id'].apply(lambda x : concat_images_double(x, '1', '2'))
    print('生成噪声图 3')
    trainmeta['id'].apply(lambda x : images_add_noise(x, '3'))
    print('生成不规则图 5')
    trainmeta['id'].apply(lambda x : overlay(x, '5'))
    print('生成原图 4')
    trainmeta['id'].apply(lambda x : original(x, '4'))
    print('生成mixup 6')
    trainmeta['id'].apply(lambda x : mixup(x, '6'))

    new_csv = os.listdir('data/images/training')
    new_csv = [x.split('.')[0] for x in new_csv]
    train_new_meta = pd.DataFrame({'id':new_csv})
    train_new_meta.to_csv('stats/train-meta.csv', index=False)

    print("新生成的数据集样本量", train_new_meta.shape[0])


