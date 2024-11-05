from PIL import Image, ImageEnhance
import numpy as np
from utils.args import DA_ARGS
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


# 以下来自：https://github.com/ZM-J/ET-Net/blob/master/datasets/data_augmentation.py
# 下面的 img 、label、edge 均要求是 PIL 的 Image 类
def data_augmentation(img, label, edge=None):
    r_img, r_label, r_edge = img, label, edge
    r_img, r_label, r_edge = _random_mirror(r_img, r_label, r_edge)
    #r_img, r_label, r_edge = _random_scale(r_img, r_label, r_edge)
    r_img, r_label, r_edge = _random_rotation(r_img, r_label, r_edge, is_reg=True)
    # TODO: 可以在这里添加其他的数据增强。 基本处理逻辑是： 输入是个 Image 对象，中间可以转化为 np.array 便于使用opencv或者其他图像处理库。 最后的返回值，也应该是 Image 对象， 因为torchvision.tranforms.ToTensor 以及 resize 要求只能对 Image 对象处理
    #r_img = 
    # 添加高斯滤波
    #r_img = usm_sharpen(r_img)
    r_img = gaussian_blur(r_img)  # 调用高斯滤波函数
    r_img = CLAHE(r_img)
    if r_edge:
        return r_img, r_label, r_edge
    else:
        return r_img, r_label
def usm_sharpen(image, radius=5, threshold=10, amount=150):
    # 将 PIL Image 对象转换为 np.array
    img_array = np.array(image)
    # 使用 OpenCV 的 GaussianBlur 进行模糊处理
    blurred = cv2.GaussianBlur(img_array, (radius, radius), 0)
    # 计算原图像和模糊图像的差值
    mask = img_array - blurred
    # 对差值进行增强处理（锐化）
    sharpened = img_array + amount * mask / 100
    # 使用阈值筛选变化
    low_contrast_mask = np.abs(mask) < threshold
    np.copyto(sharpened, img_array, where=low_contrast_mask)
    # 确保像素值合法
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    # 将结果转换回 PIL Image 对象
    return Image.fromarray(sharpened)

def gaussian_blur(image):
    img_array = np.array(image)  # 转换为 np.array
    if np.random.random() < DA_ARGS['p_random_gaussian_blur']:
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)  # 应用高斯滤波
    return Image.fromarray(img_array)  # 转换回 Image 对象

def _random_mirror(img, label, edge=None):
    r_img, r_label, r_edge = img, label, edge
    if np.random.random() < DA_ARGS['p_random_mirror']:
        r_img = r_img.transpose(Image.FLIP_LEFT_RIGHT)
        r_label = r_label.transpose(Image.FLIP_LEFT_RIGHT)
        if edge:
            r_edge = r_edge.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.random() < DA_ARGS['p_random_mirror']:
        r_img = r_img.transpose(Image.FLIP_TOP_BOTTOM)
        r_label = r_label.transpose(Image.FLIP_TOP_BOTTOM)
        if edge:
            r_edge = r_edge.transpose(Image.FLIP_TOP_BOTTOM)
    return r_img, r_label, r_edge

def _random_scale(img, label, edge=None):
    r_img, r_label, r_edge = img, label, edge
    if np.random.random() < DA_ARGS['p_random_scale']:
        z = np.random.uniform(0.8, 1.2) # 0.5 ~ 2
        width, height = img.size
        to_width, to_height = int(z*width), int(z*height)
        r_img = img.resize((to_width, to_height), Image.LANCZOS)
        r_label = label.resize((to_width, to_height), Image.LANCZOS)
        if edge:
            r_edge = edge.resize((to_width, to_height), Image.LANCZOS)
    return r_img, r_label, r_edge

def _random_rotation(img, label, edge=None, is_reg=False):
    r_img, r_label, r_edge = img, label, edge
    if np.random.random() < DA_ARGS['p_random_rotation']:
        theta = np.random.uniform(-10, 10)
        if is_reg:
            theta = np.random.choice([90, 180, 270])
        r_img = img.rotate(theta)
        r_label = label.rotate(theta)
        if r_edge:
            r_edge = edge.rotate(theta)
    return r_img, r_label, r_edge

def _random_color_jitter(img):
    r_img = img
    transform_tuples = [
        (ImageEnhance.Brightness, 0.1026),
        (ImageEnhance.Contrast, 0.0935),
        (ImageEnhance.Sharpness, 0.8386),
        (ImageEnhance.Color, 0.1592)
    ]
    if np.random.random() < DA_ARGS['p_random_jitter']:
        rand_num = np.random.uniform(0, 1, len(transform_tuples))
        for i, (transformer, alpha) in enumerate(transform_tuples):
            r = alpha * (rand_num[i]*2.0 - 1.0) + 1   # r in [1-alpha, 1+alpha)
            r_img = transformer(r_img).enhance(r)
    return r_img

def _random_crop(img, label, edge=None):
    r_img, r_label, r_edge = img, label, edge
    width, height = img.size
    r_width, r_height = DA_ARGS['crop_size'], DA_ARGS['crop_size'] # 512, 512
    zx, zy = np.random.randint(0, width - r_width), np.random.randint(0, height - r_height)
    r_img = r_img.crop((zx, zy, zx+r_width, zy+r_height))
    r_label = r_label.crop((zx, zy, zx+r_width, zy+r_height))
    if edge:
        r_edge = r_edge.crop((zx, zy, zx+r_width, zy+r_height))
    return r_img, r_label, r_edge

def CLAHE(img):
    r_img = np.array(img)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2LAB)
    r_img_l = r_img[:, :, 0] # Convert to range [0,1]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    r_img_l = clahe.apply(r_img_l)
    r_img[:, :, 0] = r_img_l
    r_img = cv2.cvtColor(r_img, cv2.COLOR_LAB2RGB)
    return Image.fromarray(r_img)

# 初始化 HOG 描述器
from skimage import feature as ft
def HOG(img):
    r_img = np.array(img)
    gray_image = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)
    features = ft.hog(img,orientations=6,pixels_per_cell=[20,20],cells_per_block=[2,2],visualize=True)
    return features