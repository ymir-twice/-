from PIL import Image, ImageEnhance
import numpy as np
from utils.args import DA_ARGS
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


# 下面的 img 、label、edge 均要求是 PIL 的 Image 类
def data_augmentation(img, label, edge=None):
    r_img, r_label = img, label
    r_img, r_label = _random_mirror(r_img, r_label)
    #r_img, r_label = _random_scale(r_img, r_label, r_edge)
    r_img, r_label = _random_rotation(r_img, r_label, is_reg=True)
    # TODO: 可以在这里添加其他的数据增强。 
    # 基本处理逻辑是： 输入是个 Image 对象，中间可以转化为 np.array 
    # 便于使用opencv或者其他图像处理库。 最后的返回值，也应该是 Image 对象， 
    # 因为torchvision.tranforms.ToTensor 以及 resize 要求只能对 Image 对象处理

    r_img = _random_color_jitter(r_img)
    r_img = CLAHE(r_img)

    return r_img, r_label

def add_salt_pepper_noise(image, prob):
    """
    向图像添加椒盐噪声
    :param image: 输入的灰度图像或彩色图像 (numpy 数组)
    :param prob: 椒盐噪声的比例（0到1之间），例如0.01表示1%的像素被设置为椒盐噪声
    :return: 带有椒盐噪声的图像
    """
    output = np.copy(image)
    # 设置椒盐噪声
    salt_pepper_noise = np.random.rand(*image.shape[:2])
    # 设置盐（白色）噪声
    output[salt_pepper_noise < prob / 2] = 255
    # 设置椒（黑色）噪声
    output[salt_pepper_noise > 1 - prob / 2] = 0
    return output


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


def DCT_augmentation(img, block_size=8):
    # 将输入图像转换为NumPy数组并确保是RGB格式
    r_img = np.array(img)
    
    # 将图像转换为灰度图像，因为DCT通常应用于单通道图像
    gray_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
    
    # 获取图像的尺寸
    h, w = gray_img.shape
    
    # 将图像分块并对每块进行DCT变换
    dct_img = np.zeros_like(gray_img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # 提取当前块
            block = gray_img[i:i+block_size, j:j+block_size]
            
            # 对块进行DCT变换
            dct_block = cv2.dct(np.float32(block))
            
            # 对DCT变换后的系数进行增强（可以加噪声或其他处理）
            dct_block += np.random.normal(0, 1, dct_block.shape)  # 增加噪声作为增强
            
            # 将增强后的DCT系数反变换回原空间
            idct_block = cv2.idct(dct_block)
            
            # 存储反变换后的块
            dct_img[i:i+block_size, j:j+block_size] = idct_block

    # 将DCT增强后的图像转换回RGB格式
    enhanced_img = np.stack([dct_img] * 3, axis=-1).astype(np.uint8)
    
    return Image.fromarray(enhanced_img)


def DCT_denoise(img, block_size=8, threshold=10):
    # 将输入图像转换为NumPy数组并确保是RGB格式
    r_img = np.array(img)
    
    # 将图像转换为灰度图像，因为DCT通常应用于单通道图像
    gray_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2GRAY)
    
    # 获取图像的尺寸
    h, w = gray_img.shape
    
    # 将图像分块并对每块进行DCT变换
    dct_img = np.zeros_like(gray_img, dtype=np.float32)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            # 提取当前块
            block = gray_img[i:i+block_size, j:j+block_size]
            
            # 对块进行DCT变换
            dct_block = cv2.dct(np.float32(block))
            
            # 将DCT系数低于阈值的高频部分设为0（降噪）
            dct_block[np.abs(dct_block) < threshold] = 0
            
            # 将降噪后的DCT系数反变换回原空间
            idct_block = cv2.idct(dct_block)
            
            # 存储反变换后的块
            dct_img[i:i+block_size, j:j+block_size] = idct_block

    # 将降噪后的图像转换回RGB格式
    denoised_img = np.stack([dct_img] * 3, axis=-1).astype(np.uint8)
    
    return Image.fromarray(denoised_img)


def CLAHE2(img):
    r_img = np.array(img)
    r_img = cv2.cvtColor(r_img, cv2.COLOR_RGB2LAB)
    r_img_l = r_img[:, :, 0] # Convert to range [0,1]
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    r_img_l = clahe.apply(r_img_l)
    r_img[:, :, 0] = r_img_l
    r_img = cv2.cvtColor(r_img, cv2.COLOR_LAB2RGB)
    return Image.fromarray(r_img)


def gaussian_blur(image):
    img_array = np.array(image)  # 转换为 np.array
    if np.random.random() < DA_ARGS['p_random_gaussian_blur']:
        img_array = cv2.GaussianBlur(img_array, (5, 5), 0)  # 应用高斯滤波
    return Image.fromarray(img_array)  # 转换回 Image 对象


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


def median_filter_denoise(img, kernel_size=3):
    # 将图像转换为灰度
    gray_img = np.array(img.convert("L"))
    
    # 应用中值滤波
    denoised_img = cv2.medianBlur(gray_img, kernel_size)
    
    return Image.fromarray(denoised_img).convert("RGB")


def bilateral_filter_denoise(img, d=9, sigmaColor=75, sigmaSpace=75):
    # 将图像转换为灰度
    
    gray_img = np.array(img.convert("L"))
    
    # 应用双边滤波
    denoised_img = cv2.bilateralFilter(gray_img, d, sigmaColor, sigmaSpace)
    #denoised_img = denoised_img.convert("RGB")
    return Image.fromarray(denoised_img).convert("RGB")







# 以下是一些传统的数据增强方法，可以直接使用
def _random_mirror(img, label):
    r_img, r_label = img, label
    if np.random.random() < DA_ARGS['p_random_mirror']:
        r_img = r_img.transpose(Image.FLIP_LEFT_RIGHT)
        r_label = r_label.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.random() < DA_ARGS['p_random_mirror']:
        r_img = r_img.transpose(Image.FLIP_TOP_BOTTOM)
        r_label = r_label.transpose(Image.FLIP_TOP_BOTTOM)
    return r_img, r_label


def _random_scale(img, label):
    r_img, r_label = img, label
    if np.random.random() < DA_ARGS['p_random_scale']:
        z = np.random.uniform(0.8, 1.2) # 0.5 ~ 2
        width, height = img.size
        to_width, to_height = int(z*width), int(z*height)
        r_img = img.resize((to_width, to_height), Image.LANCZOS)
        r_label = label.resize((to_width, to_height), Image.LANCZOS)
    return r_img, r_label


def _random_rotation(img, label, is_reg=False):
    r_img, r_label = img, label
    if np.random.random() < DA_ARGS['p_random_rotation']:
        theta = np.random.uniform(-10, 10)
        if is_reg:
            theta = np.random.choice([90, 180, 270])
        r_img = img.rotate(theta)
        r_label = label.rotate(theta)
    return r_img, r_label


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


def _random_crop(img, label):
    r_img, r_label = img, label
    width, height = img.size
    r_width, r_height = DA_ARGS['crop_size'], DA_ARGS['crop_size'] # 512, 512
    zx, zy = np.random.randint(0, width - r_width), np.random.randint(0, height - r_height)
    r_img = r_img.crop((zx, zy, zx+r_width, zy+r_height))
    r_label = r_label.crop((zx, zy, zx+r_width, zy+r_height))
    return r_img, r_label