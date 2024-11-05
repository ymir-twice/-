import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from torchvision import transforms
import numpy as np
import torch


trainmeta = pd.read_csv('stats/train-meta.csv', dtype=object)
testmeta = pd.read_csv('stats/test-meta.csv', dtype=object)

import matplotlib.patches as mpatches
color_map = {
    0: [0, 0, 0],      # Background (黑色)
    1: [255, 165, 0],     # Inclusions (浅橙色)
    2: [30, 144, 255],    # Patches (天蓝色)
    3: [50, 205, 50],     # Scratches (亮绿色)
}
legend_patches = [
    mpatches.Patch(color=np.array(color_map[0])/255, label='Background(0)'),
    mpatches.Patch(color=np.array(color_map[1])/255, label='Inclusions(1)'),
    mpatches.Patch(color=np.array(color_map[2])/255, label='Patches(2)'),
    mpatches.Patch(color=np.array(color_map[3])/255, label='Scratches(3)'),
]
def to_rgb_im(mask):
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx, color in color_map.items():
        rgb_image[mask == class_idx] = color
    return rgb_image

image_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

def plot_result(model, imid:str, train:bool, axes, if_aug=False):
    """
        总共绘制3张图, 第一张是原图, 第二张是预测的掩码图, 第三张是GT
        args:
            imid: 希望可视化的图片的id
            train: True: 可视化训练集图片  False: 可视化测试集图片
    """
    src = "data/images/training/" if train else "data/images/test/"
    imdata = Image.open(src + imid + '.jpg')
    src = "data/annotations/training/" if train else "data/annotations/test/"
    mask = cv2.imread(src + imid + '.png', cv2.IMREAD_GRAYSCALE)
    inputs = image_transform(imdata).cuda().unsqueeze(0)
    with torch.no_grad():
        model.eval()
        pred = model(inputs)
        if if_aug:
            pred = pred[0]
        pred = torch.argmax(pred, dim=1).cpu()[0]

    # 颜色转换
    mask, pred = to_rgb_im(mask), to_rgb_im(pred)

    axes[0].imshow(imdata)
    axes[0].axis('off')
    axes[0].set_title(('train' if train else 'test') + f': {imid} original')
    axes[1].imshow(mask) 
    axes[1].axis('off') 
    axes[1].set_title(('train' if train else 'test') + f': {imid} Gground Truth') 
    axes[2].imshow(pred) 
    axes[2].axis('off') 
    axes[2].set_title(('train' if train else 'test') + f': {imid} Predicted') 