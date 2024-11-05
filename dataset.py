import os
import pandas as pd
from PIL import Image
import torch.utils
from utils.data_augmentation import *


class myDataset(torch.utils.data.Dataset):
    """
        此类用于加载训练时的数据集，不包括边label
    """
    def __init__(self, idx_path, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = list(pd.read_csv(idx_path, dtype=object)['id'])
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_enhance = data_augmentation

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image, mask = self.data_enhance(image, mask)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
    
class myBoundaryDataset(torch.utils.data.Dataset):
    """
        此类用于加载训练时的数据集，包括边label
    """
    def __init__(self, idx_path, img_dir, mask_dir, maskb_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.maskb_dir = maskb_dir
        self.img_names = list(pd.read_csv(idx_path, dtype=object)['id'])
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_enhance = data_augmentation()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')
        mask_path = os.path.join(self.mask_dir, img_name + '.png')
        maskb_path = os.path.join(self.maskb_dir, img_name + '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        maskb = Image.open(maskb_path).convert('L')

        image, mask, maskb = self.data_enhance(image, mask, maskb)

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            maskb = self.mask_transform(maskb)

        return image, mask, maskb
    

class myTestDataset(torch.utils.data.Dataset):
    """
        此类用于加载测试时所需的数据 ， 跟上边的主要差别是，使用的数据增强的技术不容
    """
    def __init__(self, idx_path, img_dir, imshape=200, if_aug=False):
        self.img_dir = img_dir
        self.img_names = list(pd.read_csv(idx_path, dtype=object)['id'])
        self.transform = transforms.Compose([
            transforms.Resize((imshape, imshape)),
            transforms.ToTensor()
        ])
        self.if_aug = if_aug

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name + '.jpg')

        image = Image.open(img_path).convert('RGB')
        if self.if_aug:
            image = CLAHE(image)

        image = self.transform(image)

        return image, img_name  # 这里返回模型的名字即可