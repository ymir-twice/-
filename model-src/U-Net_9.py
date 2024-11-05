import torch
from torch import nn
import numpy as np

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) # padding后，恰好可以保证每次图片尺寸为偶数，可供下一块maxpool
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.functional.relu

    def forward(self, X):
        X = self.maxpool(X)
        X = self.conv1(X)
        X = self.relu(self.bn1(X))
        X = self.conv2(X)
        X = self.relu(self.bn2(X))
        return X
    

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2) # 这样刚好可以按原形状反卷积回去
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.functional.relu

    def forward(self, X, _X):
        X = self.conv_transpose(X)
        X = torch.cat([X, _X], dim=-3) # -3 是通道数对应的那个维度
        X = self.conv1(X)
        X = self.relu(self.bn1(X))
        X = self.conv2(X)
        X = self.relu(self.bn2(X))
        return X

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.functional.relu
        self.down = nn.Sequential(
            DownBlock(64, 128),
            DownBlock(128, 256),
            DownBlock(256, 512),
            DownBlock(512, 1024)
        )
        self.up = nn.Sequential(
            UpBlock(1024, 512, 3),
            UpBlock(512, 256),
            UpBlock(256, 128), 
            UpBlock(128, 64)
        )
        self.conv3 = nn.Conv2d(64, 4, kernel_size=1)  # 其实这里就等价于一个64 到 1 的全连接

    def forward(self, X):
        X = self.conv1(X)
        X = self.relu(self.bn1(X))
        X = self.conv2(X)
        X = self.relu(self.bn2(X))
        down_steps = []
        for mod in self.down:
            down_steps.append(X)
            X = mod(X)
        for mod in self.up:
            X = mod(X, down_steps.pop())
        X = self.conv3(X)
        return X