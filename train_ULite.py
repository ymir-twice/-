import torch
import torch.nn as nn
import torch.nn.functional as F
import random,os
import numpy as np
def set_random_seed(seed: int) -> None:
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed = 37
set_random_seed(seed)
class AxialDW(nn.Module):
    def __init__(self, dim, mixer_kernel, dilation = 1):
        super().__init__()
        h, w = mixer_kernel
        self.dw_h = nn.Conv2d(dim, dim, kernel_size=(h, 1), padding='same', groups = dim, dilation = dilation)
        self.dw_w = nn.Conv2d(dim, dim, kernel_size=(1, w), padding='same', groups = dim, dilation = dilation)

    def forward(self, x):
        x = x + self.dw_h(x) + self.dw_w(x)
        return x

class EncoderBlock(nn.Module):
    """Encoding then downsampling"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7)):
        super().__init__()
        self.dw = AxialDW(in_c, mixer_kernel = (7, 7))
        self.bn = nn.BatchNorm2d(in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.down = nn.MaxPool2d((2,2))
        self.act = nn.GELU()

    def forward(self, x):
        skip = self.bn(self.dw(x))
        x = self.act(self.down(self.pw(skip)))
        return x, skip

class DecoderBlock(nn.Module):
    """Upsampling then decoding"""
    def __init__(self, in_c, out_c, mixer_kernel = (7, 7), size = False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.size = size
            # self.up = nn.ConvTranspose2d(
            #             in_channels=256,
            #             out_channels=256,
            #             kernel_size=5,  
            #             stride=2, 
            #             padding=1  
            #         

        self.pw = nn.Conv2d(in_c + out_c, out_c,kernel_size=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.dw = AxialDW(out_c, mixer_kernel = (7, 7))
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(out_c, out_c, kernel_size=1)

    def forward(self, x, skip):
        if(self.size):
            x = F.interpolate(x, size=(25, 25), mode='bilinear', align_corners=True)
        else:
            x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        # x = F.interpolate(x, size=(25, 25), mode='bilinear', align_corners=True)
        x = self.act(self.pw2(self.dw(self.bn(self.pw(x)))))

        return x

class BottleNeckBlock(nn.Module):
    """Axial dilated DW convolution"""
    def __init__(self, dim):
        super().__init__()

        gc = dim//4
        self.pw1 = nn.Conv2d(dim, gc, kernel_size=1)
        self.dw1 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 1)
        self.dw2 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 2)
        self.dw3 = AxialDW(gc, mixer_kernel = (3, 3), dilation = 3)

        self.bn = nn.BatchNorm2d(4*gc)
        self.pw2 = nn.Conv2d(4*gc, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.pw1(x)
        x = torch.cat([x, self.dw1(x), self.dw2(x), self.dw3(x)], 1)
        x = self.act(self.pw2(self.bn(x)))
        return x

class self_net(nn.Module):
    def __init__(self):
        super().__init__()

        """Encoder"""
        self.conv_in = nn.Conv2d(3, 16, kernel_size=7, padding='same')
        self.e1 = EncoderBlock(16, 32)
        self.e2 = EncoderBlock(32, 64)
        self.e3 = EncoderBlock(64, 128)
        self.e4 = EncoderBlock(128, 256)
        self.e5 = EncoderBlock(256, 512)

        """Bottle Neck"""
        self.b5 = BottleNeckBlock(512)

        """Decoder"""
        self.d5 = DecoderBlock(512, 256)
        self.d4 = DecoderBlock(256, 128, size = True)
        self.d3 = DecoderBlock(128, 64)
        self.d2 = DecoderBlock(64, 32)
        self.d1 = DecoderBlock(32, 16)
        self.conv_out = nn.Conv2d(16, 4, kernel_size=1)

    def forward(self, x):
        """Encoder"""
        x = self.conv_in(x)
        x, skip1 = self.e1(x)
        x, skip2 = self.e2(x)
        x, skip3 = self.e3(x)
        x, skip4 = self.e4(x)
        x, skip5 = self.e5(x)

        """BottleNeck"""
        x = self.b5(x)          # 512 6 6
        """Decoder"""
        x = self.d5(x, skip5)   # 256 12 12
        x = self.d4(x, skip4)
        x = self.d3(x, skip3)
        x = self.d2(x, skip2)
        x = self.d1(x, skip1)
        x = self.conv_out(x)
        return x
    
from dataset import *
from torchvision import transforms
import numpy as np

# 定义图像和掩码的预处理
image_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long)) 
])

# 创建数据集对象
trainset = myDataset(idx_path='stats/train-meta.csv', img_dir='data/images/training/', mask_dir='data/annotations/training/', transform=image_transform, mask_transform=mask_transform)

batch_size = 32
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, batch_size=batch_size)

device = torch.device('cuda:0')
model = torch.load('models/ULite19.pth').to(device)

lr = 0.001
betas = (0.9, 0.999)
weight_decay = 5e-3
#optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
optimizer = torch.optim.Adam(model.parameters(), lr, betas)

from utils.loss_function.dice_loss import DiceLoss
criterion = DiceLoss(weights=[0.68,1.5,0.81,1])#10 20 12 17

from tqdm import tqdm
from utils.lr_scheduler import WarmupMultiStepLR, WarmupCosineLR

num_epochs = 1000
total_loss = []
epoch_loss = 0
milestones = [50,100,150,200,250]
for epoch in range(1, num_epochs + 1):
    pbar = tqdm(trainloader, colour='#C0FF20')
    total_batchs = len(trainloader)
    pbar.set_description(f'{epoch}/{num_epochs}, total loss {epoch_loss:.5f}')
    scheduler = WarmupCosineLR(optimizer, T_max=num_epochs + 1, last_epoch=epoch - 2, warmup_factor=1.0 / 3, warmup_iters=200)    # 有热身的cos loss 
    #scheduler = WarmupMultiStepLR(optimizer,milestones=milestones,gamma=0.7,warmup_factor=1.0 / 3,warmup_iters=300)
    epoch_loss = 0

    for i, (inputs, gts) in enumerate(pbar):
        inputs, gts = inputs.to(device), gts.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, gts)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
    
    if i == 200 or i == 500 or i == 800:
        torch.save(model, "ULite_tmp.pth")
    scheduler.step()
    total_loss.append(epoch_loss)

torch.save(model, 'ULite_cos1.pth')
