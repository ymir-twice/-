import torch
import torch.nn as nn
import torch.nn.functional as F

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

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        
    def forward(self, x):
        se_weight = torch.mean(x, dim=(2, 3), keepdim=True)
        se_weight = torch.relu(self.fc1(se_weight))
        se_weight = torch.sigmoid(self.fc2(se_weight))
        return x * se_weight

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, out_channels=1):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, out_channels, kernel_size, padding=kernel_size // 2)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = torch.sigmoid(self.conv(x))
        return x * x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7, out_channels=1):
        super(CBAM, self).__init__()
        self.channel_attention = SEBlock(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size, out_channels)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class self_net(nn.Module):
    def __init__(self):
        super().__init__()

        """Attention Enhance"""
        self.cbam256 = CBAM(256, 16, 256)
        self.cbam128 = CBAM(128, 16, 128)
        self.cbam64 = CBAM(64, 4, 64)

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
        x = self.cbam256(x)
        x = self.d4(x, skip4)   # 128 25 25
        x = self.cbam128(x)
        x = self.d3(x, skip3)   # 64 50 50
        x = self.cbam64(x)
        x = self.d2(x, skip2)   # 32 100 100
        x = self.d1(x, skip1)   # 16 200 200
        x = self.conv_out(x)
        return x