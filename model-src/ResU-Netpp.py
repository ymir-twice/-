import torch 
from torch import nn

# Squeeze_Excitation模块，适用于通道维度上的自适应权重分配，用于抑制不重要的特征增强关键特征
class Squeeze_Excitation(nn.Module):
    def __init__(self, channel, r=8):  # r为缩放比，控制通道维数的缩减程度
        super().__init__()

        # 自适应平均池化，将输入的H×W的特征图转化为1×1（即全局信息）
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 两个全连接层，分别对通道进行压缩和恢复，并使用Sigmoid激活来生成通道的权重
        self.net = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),   # 压缩通道数
            nn.ReLU(inplace=True),                          # 激活
            nn.Linear(channel // r, channel, bias=False),   # 恢复通道数
            nn.Sigmoid(),                                   # 用Sigmoid激活得到通道权重
        )

    def forward(self, inputs):
        batchsize, channel, _, _ = inputs.shape         # 获取输入的batch size和通道数
        x = self.pool(inputs).view(batchsize, channel)  # 通过平均池化并展平成(b, c)
        x = self.net(x).view(batchsize, channel, 1, 1)  # 通过全连接层处理并reshape回到(b, c, 1, 1)
        x = inputs * x                                  # 输入与权重相乘，调整特征图
        return x


# Stem_Block模块，类似ResNet中的stem block，用于对输入进行初步特征提取和压缩
class Stem_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        # 第一个卷积分支，主要负责初步特征提取
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1),  
            nn.BatchNorm2d(out_c),  
            nn.ReLU(),  
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),  
        )

        # 第二个卷积分支，用于调整输入的通道数以匹配特征分支输出
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0), 
            nn.BatchNorm2d(out_c),  
        )

        # 使用Squeeze_Excitation模块进行通道注意力调整
        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        X = self.c1(inputs)                 # 特征提取
        inputjusted = self.c2(inputs)       # 输入通道匹配
        out = self.attn(X + inputjusted)    # 特征和输入相加（跳跃连接）后进行注意力调整
        return out


# ResNet_Block模块，类似ResNet中的残差块，用于深度特征提取
class ResNet_Block(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        # 主分支：两层卷积和BN，进行特征提取
        self.c1 = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=stride),  
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)  
        )

        # Shortcut分支：1x1卷积用于通道匹配和特征图大小调整
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0),  
            nn.BatchNorm2d(out_c),
        )

        # 使用Squeeze_Excitation模块进行通道注意力调整
        self.attn = Squeeze_Excitation(out_c)

    def forward(self, inputs):
        X = self.c1(inputs)                 # 特征提取
        inputjusted = self.c2(inputs)       # 输入通道匹配
        out = self.attn(X + inputjusted)    # 特征和输入相加（跳跃连接）后进行注意力调整
        return out


# ASPP模块，分割任务中的桥接层
class ASPP(nn.Module):
    def __init__(self, in_c, out_c, rate=[1, 6, 12, 18]):  # rate控制空洞卷积的扩张率dilation
        super().__init__()

        # 每个空洞卷积层，分别使用不同的扩张率，来提取不同尺度的特征，在不同的感受野下捕获空间信息
        self.c1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[0], padding=rate[0]),
            nn.BatchNorm2d(out_c)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[1], padding=rate[1]),
            nn.BatchNorm2d(out_c)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[2], padding=rate[2]),
            nn.BatchNorm2d(out_c)
        )
        self.c4 = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, dilation=rate[3], padding=rate[3]),
            nn.BatchNorm2d(out_c)
        )

        # 最后通过1x1卷积整合不同尺度的特征
        self.c5 = nn.Conv2d(out_c, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        x1 = self.c1(inputs)  
        x2 = self.c2(inputs)  
        x3 = self.c3(inputs)  
        x4 = self.c4(inputs)  
        x = x1 + x2 + x3 + x4  # 合并所有空洞卷积特征

        out = self.c5(x)  # 通过1x1卷积输出
        return out


# Attention_Block模块，用于解码过程中的特征引导
class Attention_Block(nn.Module):
    def __init__(self, in_c):  # in_c是一个tuple，包含两个输入的通道数
        super().__init__()
        out_c = in_c[1]  # 输出通道数等于第二个输入的通道数

        # 引导分支，用于生成引导特征
        self.g_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[0]),
            nn.ReLU(),
            nn.Conv2d(in_c[0], out_c, kernel_size=3, padding=1),
            nn.MaxPool2d((2, 2))  # 降采样
        )
        # 这里操作的是图中左侧的层，通道数和图像尺寸需要变得和右侧一致

        # 主分支特征提取
        self.x_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(in_c[1], out_c, kernel_size=3, padding=1),
        )
        # 尺寸不变

        # 将引导特征与主分支特征结合后的进一步处理
        self.gc_conv = nn.Sequential(
            nn.BatchNorm2d(in_c[1]),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, g, x):
        g_pool = self.g_conv(g)  # 引导特征生成
        x_conv = self.x_conv(x)  # 主分支特征提取
        gc_sum = g_pool + x_conv  # 引导特征与主分支特征相加
        gc_conv = self.gc_conv(gc_sum)  # 进一步处理
        out = gc_conv * x  # 注意力机制，调整主分支特征
        return out


# Decoder_Block模块，包含注意力引导和上采样，主要用于解码过程中恢复特征图尺寸
class Decoder_Block(nn.Module):
    def __init__(self, in_c, out_c):  # in_c是一个tuple
        super().__init__()

        self.a1 = Attention_Block(in_c)  # 注意力引导块
        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # 上采样，用于恢复特征图大小
        self.r1 = ResNet_Block(in_c[0] + in_c[1], out_c, stride=1)  # 使用ResNet块进一步提取特征

    def forward(self, g, x):
        d = self.a1(g, x)  # 注意力引导处理
        d = self.up(d)  # 上采样
        d = torch.cat([d, g], axis=1)  # 拼接特征图
        d = self.r1(d)  # 使用ResNet块进一步处理
        return d


# ResUNet++模型
class self_net(nn.Module):
    def __init__(self):
        super().__init__()

        # 编码器
        self.conv1 = Stem_Block(3, 16, stride=1)  # 初始卷积层
        self.conv2 = ResNet_Block(16, 32, stride=2)  # 残差块1
        self.conv3 = ResNet_Block(32, 64, stride=2)  # 残差块2
        self.conv4 = ResNet_Block(64, 128, stride=2)  # 残差块3

        # ASPP模块
        self.bridge = ASPP(128, 256)

        # 解码器
        self.decode1 = Decoder_Block([64, 256], 128)
        self.decode2 = Decoder_Block([32, 128], 64)
        self.decode3 = Decoder_Block([16, 64], 32)

        # ASPP模块，进行最后的特征融合
        self.aspp = ASPP(32, 16)
        self.output = nn.Conv2d(16, 4, kernel_size=1, padding=0)  # 最终输出层

    def forward(self, inputs):
        # 编码阶段
        c1 = self.conv1(inputs) # output: 1 16 224 224
        c2 = self.conv2(c1)     # 1 32 112 112
        c3 = self.conv3(c2)     # 1 64 56 56
        c4 = self.conv4(c3)     # 1 128 28 28

        # 中间桥接层
        b1 = self.bridge(c4)    # 1 256 28 28

        # 解码阶段
        d1 = self.decode1(c3, b1) # 注意力机制后，尺寸与b1一致；再从1 256 28 28 上采样到 1 256 56 56 特征拼接（这里是Unet的方法）后在改变通道数为128
        d2 = self.decode2(c2, d1) # 1 64 112 112
        d3 = self.decode3(c1, d2) # 1 32 224 224

        # 输出结果
        output = self.aspp(d3)       # 1 16 224 224
        output = self.output(output) # 1 4 224 224

        return output