import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, bias=False, se_ratio=4):
        super(DepthwiseSeparableConv, self).__init__()

        if dilation != 1:
            padding = dilation * (kernel_size - 1) // 2

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()  # 激活函数
        self.se = SqueezeExcitation(in_channels=in_channels, out_channels=in_channels, reduction=se_ratio)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.se(out)
        out = self.pointwise(out)
        return out

class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes,act = 'relu'):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1,bias=False)  #卷积层  就是赋值的一个操作   然后在类中创建了一个对象  算一种格式把
        self.bn = nn.BatchNorm2d(planes)  # bn层  批量归一化
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        if act == 'gelu':
            self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class IConv1x1(nn.Module):
    def __init__(self, inplanes, planes,bias = False):
        super(IConv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1,bias=bias)  #卷积层  就是赋值的一个操作   然后在类中创建了一个对象  算一种格式把

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3):
        super(ConvBN, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels=64, kernel_size=3, act='relu'):
        super(ConvBNReLU, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                              padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        if act == 'gelu':
            self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduction: int = 4,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid,
            pool='avgpool'
    ):
        super(SqueezeExcitation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.transition = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        if out_channels // reduction == 0:
            reduction = 1

        if pool == 'avgpool':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'maxpool':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            print('Parameter pool is not avgpool or maxpool')
            return
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.transition(x)
        scale = self._scale(x)
        return scale * x

class DilatedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 2,
        bias: bool = False,
        use_bn: bool = True,
        activation: str = "relu",
    ):
        super(DilatedConvBlock, self).__init__()

        padding = dilation * (kernel_size - 1) // 2
        # 带膨胀率的卷积层
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,   # 建议 padding ≈ dilation * (kernel_size - 1) // 2 以大致保持尺寸
            dilation=dilation, # 设置膨胀率
            bias=bias,
        )

        # BatchNorm，可选
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = nn.Identity()

        # 激活函数，可选
        activation = activation.lower()
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "none":
            self.act = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}. "
                             f"Supported: 'relu', 'none'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：
            1. 带 dilation 的卷积；
            2. BatchNorm（若启用）；
            3. 激活函数（若启用）。
        """
        x = self.conv(x)  # 膨胀卷积：通过 dilation 扩大感受野
        x = self.bn(x)    # 批归一化（可选）
        x = self.act(x)   # 激活函数（可选）
        return x

class ChannelAttention(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()

        # 中间通道数，至少为 1，防止 in_channels < reduction 时出现 0 通道
        mid_channels = max(in_channels // reduction, 1)

        # 自适应全局平均池化和最大池化：将空间维度聚合为 1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 输出形状: (N, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 输出形状: (N, C, 1, 1)

        # 共享 MLP：使用 1×1 卷积实现 C -> C // reduction -> C
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=True),
        )

        # Sigmoid 激活，用于生成 [0, 1] 范围的信道权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播步骤：
            1. 对输入特征在空间维度上分别做全局平均池化和全局最大池化；
            2. 将两个池化结果分别通过共享 MLP；
            3. 将两个分支输出相加后通过 sigmoid 得到信道注意力权重；
            4. 将原始输入特征按通道乘以该权重，得到加权后的输出特征。
        """
        # 全局平均池化分支，输出形状: (N, C, 1, 1)
        avg_out = self.avg_pool(x)
        avg_out = self.mlp(avg_out)

        # 全局最大池化分支，输出形状: (N, C, 1, 1)
        max_out = self.max_pool(x)
        max_out = self.mlp(max_out)

        # 将两个分支结果相加并通过 sigmoid 得到信道注意力权重
        # 注意力权重形状: (N, C, 1, 1)
        attn = self.sigmoid(avg_out + max_out)

        # 将注意力权重与输入特征按通道相乘（广播机制），得到加权后的特征
        return attn

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()

        # 根据卷积核大小设置 padding，以尽量保持 H, W 不变
        padding = (kernel_size - 1) // 2

        # 输入通道为 2（通道平均池化与最大池化的拼接），输出通道为 1（空间注意力图）
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        # Sigmoid 用于生成 0~1 范围的空间注意力权重
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)          # (N, 1, H, W)

        # 通道最大池化：在 dim=1 上求最大值，保持维度 (N, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)        # (N, 1, H, W)

        # 在通道维上拼接两个池化结果，得到 (N, 2, H, W)
        feat = torch.cat([avg_out, max_out], dim=1)

        # 卷积生成空间注意力图，再通过 Sigmoid 得到权重 (N, 1, H, W)
        attn = self.sigmoid(self.conv(feat))

        # 对输入特征在空间位置上逐点加权（广播机制）
        return attn

#坐标注意力
class CoordinateAttention(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(8, in_channels // reduction)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_h = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)
        self.conv_w = nn.Conv2d(hidden, in_channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (B,C,H,W)
        B, C, H, W = x.shape

        x_h = x.mean(dim=3, keepdim=True)                 # (B,C,H,1)
        x_w = x.mean(dim=2, keepdim=True).transpose(2, 3) # (B,C,W,1)

        y = torch.cat([x_h, x_w], dim=2)                  # (B,C,H+W,1)
        y = self.conv1(y)                                 # (B,hidden,H+W,1)

        y_h, y_w = torch.split(y, [H, W], dim=2)
        y_w = y_w.transpose(2, 3)                         # (B,hidden,1,W)

        g_h = torch.sigmoid(self.conv_h(y_h))             # (B,C,H,1)
        g_w = torch.sigmoid(self.conv_w(y_w))             # (B,C,1,W)
        return g_h, g_w


class Haar(nn.Module):
    def __init__(self):
        super(Haar, self).__init__()

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. 若 H 或 W 是奇数，就在底 / 右各补 1 行 / 列，使之变成偶数
        pad_h = H % 2  # 若为 1，说明需要多 pad 一行
        pad_w = W % 2  # 若为 1，说明需要多 pad 一列
        if pad_h or pad_w:
            # F.pad 的参数顺序是 (left, right, top, bottom)
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            B, C, H, W = x.shape  # 更新一下 H, W

        # 2. 正常做二维 Haar 分解
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2

        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        return x_LL, x_HL, x_LH, x_HH


class Ihaar(nn.Module):
    def __init__(self):
        super(Ihaar, self).__init__()

    def forward(self, x):
        r = 2
        in_batch, in_channel, in_height, in_width = x.size()
        out_batch, out_channel, out_height, out_width = in_batch, int(
            in_channel / (4)), 2 * in_height, 2 * in_width
        x1 = x[:, 0:out_channel, :, :] / 2
        x2 = x[:, out_channel:out_channel * 2, :, :] / 2
        x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
        x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
        h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
        h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
        h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
        h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
        h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
        return h




