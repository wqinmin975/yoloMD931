"""
MAC (Multi-scale Adaptive Channel) Module
替换 YOLOv11 原生 stride=2 下采样，保留细粒度信息并增强多尺度表达。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    """Conv + BN + SiLU，YOLOv11 风格基础卷积块"""
    def __init__(self, c_in, c_out, k=3, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MACBlock(nn.Module):
    """
    Multi-scale Adaptive Channel (MAC) 模块
    输入: (B, C, H, W)
    输出: (B, C_out, H/2, W/2)  —— 替代 stride=2 下采样

    设计:
      1) 分支1: 1x1 卷积，大感受野快速降维
      2) 分支2: 3x3 stride=2 卷积，标准下采样
      3) 分支3: 5x5 stride=2 卷积，更大感受野
      4) 分支4: 平均池化 + 1x1 卷积，保留全局上下文
      5) 门控融合: 逐分支预测权重 (sigmoid)，自适应选择
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        mid = c_out // 4

        # 分支1: 1x1 降维 + 深度可分离 3x3 stride=2
        self.branch1 = nn.Sequential(
            ConvBNAct(c_in, mid, k=1, s=1),
            ConvBNAct(mid, mid, k=3, s=2, g=mid),  # depthwise
        )
        # 分支2: 3x3 stride=2 标准卷积
        self.branch2 = ConvBNAct(c_in, mid, k=3, s=2)
        # 分支3: 5x5 stride=2 卷积
        self.branch3 = ConvBNAct(c_in, mid, k=5, s=2)
        # 分支4: AvgPool + 1x1 (全局上下文)
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNAct(c_in, mid, k=1, s=1),
        )

        # 门控融合: 自适应学习每个分支权重
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in, 4, kernel_size=1),
            nn.Sigmoid(),
        )
        # 融合后投影到 c_out (门控加权求和后通道数为 mid)
        self.proj = ConvBNAct(mid, c_out, k=1, s=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        # 对齐尺寸 (理论上都 H/2 x W/2，加安全 padding)
        target_h, target_w = b2.shape[-2], b2.shape[-1]
        b1 = F.interpolate(b1, size=(target_h, target_w), mode='bilinear', align_corners=False)
        b3 = F.interpolate(b3, size=(target_h, target_w), mode='bilinear', align_corners=False)
        b4 = F.interpolate(b4, size=(target_h, target_w), mode='bilinear', align_corners=False)

        # 门控权重 (B, 4, 1, 1)
        w = self.gate(x)
        w = w.chunk(4, dim=1)  # 4 x (B,1,1,1)

        out = w[0] * b1 + w[1] * b2 + w[2] * b3 + w[3] * b4
        return self.proj(out)


if __name__ == "__main__":
    # 单元测试
    m = MACBlock(64, 128)
    x = torch.randn(2, 64, 80, 80)
    y = m(x)
    print(f"MAC input: {x.shape} -> output: {y.shape}")
    assert y.shape == (2, 128, 40, 40), "MAC shape mismatch"
    print("MACBlock OK")
