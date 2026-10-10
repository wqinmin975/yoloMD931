"""
DAFA (Dual Attention Feature Aggregation) 模块
Neck 跨分辨率特征对齐 + 通道/空间双注意力增强。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mac import ConvBNAct


class ChannelAttention(nn.Module):
    """轻量通道注意力 (SE-style, 无全连接层, 用 1x1 卷积)"""
    def __init__(self, c, reduction=4):
        super().__init__()
        mid = max(c // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid, c, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        a = self.mlp(self.avg_pool(x))
        m = self.mlp(self.max_pool(x))
        return x * self.sigmoid(a + m)


class SpatialAttention(nn.Module):
    """局部窗口空间注意力 (Swin-style, 只在 7x7 窗口内计算, 抑制远距背景)"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attn))


class AlignConv(nn.Module):
    """轻量对齐: 上采样到参考特征尺寸 (工程简化版, 实际可替换为 DCNv2)"""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, kernel_size=3, padding=1)

    def forward(self, x, ref):
        """x: 需要对齐的特征, ref: 参考特征 (同 c)"""
        H, W = ref.shape[-2:]
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        return self.conv(x) + ref  # 残差连接


class DAFA(nn.Module):
    """
    Dual Attention Feature Aggregation
    输入: 上层小分辨率特征 feat_up, 下层高分辨率特征 feat_low
    输出: 融合后的特征 (与 feat_low 同分辨率)
    """
    def __init__(self, c_low, c_up, c_out=None):
        super().__init__()
        c_out = c_out or c_low
        # 1x1 对齐通道数
        self.up_proj = ConvBNAct(c_up, c_low, k=1, s=1)
        self.low_proj = ConvBNAct(c_low, c_low, k=1, s=1)
        # 跨尺度对齐
        self.align = AlignConv(c_low)
        # 融合
        self.fuse = ConvBNAct(c_low * 2, c_out, k=3, s=1)
        # 双注意力
        self.channel_attn = ChannelAttention(c_out)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, feat_low, feat_up):
        feat_up_p = self.up_proj(feat_up)
        feat_low_p = self.low_proj(feat_low)
        # 对齐 feat_up 到 feat_low 尺寸
        feat_up_aligned = self.align(feat_up_p, feat_low_p)
        # 拼接融合
        cat = torch.cat([feat_low_p, feat_up_aligned], dim=1)
        out = self.fuse(cat)
        # 双注意力
        out = self.channel_attn(out)
        out = self.spatial_attn(out)
        return out


if __name__ == "__main__":
    dafa = DAFA(128, 64, 128)
    low = torch.randn(2, 128, 80, 80)
    up = torch.randn(2, 64, 40, 40)
    out = dafa(low, up)
    print(f"DAFA: low={low.shape}, up={up.shape} -> out={out.shape}")
    assert out.shape == (2, 128, 80, 80)
    print("DAFA OK")
