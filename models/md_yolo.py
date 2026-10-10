"""
MD-YOLO: Multi-scale Adaptive & Dual-Attention Enhanced YOLOv11
- Backbone: 用 MAC 替换原生 stride=2 下采样
- Neck:     用 DAFA 替换原生 C2f-PAN 融合
- Head:     P2/P3/P4/P5 四分支检测头
"""
import torch
import torch.nn as nn
from .mac import MACBlock, ConvBNAct
from .dafa import DAFA


class C2fBlock(nn.Module):
    """简化版 C2f: 两路 split + 串接 bottleneck + 投影回 c"""
    def __init__(self, c, n=1):
        super().__init__()
        self.cv1 = ConvBNAct(c, c // 2, k=1)
        self.cv2 = ConvBNAct(c, c // 2, k=1)
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(ConvBNAct(c // 2, c // 2, k=3), ConvBNAct(c // 2, c // 2, k=3))
            for _ in range(n)
        ])
        # concat: cv1 输出 + cv2 输出 + n 个 bottleneck 输出, 每路 c//2
        total = (2 + n) * (c // 2)
        self.cv3 = ConvBNAct(total, c, k=1)

    def forward(self, x):
        a = self.cv1(x)
        b = self.cv2(x)
        feats = [a, b]
        cur = b
        for blk in self.bottlenecks:
            cur = blk(cur)
            feats.append(cur)
        return self.cv3(torch.cat(feats, dim=1))


class DetectHead(nn.Module):
    """解耦检测头 (anchor-free, 与 YOLOv8/v11 一致):
       每个空间位置预测 1 个框: 4 个回归坐标 + nc 个分类 logits。
    """
    def __init__(self, in_channels, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # 分类分支: nc 通道 (sigmoid 后各类独立概率)
        self.cls = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        # 回归分支: 4 通道 (tx, ty, tw, th, 单位为 stride)
        self.reg = nn.Conv2d(in_channels, 4, kernel_size=1)

    def forward(self, x):
        return self.cls(x), self.reg(x)


class MDYOLO(nn.Module):
    """
    MD-YOLO 骨干 + Neck + Head 整体结构
    输入: (B, 3, 640, 640)
    输出: 四个检测头的预测 (P2/P3/P4/P5)
    """
    def __init__(self, num_classes=10, width=0.50, depth=0.34):
        super().__init__()
        nc = num_classes
        # 通道基数 (YOLOv11s 宽度系数)
        base = [64, 128, 256, 512, 1024]
        ch = [max(int(c * width), 16) for c in base]
        dep = max(round(depth), 1)

        # ---- Backbone (MAC 下采样) ----
        self.stem = ConvBNAct(3, ch[0], k=3, s=2)           # /2
        self.down1 = MACBlock(ch[0], ch[1])                 # /4
        self.stage1 = C2fBlock(ch[1], n=dep)
        self.down2 = MACBlock(ch[1], ch[2])                 # /8
        self.stage2 = C2fBlock(ch[2], n=dep * 2)
        self.down3 = MACBlock(ch[2], ch[3])                 # /16
        self.stage3 = C2fBlock(ch[3], n=dep * 2)
        self.down4 = MACBlock(ch[3], ch[4])                 # /32
        self.stage4 = C2fBlock(ch[4], n=dep)

        # ---- Neck (DAFA 融合) ----
        # P5 -> P4 -> P3 -> P2 (自顶向下)
        self.up_p4 = DAFA(ch[3], ch[4], ch[3])
        self.up_p3 = DAFA(ch[2], ch[3], ch[2])
        self.up_p2 = DAFA(ch[1], ch[2], ch[1])
        # 自底向上 (轻量, 跨通道下采样)
        self.down_p3 = ConvBNAct(ch[1], ch[2], k=3, s=2)   # p2(64) -> p3(128)
        self.down_p4 = ConvBNAct(ch[2], ch[3], k=3, s=2)   # p3(128) -> p4(256)

        # ---- Head (P2/P3/P4/P5) ----
        self.head_p2 = DetectHead(ch[1], nc)
        self.head_p3 = DetectHead(ch[2], nc)
        self.head_p4 = DetectHead(ch[3], nc)
        self.head_p5 = DetectHead(ch[4], nc)

        # 保存骨干输出通道 (供 MAE 加载)
        self.backbone_channels = ch
        # 四个检测头的下采样倍率 (P2=4, P3=8, P4=16, P5=32)
        # c3=/4, c4=/8, c5=/16, c6=/32
        self.head_strides = (4, 8, 16, 32)

    def forward(self, x):
        # Backbone
        c2 = self.stem(x)            # /2
        c3 = self.stage1(self.down1(c2))   # /4  (P2 输入)
        c4 = self.stage2(self.down2(c3))   # /8  (P3 输入)
        c5 = self.stage3(self.down3(c4))   # /16 (P4 输入)
        c6 = self.stage4(self.down4(c5))   # /32 (P5 输入)

        # Neck: 自顶向下
        p4 = self.up_p4(c5, c6)
        p3 = self.up_p3(c4, p4)
        p2 = self.up_p2(c3, p3)
        # 自底向上 (轻量融合)
        p3 = p3 + self.down_p3(p2)
        p4 = p4 + self.down_p4(p3)

        # Head
        out_p2 = self.head_p2(p2)
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(c6)
        return [out_p2, out_p3, out_p4, out_p5]

    def get_backbone_state_dict(self):
        """导出骨干权重 (供 MAE 预训练初始化)"""
        keys = ['stem', 'down1', 'stage1', 'down2', 'stage2',
                'down3', 'stage3', 'down4', 'stage4']
        sd = self.state_dict()
        return {k: v for k, v in sd.items() if any(k.startswith(p + '.') for p in keys)}


if __name__ == "__main__":
    model = MDYOLO(num_classes=10)
    x = torch.randn(1, 3, 640, 640)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"Head P{2+i}: cls={o[0].shape}, reg={o[1].shape}")
    # 参数量
    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {n:.2f}M")
