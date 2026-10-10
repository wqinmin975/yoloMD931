"""
MAE (Masked Autoencoder) 航拍域骨干预训练
- Encoder: 直接复用 MD-YOLO 的 Backbone (stem + MAC + C2f)
- 掩码策略: 在「输入图像」上做 patch 掩码 (SimMIM 风格)
    -> 被掩区域像素置 0 后再送入 backbone, 编码器无法看到被掩内容,
       必须依靠周围上下文重建, 从而避免出现 "在 C5 特征图上掩码导致的平凡解"。
- Decoder: 轻量 Transformer 解码器, 重建被掩 patch 的像素
- 预训练完成后, 丢弃 decoder, 只把 encoder(backbone) 权重加载回 MD-YOLO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .md_yolo import MDYOLO, ConvBNAct


class MAEDecoder(nn.Module):
    """轻量 MAE decoder: Transformer blocks + 像素重建头
    位置编码改为可学习 (不再冻结为 0), 提升重建质量。
    """
    def __init__(self, embed_dim=512, decoder_dim=256, depth=4, num_heads=8,
                 patch_size=32, img_size=640):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        # 可学习位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
        # 简易 Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads, dim_feedforward=decoder_dim * 4,
            batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        # 重建: 每个 patch 输出 patch_size * patch_size * 3 像素
        self.pred = nn.Linear(decoder_dim, patch_size * patch_size * 3)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, tokens):
        """
        tokens: (B, N, D) 全部 patch 的 encoder 特征 (含被掩位置)
        返回: (B, N, patch^2*3) 重建像素
        """
        x = self.decoder_embed(tokens)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        pred = self.pred(x)
        return pred


class MAEForPretrain(nn.Module):
    """
    完整 MAE 预训练模型:
      1) 在输入图像上随机 mask (mask_ratio) 若干 patch, 被掩像素置 0
      2) encoder (MD-YOLO backbone) 编码 (含被掩位置的) 全部特征
      3) decoder 重建全部 patch 像素
      4) loss 只在被 mask 的 patch 上计算
    """
    def __init__(self, md_yolo_model, img_size=640, patch_size=32,
                 mask_ratio=0.75, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.mask_ratio = mask_ratio
        self.img_size = img_size

        # Encoder: 复用 MD-YOLO backbone (stem + 4 stage)
        self.model = md_yolo_model
        self.encoder_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)

        # Decoder
        self.decoder = MAEDecoder(embed_dim=embed_dim, decoder_dim=256,
                                 depth=4, num_heads=8,
                                 patch_size=patch_size, img_size=img_size)

    def backbone_parameters(self):
        """返回仅骨干 + encoder_proj 的可训练参数 (neck/head 不训练)。"""
        m = self.model
        return (list(m.stem.parameters()) + list(m.down1.parameters()) +
                list(m.stage1.parameters()) + list(m.down2.parameters()) +
                list(m.stage2.parameters()) + list(m.down3.parameters()) +
                list(m.stage3.parameters()) + list(m.down4.parameters()) +
                list(m.stage4.parameters()) + list(self.encoder_proj.parameters()))

    def random_mask(self, B, device):
        """
        生成 patch 级掩码。返回:
          mask: (B, N)  1=被掩, 0=可见
          len_keep: 可见 patch 数
        """
        N = self.num_patches
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask = torch.ones(B, N, device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask, len_keep

    def forward_encoder(self, imgs):
        """
        在输入图像上做 patch 掩码后送入 backbone。
        返回: tokens (B,N,D), mask (B,N)
        """
        B = imgs.shape[0]
        device = imgs.device
        G = self.grid_size
        p = self.patch_size

        # patch 级掩码 -> 像素级掩码, 被掩 patch 像素置 0
        mask, _ = self.random_mask(B, device)            # (B, N)
        mask_img = mask.reshape(B, 1, G, G)
        mask_img = mask_img.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
        masked_imgs = imgs * (1.0 - mask_img)            # (B,3,H,W)

        # backbone (使用被掩图像)
        m = self.model
        c2 = m.stem(masked_imgs)
        c3 = m.stage1(m.down1(c2))
        c4 = m.stage2(m.down2(c3))
        c5 = m.stage3(m.down3(c4))
        c6 = m.stage4(m.down4(c5))                       # (B, D, G, G)
        tokens = self.encoder_proj(c6).flatten(2).transpose(1, 2)  # (B, N, D)
        return tokens, mask

    def forward_loss(self, imgs, pred, mask):
        """
        pred: (B, N, patch^2*3)
        mask: (B, N), 1=被掩
        """
        B = imgs.shape[0]
        p = self.patch_size
        G = self.grid_size
        # 把 imgs 切 patch 作为 target (B, 3, G, p, G, p) -> (B, N, p*p*3)
        target = imgs.reshape(B, 3, G, p, G, p)
        target = target.permute(0, 2, 4, 1, 3, 5).reshape(B, self.num_patches, -1)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                         # (B, N)
        loss = (loss * mask).sum() / mask.sum()          # 只在被掩 patch
        return loss

    def forward(self, imgs):
        tokens, mask = self.forward_encoder(imgs)
        pred = self.decoder(tokens)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def build_mae_from_md_yolo(num_classes=10, img_size=640, patch_size=32, mask_ratio=0.75):
    """从 MD-YOLO 构建 MAE 预训练模型"""
    md = MDYOLO(num_classes=num_classes)
    embed_dim = md.backbone_channels[-1]
    mae = MAEForPretrain(md, img_size=img_size, patch_size=patch_size,
                         mask_ratio=mask_ratio, embed_dim=embed_dim)
    return mae, md


if __name__ == "__main__":
    mae, md = build_mae_from_md_yolo()
    x = torch.randn(2, 3, 640, 640)
    loss, pred, mask = mae(x)
    print(f"MAE loss: {loss.item():.4f}, pred: {pred.shape}, mask ratio: {mask.mean().item():.2f}")
    n = sum(p.numel() for p in mae.parameters()) / 1e6
    print(f"MAE total params: {n:.2f}M")
