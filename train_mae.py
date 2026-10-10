"""
MAE 预训练主脚本 (航拍域自监督)
用法:
    python train_mae.py --config configs/mae_config.yaml
    # 或直接传参:
    python train_mae.py --unlabeled_dir /path/to/unlabeled/images --save_dir ./checkpoints/mae
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.mae_pretrain import build_mae_from_md_yolo
from data.dataset import VisDroneUnlabeled
from utils.config import apply_config_to_args


def train_one_epoch(mae, loader, optimizer, device, epoch, log_interval=20):
    mae.train()
    total_loss = 0.0
    for i, (imgs, _) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        loss, _, _ = mae(imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % log_interval == 0:
            print(f"  Epoch {epoch} | iter {i}/{len(loader)} | loss={loss.item():.4f}")
    return total_loss / max(len(loader), 1)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[MAE Pretrain] device={device}, unlabeled_dir={args.unlabeled_dir}")

    # 1. 构建模型
    mae, md_model = build_mae_from_md_yolo(
        num_classes=args.num_classes,
        img_size=args.img_size,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
    )
    mae = mae.to(device)

    # 2. 数据
    dataset = VisDroneUnlabeled(args.unlabeled_dir, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)

    # 3. 优化器: 仅训练 backbone(encoder) + decoder; neck/head 不参与 MAE
    optimizer = AdamW(
        mae.backbone_parameters() + list(mae.decoder.parameters()),
        lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 4. 训练循环
    os.makedirs(args.save_dir, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(mae, loader, optimizer, device, epoch)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | avg_loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        # 保存最优 backbone
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, 'mae_backbone_best.pth')
            torch.save({
                'epoch': epoch,
                'backbone': md_model.get_backbone_state_dict(),  # 仅骨干, 加载时 strict=False
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
            }, save_path)
            print(f"  Saved best backbone -> {save_path}")

    print(f"[MAE Pretrain] Done. Best loss={best_loss:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--unlabeled_dir', type=str, required=True,
                   help='无标注航拍图像目录 (VisDrone 训练帧 / UAVDT 抽帧)')
    p.add_argument('--save_dir', type=str, default='./checkpoints/mae')
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--patch_size', type=int, default=32)
    p.add_argument('--mask_ratio', type=float, default=0.75)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=800)
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--config', type=str, default=None, help='YAML 配置 (configs/mae_config.yaml)')
    args = p.parse_args()
    apply_config_to_args(args, args.config)
    main(args)
