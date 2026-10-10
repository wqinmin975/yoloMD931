"""
MD-YOLO 纯监督训练 (随机初始化, 用于实测验证 P0-A: 检测损失确实驱动学习)。
也可通过 --mae_ckpt 加载 MAE 预训练 backbone。

用法:
    python tools/train_sup.py \
        --img_dir E:/visdrone_work/VisDrone2019-DET-train/images \
        --lbl_dir E:/visdrone_work/VisDrone2019-DET-train/labels \
        --img_size 512 --subset 300 --epochs 20 --batch 4 \
        --save_dir ./checkpoints/sup
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.md_yolo import MDYOLO
from models.detection_loss import DetectionLoss
from data.dataset import VisDroneLabeled, collate_flex


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', type=str, required=True)
    p.add_argument('--lbl_dir', type=str, required=True)
    p.add_argument('--mae_ckpt', type=str, default=None)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--subset', type=int, default=0, help='仅用前 N 张图 (0=全部)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--opt', type=str, default='adamw', choices=['sgd', 'adamw'])
    p.add_argument('--sched', type=str, default='cosine', choices=['none', 'cosine'])
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--box_loss', type=str, default='wiou', choices=['wiou', 'ciou'])
    p.add_argument('--assign_topk', type=int, default=10)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--save_dir', type=str, default='./checkpoints/sup')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[train_sup] device={device}")

    model = MDYOLO(num_classes=args.num_classes).to(device)
    if args.mae_ckpt and os.path.exists(args.mae_ckpt):
        ckpt = torch.load(args.mae_ckpt, map_location=device)
        sd = ckpt.get('backbone', ckpt)
        model.load_state_dict(sd, strict=False)
        print(f"[train_sup] 加载 MAE backbone: {args.mae_ckpt}")

    criterion = DetectionLoss(strides=model.head_strides, img_size=args.img_size, nc=args.num_classes,
                              box_loss_type=args.box_loss, assign_topk=args.assign_topk)
    if args.opt == 'adamw':
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = None
    if args.sched == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(opt, T_max=max(args.epochs - args.warmup, 1), eta_min=1e-5)

    ds = VisDroneLabeled(args.img_dir, args.lbl_dir, img_size=args.img_size, augment=True)
    if args.subset:
        ds.imgs = ds.imgs[:args.subset]
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        collate_fn=collate_flex)
    print(f"[train_sup] 图像数={len(ds)}, batch={args.batch}, epochs={args.epochs}")

    os.makedirs(args.save_dir, exist_ok=True)
    history = []
    model.train()
    for ep in range(1, args.epochs + 1):
        # warmup: 前 warmup 个 epoch 线性升温, 之后 cosine
        if ep <= args.warmup:
            cur_lr = args.lr * (ep / max(args.warmup, 1))
            for g in opt.param_groups:
                g['lr'] = cur_lr
        elif scheduler is not None:
            scheduler.step()
        tot = 0.0
        n = 0
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            n += 1
        avg = tot / max(n, 1)
        history.append(avg)
        print(f"  epoch {ep}/{args.epochs} | avg_loss={avg:.4f} | lr={opt.param_groups[0]['lr']:.2e}")
    torch.save({'epoch': args.epochs, 'model': model.state_dict(),
                'history': history},
               os.path.join(args.save_dir, 'sup_last.pth'))
    print(f"[train_sup] 已保存 -> {os.path.join(args.save_dir, 'sup_last.pth')}")
    print(f"[train_sup] loss 首={history[0]:.4f} -> 末={history[-1]:.4f}, "
          f"下降={history[-1] < history[0]}")


if __name__ == '__main__':
    main()
