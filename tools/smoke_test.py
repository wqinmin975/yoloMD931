"""
真实数据 smoke test: 验证 MD-YOLO + DetectionLoss 在 VisDrone 子集上
- 数据可加载 (letterbox 后框与图对齐)
- 前向 / 损失 / 反向不崩溃, loss 有限
- 一次优化 step 后 loss 下降 (证明 P0-A 已真正可训练)
- 预测解码可用 (decode_predictions 产出有限框)

用法:
    python tools/smoke_test.py \
        --img_dir E:/visdrone_work/VisDrone2019-DET-train/images \
        --lbl_dir E:/visdrone_work/VisDrone2019-DET-train/labels \
        --img_size 512 --subset 16
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.md_yolo import MDYOLO
from models.detection_loss import DetectionLoss
from models.box_utils import decode_predictions
from data.dataset import VisDroneLabeled, collate_flex


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--img_dir', type=str, required=True)
    p.add_argument('--lbl_dir', type=str, required=True)
    p.add_argument('--img_size', type=int, default=512)
    p.add_argument('--subset', type=int, default=16)
    p.add_argument('--num_classes', type=int, default=10)
    args = p.parse_args()

    device = torch.device('cpu')
    print(f"[smoke] device={device}, img_size={args.img_size}, subset={args.subset}")

    ds = VisDroneLabeled(args.img_dir, args.lbl_dir, img_size=args.img_size, augment=False)
    if args.subset:
        ds.imgs = ds.imgs[:args.subset]
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True,
                                        num_workers=0, collate_fn=collate_flex)

    model = MDYOLO(num_classes=args.num_classes).to(device)
    criterion = DetectionLoss(strides=model.head_strides, img_size=args.img_size, nc=args.num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print(f"[smoke] 模型参数量: {sum(x.numel() for x in model.parameters())/1e6:.2f}M")

    # ---- 1) 数据对齐检查: target 框是否都在 [0, img_size] 内 ----
    imgs, targets = next(iter(loader))
    boxes_all = torch.cat([t['boxes'] for t in targets if t['boxes'].numel() > 0], 0)
    in_bounds = ((boxes_all >= 0).all(1) & (boxes_all <= args.img_size).all(1)).all().item()
    n_boxes = int(boxes_all.numel() > 0 and boxes_all.shape[0])
    print(f"[smoke] target 框总数={boxes_all.shape[0] if boxes_all.numel() else 0}, "
          f"全部在 [0,{args.img_size}] 内={in_bounds}")

    # ---- 2) 前向 + 损失 + 反向 ----
    model.train()
    imgs = imgs.to(device)
    preds = model(imgs)
    loss = criterion(preds, targets)
    finite = torch.isfinite(loss).item()
    loss.backward()
    g_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"[smoke] loss={loss.item():.4f}, finite={finite}, grad_finite={g_ok}")

    # ---- 3) 一次 step 后 loss 是否下降 ----
    opt.step()
    preds2 = model(imgs)
    loss2 = criterion(preds2, targets)
    print(f"[smoke] loss_after_step={loss2.item():.4f}, 下降={loss2.item() < loss.item()}")

    # ---- 4) 解码预测 ----
    model.eval()
    with torch.no_grad():
        out = model(imgs[:2])
        dets = decode_predictions(out, model.head_strides, args.img_size, conf_thr=0.01)
    n_det = int(dets[0]['boxes'].shape[0])
    det_finite = all(torch.isfinite(d['boxes']).all() for d in dets)
    print(f"[smoke] 解码预测 img0 框数={n_det}, 所有框有限={det_finite}")

    ok = in_bounds and finite and g_ok and (loss2.item() < loss.item()) and det_finite
    print(f"\n[smoke] 整体通过={ok}")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
