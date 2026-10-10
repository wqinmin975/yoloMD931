"""
跨域诊断: 在真实 UAVDT 测试集上检查 VisDrone 训练模型 (v2) 的 car 定位/分类迁移情况。
- 低置信度解码 -> 按 map_src 仅保留 car (3->0,4->0)
- 统计每图检测数、最大得分
- 计算预测 car 框与 GT car 框的 best-IoU, 统计 >=0.25 / >=0.5 的图比例
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.md_yolo import MDYOLO
from models.box_utils import decode_predictions
from data.dataset import ValDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--val_img', required=True)
    ap.add_argument('--val_lbl', required=True)
    ap.add_argument('--img_size', type=int, default=320)
    ap.add_argument('--subset', type=int, default=200)
    ap.add_argument('--conf', type=float, default=0.001)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MDYOLO(num_classes=10).to(device)
    ck = torch.load(args.weights, map_location='cpu')
    sd = ck.get('model', ck)
    model.load_state_dict(sd, strict=False)
    model.eval()

    ds = ValDataset(args.val_img, args.val_lbl, ['car'], img_size=args.img_size)
    if args.subset:
        idx = list(range(min(args.subset, len(ds))))
    else:
        idx = list(range(len(ds)))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: b)

    map_src = {3: 0, 4: 0}     # VisDrone car/van -> UAVDT car
    n = 0
    tot_det = 0
    max_sc = 0.0
    sum_best = 0.0
    ge25 = 0
    ge50 = 0
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if i >= len(idx):
                break
            img, target = sample[0]
            im = img.unsqueeze(0).to(device) if img.dim() == 3 else img.to(device)
            out = model(im)
            d = decode_predictions(out, model.head_strides, args.img_size,
                                  conf_thr=args.conf, iou_thr=0.7)[0]
            boxes, scores, labels = d['boxes'], d['scores'], d['labels']
            # 仅保留 car (3,4 -> 0)
            m = torch.tensor([int(l) in map_src for l in labels.tolist()])
            boxes = boxes[m]
            scores = scores[m]
            tot_det += boxes.shape[0]
            if scores.numel() > 0:
                max_sc = max(max_sc, float(scores.max()))
            # GT car
            g = target['boxes'].cpu()
            gl = target['labels'].cpu()
            if boxes.numel() > 0 and g.numel() > 0:
                # 逐预测取与任意 GT 的最大 IoU
                best_per_img = 0.0
                for k in range(boxes.shape[0]):
                    b = boxes[k]
                    x1 = torch.max(b[0], g[:, 0]); y1 = torch.max(b[1], g[:, 1])
                    x2 = torch.min(b[2], g[:, 2]); y2 = torch.min(b[3], g[:, 3])
                    inter = (x2-x1).clamp(0)*(y2-y1).clamp(0)
                    a1 = (b[2]-b[0]).clamp(0)*(b[3]-b[1]).clamp(0)
                    a2 = (g[:, 2]-g[:, 0]).clamp(0)*(g[:, 3]-g[:, 1]).clamp(0)
                    iou = inter/(a1+a2-inter+1e-9)
                    best_per_img = max(best_per_img, float(iou.max()) if iou.numel() else 0.0)
                sum_best += best_per_img
                if best_per_img >= 0.25:
                    ge25 += 1
                if best_per_img >= 0.5:
                    ge50 += 1
            n += 1
    print(f"[xdiag] images={n} dets/img={tot_det/max(n,1):.1f} max_score={max_sc:.3f}")
    print(f"[xdiag] mean best-IoU(car)={sum_best/max(n,1):.3f}  "
          f">=0.25:{ge25}/{n}  >=0.5:{ge50}/{n}")


if __name__ == '__main__':
    main()
