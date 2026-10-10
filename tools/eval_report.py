"""
在 VisDrone val 全量上评测, 输出详细的真实 mAP:
  mAP@0.25, mAP@0.5, mAP@0.5:0.95, 以及每类 AP@0.5。
用于实测验证报告。
"""
import os, sys, argparse, torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.md_yolo import MDYOLO
from models.box_utils import decode_predictions
from data.dataset import ValDataset
import eval_cross_domain as E

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--val_img', required=True)
    ap.add_argument('--val_lbl', required=True)
    ap.add_argument('--img_size', type=int, default=320)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--classes', nargs='+',
                   default=['pedestrian','people','bicycle','car','van','truck',
                            'tricycle','awning-tricycle','bus','motor'])
    args = ap.parse_args()
    dev = torch.device('cpu')
    nc = len(args.classes)
    m = MDYOLO(num_classes=nc)
    sd = torch.load(args.weights, map_location='cpu')
    m.load_state_dict(sd.get('model', sd.get('student', sd)), strict=False)
    m.eval()
    ds = ValDataset(args.val_img, args.val_lbl, args.classes, img_size=args.img_size)
    preds, gts = E.collect_predictions(m, ds, {}, args.img_size, dev, args.conf, 0.7)
    for thr in [0.25, 0.5]:
        apm = E.compute_map(preds, gts, nc, torch.tensor([thr]))
        valid = ~torch.isnan(apm[:, 0])
        mAP = float(apm[:, 0][valid].mean()) if valid.any() else float('nan')
        print(f"mAP@{thr} = {mAP:.4f}")
    # @0.5:0.95
    ap95 = E.compute_map(preds, gts, nc, torch.linspace(0.5, 0.95, 10))
    valid = ~torch.isnan(ap95)
    print(f"mAP@0.5:0.95 = {float(ap95[valid].mean()):.4f}")
    # 每类 AP@0.5
    ap50 = E.compute_map(preds, gts, nc, torch.tensor([0.5]))
    print("Per-class AP@0.5:")
    for c in range(nc):
        v = ap50[c, 0]
        print(f"  {args.classes[c]:16s}: {'nan' if torch.isnan(v) else f'{float(v):.4f}'}")

if __name__ == '__main__':
    main()
