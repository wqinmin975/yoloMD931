"""
诊断2: 在极低 conf=0.001 下算逐类 AP@0.25/@0.5, 并打印预测标签分布,
以区分"定位退化"还是"分类退化"。
"""
import os, sys, torch
from torch.utils.data import DataLoader, Subset
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.md_yolo import MDYOLO
from data.dataset import ValDataset
import eval_cross_domain as E

def main():
    w = "E:/visdrone_work/ckpt_sup/sup_last.pth"
    vi = "E:/visdrone_work/VisDrone2019-DET-val/images"
    vl = "E:/visdrone_work/VisDrone2019-DET-val/labels"
    classes = ['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
    nc = len(classes); img_size = 320; subset = 200
    m = MDYOLO(num_classes=nc)
    sd = torch.load(w, map_location='cpu')
    m.load_state_dict(sd.get('model', sd.get('student', sd)), strict=False)
    m.eval()
    ds = ValDataset(vi, vl, classes, img_size=img_size)
    ds = Subset(ds, list(range(subset)))
    preds, gts = E.collect_predictions(m, ds, {}, img_size, torch.device('cpu'), 0.001, 0.7)
    # 预测标签分布
    from collections import Counter
    c = Counter()
    for p in preds:
        for l in p['labels'].tolist():
            c[int(l)] += 1
    print("预测标签分布 (top6):", c.most_common(6))
    print("总预测数:", sum(c.values()))
    for thr in [0.25, 0.5]:
        apm = E.compute_map(preds, gts, nc, torch.tensor([thr]))
        valid = ~torch.isnan(apm[:,0])
        mAP = float(apm[:,0][valid].mean()) if valid.any() else float('nan')
        print(f"mAP@{thr} (conf=0.001) = {mAP:.4f}")
    ap50 = E.compute_map(preds, gts, nc, torch.tensor([0.5]))
    print("Per-class AP@0.5 (conf=0.001):")
    for i in range(nc):
        v = ap50[i,0]
        print(f"  {classes[i]:16s}: {'nan' if torch.isnan(v) else f'{float(v):.4f}'}")

if __name__ == '__main__':
    main()
