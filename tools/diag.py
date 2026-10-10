"""
实测诊断: 验证 mAP=0 的根因。
- 统计每图通过不同 conf 阈值的检测数
- 统计分数分布 (max / 中位数)
- 在极低 conf=0.001 下, 计算预测框与 GT 的最佳 IoU 分布, 看模型是否"能定位但分类被压垮"
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
    ap.add_argument('--subset', type=int, default=120)
    ap.add_argument('--classes', nargs='+',
                   default=['pedestrian','people','bicycle','car','van','truck',
                            'tricycle','awning-tricycle','bus','motor'])
    args = ap.parse_args()
    dev = torch.device('cpu'); nc = len(args.classes)
    m = MDYOLO(num_classes=nc)
    sd = torch.load(args.weights, map_location='cpu')
    m.load_state_dict(sd.get('model', sd.get('student', sd)), strict=False)
    m.eval()
    ds = ValDataset(args.val_img, args.val_lbl, args.classes, img_size=args.img_size)
    n = min(args.subset, len(ds))
    ds = torch.utils.data.Subset(ds, list(range(n)))

    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: b)
    cnt05 = cnt001 = 0
    max_scores = []
    best_iou_at001 = []   # 每图: 在 conf=0.001 下, 任一预测与任一 GT 的最佳 IoU
    total_pred_boxes_001 = 0
    imgs_with_pred_001 = 0
    for sample in loader:
        img, target = sample[0]
        img = img.unsqueeze(0)
        out = m(img)
        # 高阈值 0.05
        d05 = decode_predictions(out, m.head_strides, args.img_size, conf_thr=0.05, iou_thr=0.7)[0]
        # 低阈值 0.001
        d001 = decode_predictions(out, m.head_strides, args.img_size, conf_thr=0.001, iou_thr=0.7)[0]
        cnt05 += int(d05['scores'].numel())
        cnt001 += int(d001['scores'].numel())
        if d001['scores'].numel() > 0:
            max_scores.append(float(d001['scores'].max()))
            imgs_with_pred_001 += 1
            total_pred_boxes_001 += int(d001['scores'].numel())
            if target['boxes'].numel() > 0 and d001['boxes'].numel() > 0:
                # iou_vector 是单框 vs M; 这里逐预测取最大
                best = 0.0
                for bx in d001['boxes']:
                    ii = E.iou_vector(bx, target['boxes'])
                    best = max(best, float(ii.max()) if ii.numel()>0 else 0.0)
                best_iou_at001.append(best)
    print(f"=== 子集 {n} 张图 ===")
    print(f"conf=0.05 检测总数={cnt05}  有检测图占比(0.05)={ (1 if cnt05>0 else 0) }")
    print(f"conf=0.001 检测总数={cnt001}  平均每图={cnt001/n:.1f}  有预测的图={imgs_with_pred_001}/{n}")
    if max_scores:
        import statistics
        print(f"conf=0.001 下 max score: 均值={statistics.mean(max_scores):.4f} 最小={min(max_scores):.4f} 最大={max(max_scores):.4f}")
    if best_iou_at001:
        import statistics
        print(f"conf=0.001 下 每图最佳IoU 均值={statistics.mean(best_iou_at001):.4f} 最大={max(best_iou_at001):.4f}")
        print(f"  最佳IoU>=0.25 的图数={sum(1 for x in best_iou_at001 if x>=0.25)}/{len(best_iou_at001)}")
        print(f"  最佳IoU>=0.5  的图数={sum(1 for x in best_iou_at001 if x>=0.5)}/{len(best_iou_at001)}")

if __name__ == '__main__':
    main()
