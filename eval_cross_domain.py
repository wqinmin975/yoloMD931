"""
跨数据集泛化评估 (拓展三)
- zero_shot: 用源域 (VisDrone) 训练的权重直接在目标域 (UAVDT 等) 测试, 通过类映射对齐
- few_shot:  在目标域 5% 标注上微调 (重新初始化检测头), 再在剩余目标域数据上测试
- 指标: COCO 风格 mAP@0.5 与 mAP@0.5:0.95 (IoU 0.5:0.95:0.05, 101 点插值)
用法:
    python eval_cross_domain.py --weights ./checkpoints/softteacher/final.pth \
        --mode zero_shot --val_img .../images --val_lbl .../labels \
        --classes car truck bus motor --map_src "3:0,4:0,5:1,8:2,9:3"
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from models.md_yolo import MDYOLO
from models.detection_loss import DetectionLoss
from models.box_utils import decode_predictions
from data.dataset import ValDataset


def parse_map(src):
    """'3:0,4:0' -> {3:0, 4:0}"""
    d = {}
    for item in src.split(','):
        if not item.strip():
            continue
        s, t = item.split(':')
        d[int(s)] = int(t)
    return d


def iou_vector(box, g):
    """单框 vs M 个 gt 的 IoU -> (M,)"""
    if g.numel() == 0:
        return torch.zeros(0)
    x1 = torch.max(box[0], g[:, 0])
    y1 = torch.max(box[1], g[:, 1])
    x2 = torch.min(box[2], g[:, 2])
    y2 = torch.min(box[3], g[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    a2 = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
    return inter / (a1 + a2 - inter + 1e-9)


def compute_map(preds, gts, num_classes, iou_thresholds):
    """
    preds: list[img] -> dict{'boxes':(N,4), 'scores':(N,), 'labels':(N,)}
    gts:   list[img] -> dict{'boxes':(M,4), 'labels':(M,)}
    返回: ap_matrix (num_classes, len(thr))  (无 gt 的类为 nan)
    """
    ap_matrix = torch.full((num_classes, len(iou_thresholds)), float('nan'))
    for c in range(num_classes):
        dets = []          # (score, box(4), img_idx)
        gt_by_img = []
        total_gt = 0
        for i in range(len(preds)):
            lb = preds[i]['labels']
            m = lb == c
            if m.any():
                sc = preds[i]['scores'][m]
                bx = preds[i]['boxes'][m]
                for k in range(bx.shape[0]):
                    dets.append((float(sc[k]), bx[k], i))
            gboxes = gts[i]['boxes']
            glb = gts[i]['labels']
            gm = glb == c
            gt_by_img.append(gboxes[gm])
            total_gt += int(gm.sum())
        if total_gt == 0:
            continue
        dets.sort(key=lambda x: -x[0])
        for ti, thr in enumerate(iou_thresholds):
            gt_used = [torch.zeros(g.shape[0], dtype=torch.bool) for g in gt_by_img]
            tp = torch.zeros(len(dets))
            fp = torch.zeros(len(dets))
            for d_idx, (score, box, i) in enumerate(dets):
                g = gt_by_img[i]
                if g.shape[0] == 0:
                    fp[d_idx] = 1.0
                    continue
                iou = iou_vector(box, g)
                best = float(iou.max()) if iou.numel() > 0 else 0.0
                best_j = int(iou.argmax()) if iou.numel() > 0 else -1
                if best >= thr and best_j >= 0 and not gt_used[i][best_j]:
                    gt_used[i][best_j] = True
                    tp[d_idx] = 1.0
                else:
                    fp[d_idx] = 1.0
            cum_tp = torch.cumsum(tp, 0)
            cum_fp = torch.cumsum(fp, 0)
            recalls = cum_tp / total_gt
            precisions = cum_tp / (cum_tp + cum_fp + 1e-9)
            ap = 0.0
            for t in torch.linspace(0, 1, 101):
                p = precisions[recalls >= t].max() if (recalls >= t).any() else torch.tensor(0.0)
                ap += float(p) / 101.0
            ap_matrix[c, ti] = ap
    return ap_matrix


@torch.no_grad()
def collect_predictions(model, dataset, map_src, img_size, device, conf, iou):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                        collate_fn=lambda b: b)
    preds_all, gts_all = [], []
    model.eval()
    for sample in loader:
        img, target = sample[0]
        img = img.unsqueeze(0).to(device) if img.dim() == 3 else img.to(device)
        out = model(img)
        d = decode_predictions(out, model.head_strides, img_size,
                              conf_thr=conf, iou_thr=iou)[0]
        boxes, scores, labels = d['boxes'], d['scores'], d['labels']
        if labels.numel() > 0 and map_src:
            # 仅当提供了跨域类映射时才过滤/重映射; 域内评估 (map_src 为空) 保留全部预测
            keep = torch.tensor([int(l) in map_src for l in labels.tolist()],
                                device=labels.device)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if labels.numel() > 0:
                labels = torch.tensor([map_src[int(l)] for l in labels.tolist()],
                                      device=labels.device, dtype=torch.long)
        preds_all.append({'boxes': boxes.cpu(), 'scores': scores.cpu(),
                          'labels': labels.cpu()})
        # GT 同样按 map_src 过滤/重映射, 使 GT 与预测处于同一目标域标签空间
        gboxes, glabels = target['boxes'].cpu(), target['labels'].cpu()
        if glabels.numel() > 0 and map_src:
            keep_g = torch.tensor([int(l) in map_src for l in glabels.tolist()],
                                  dtype=torch.bool)
            gboxes, glabels = gboxes[keep_g], glabels[keep_g]
            if glabels.numel() > 0:
                glabels = torch.tensor([map_src[int(l)] for l in glabels.tolist()],
                                       dtype=torch.long)
        gts_all.append({'boxes': gboxes, 'labels': glabels})
    return preds_all, gts_all


def load_weights(model, weights, backbone_only):
    ckpt = torch.load(weights, map_location='cpu')
    if 'student' in ckpt:
        sd = ckpt['student']
    elif 'backbone' in ckpt:
        sd = ckpt['backbone']
    elif 'model' in ckpt:
        sd = ckpt['model']
    else:
        sd = ckpt
    if backbone_only:
        from models.md_yolo import MDYOLO as _M
        sd = {k: v for k, v in sd.items()
              if any(k.startswith(p + '.') for p in
                     ['stem', 'down1', 'stage1', 'down2', 'stage2',
                      'down3', 'stage3', 'down4', 'stage4'])}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval] loaded {weights}: missing={len(missing)}, unexpected={len(unexpected)}")


def evaluate(model, dataset, map_src, num_classes, img_size, device, conf, iou):
    preds, gts = collect_predictions(model, dataset, map_src, img_size, device, conf, iou)
    iou_thr = torch.linspace(0.5, 0.95, 10)
    ap = compute_map(preds, gts, num_classes, iou_thr)
    valid = ~torch.isnan(ap)
    mAP_50 = float(ap[:, 0][valid[:, 0]].mean()) if valid[:, 0].any() else float('nan')
    mAP_5095 = float(ap[valid].mean()) if valid.any() else float('nan')
    return mAP_50, mAP_5095, ap


def fine_tune(model, dataset, epochs, img_size, lr, device, nc):
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0,
                        collate_fn=lambda b: b)
    criterion = DetectionLoss(strides=model.head_strides, img_size=img_size, nc=nc)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0
        for sample in loader:
            img, target = sample[0]
            img = img.to(device)
            preds = model(img)
            loss = criterion(preds, [{'boxes': target['boxes'].to(device),
                                     'labels': target['labels'].to(device)}])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        print(f"  finetune ep{ep}/{epochs} loss={tot:.3f}")


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map_src = parse_map(args.map_src) if args.map_src else {}
    num_classes = len(args.classes)
    iou_thr = 0.7

    if args.mode == 'zero_shot':
        # 源域 head (VisDrone 10 类), 预测后按 map_src 映射到目标类
        model = MDYOLO(num_classes=args.source_nc).to(device)
        load_weights(model, args.weights, backbone_only=False)
        ds = ValDataset(args.val_img, args.val_lbl, args.classes, img_size=args.img_size)
        m50, m5095, _ = evaluate(model, ds, map_src, num_classes,
                                 args.img_size, device, args.conf, iou_thr)
        print(f"[zero-shot] mAP@0.5={m50:.4f}  mAP@0.5:0.95={m5095:.4f}")

    elif args.mode == 'few_shot':
        # 目标域 head, 仅加载骨干, 在目标域子集上微调
        model = MDYOLO(num_classes=num_classes).to(device)
        load_weights(model, args.weights, backbone_only=True)
        full_ds = ValDataset(args.val_img, args.val_lbl, args.classes, img_size=args.img_size)
        n = len(full_ds)
        idx = list(range(n))
        import random
        random.seed(0)
        random.shuffle(idx)
        k = max(1, int(n * args.finetune_frac))
        ft_idx, test_idx = idx[:k], idx[k:]
        print(f"[few-shot] finetune={len(ft_idx)} test={len(test_idx)}")
        fine_tune(model, Subset(full_ds, ft_idx), args.finetune_epochs,
                  args.img_size, args.lr, device, num_classes)
        if test_idx:
            test_ds = Subset(full_ds, test_idx)
            m50, m5095, _ = evaluate(model, test_ds, {}, num_classes,
                                     args.img_size, device, args.conf, iou_thr)
            print(f"[few-shot] mAP@0.5={m50:.4f}  mAP@0.5:0.95={m5095:.4f}")
    else:
        raise ValueError(f"未知 mode: {args.mode}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--mode', type=str, default='zero_shot',
                   choices=['zero_shot', 'few_shot'])
    p.add_argument('--val_img', type=str, required=True)
    p.add_argument('--val_lbl', type=str, required=True)
    p.add_argument('--classes', type=str, nargs='+',
                   default=['car', 'truck', 'bus', 'motor'])
    p.add_argument('--map_src', type=str, default='',
                   help='源类->目标类映射, 如 "3:0,4:0,5:1,8:2,9:3"')
    p.add_argument('--source_nc', type=int, default=10, help='源域类别数 (VisDrone=10)')
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.05)
    p.add_argument('--finetune_epochs', type=int, default=20)
    p.add_argument('--finetune_frac', type=float, default=0.05)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--device', type=str, default=None)
    args = p.parse_args()
    main(args)
