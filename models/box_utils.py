"""
边界框与评估工具 (anchor-free, 与 MD-YOLO 解码头对齐)
- make_anchor_grid: 为每个检测头生成像素级网格中心
- decode_predictions: 把模型输出解码为 xyxy 框 + 置信度 + 类别
- ciou_elementwise: 逐元素 CIoU
- batched_nms / iou_matrix: NMS 与 IoU 计算
- letterbox: 保持长宽比的 resize + 灰边填充
"""
import math
import torch
import torch.nn.functional as F


def make_anchor_grid(strides, img_size):
    """
    为每个 stride 生成网格中心坐标 (像素)。
    返回 list, 每个元素 = (gx, gy, stride, grid_size)
      gx, gy: (grid_size, grid_size) 的像素中心坐标张量
    """
    grids = []
    for s in strides:
        n = img_size // s
        xs = torch.arange(n, dtype=torch.float32) * s + 0.5 * s
        ys = torch.arange(n, dtype=torch.float32) * s + 0.5 * s
        gx, gy = torch.meshgrid(xs, ys, indexing='ij')
        grids.append((gx, gy, s, n))
    return grids


def _decode_one_head(cls, reg, gx, gy, s, n):
    """
    解码单个检测头:
      cls: (nc, H, W) 分类 logits
      reg: (4, H, W) 回归预测 (tx,ty,tw,th, 单位为 stride)
    返回:
      boxes: (H*W, 4) xyxy 像素
      scores: (H*W,) 各类最大置信度
      labels: (H*W,) 类别索引
    """
    device = cls.device
    gx = gx.to(device)
    gy = gy.to(device)
    tx = reg[0]; ty = reg[1]; tw = reg[2]; th = reg[3]
    # gx,gy 已是像素中心; offset 单位为 stride, 故 px = gx + offset * s (不要再乘 s)
    px = gx + (2.0 * tx.sigmoid() - 1.0) * s
    py = gy + (2.0 * ty.sigmoid() - 1.0) * s
    pw = (2.0 * tw.sigmoid()) ** 2 * s
    ph = (2.0 * th.sigmoid()) ** 2 * s
    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
    scores, labels = cls.sigmoid().reshape(n * n, -1).max(dim=-1)
    return boxes, scores, labels


def decode_predictions(heads_preds, strides, img_size, conf_thr=0.25, iou_thr=0.7, max_det=300):
    """
    把模型四个头的输出解码 + 跨头 NMS。
    heads_preds: list[(cls(B,nc,H,W), reg(B,4,H,W))]
    返回: list[ dict{'boxes':(N,4) xyxy, 'scores':(N,), 'labels':(N,)} ]  (B 个)
    """
    grids = make_anchor_grid(strides, img_size)
    device = heads_preds[0][0].device
    B = heads_preds[0][0].shape[0]
    results = []
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for hi, (cls, reg) in enumerate(heads_preds):
            gx, gy, s, n = grids[hi]
            boxes, scores, labels = _decode_one_head(cls[b], reg[b], gx, gy, s, n)
            m = scores > conf_thr
            all_boxes.append(boxes[m])
            all_scores.append(scores[m])
            all_labels.append(labels[m])
        if all_boxes:
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)
        else:
            boxes = torch.zeros((0, 4), device=device)
            scores = torch.zeros(0, device=device)
            labels = torch.zeros(0, dtype=torch.long, device=device)
        if labels.numel() > 0:
            keep = batched_nms(boxes, scores, labels, iou_thr)
            boxes, scores, labels = boxes[keep][:max_det], scores[keep][:max_det], labels[keep][:max_det]
        results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
    return results


def ciou_elementwise(box1, box2):
    """
    逐元素 CIoU。box1, box2: (N, 4) xyxy。返回 (N,) CIoU。
    """
    x1 = torch.max(box1[:, 0], box2[:, 0])
    y1 = torch.max(box1[:, 1], box2[:, 1])
    x2 = torch.min(box1[:, 2], box2[:, 2])
    y2 = torch.min(box1[:, 3], box2[:, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    union = area1 + area2 - inter + 1e-9
    iou = inter / union

    cx1 = (box1[:, 0] + box1[:, 2]) / 2.0
    cy1 = (box1[:, 1] + box1[:, 3]) / 2.0
    cx2 = (box2[:, 0] + box2[:, 2]) / 2.0
    cy2 = (box2[:, 1] + box2[:, 3]) / 2.0
    d2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    cw = torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0])
    ch = torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1])
    c2 = cw ** 2 + ch ** 2 + 1e-9
    rho2 = d2 / c2

    w1 = (box1[:, 2] - box1[:, 0]).clamp(min=1e-6)
    h1 = (box1[:, 3] - box1[:, 1]).clamp(min=1e-6)
    w2 = (box2[:, 2] - box2[:, 0]).clamp(min=1e-6)
    h2 = (box2[:, 3] - box2[:, 1]).clamp(min=1e-6)
    v = (4.0 / (math.pi ** 2)) * ((torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + 1e-9)
    return iou - rho2 - alpha * v


def batched_nms(boxes, scores, labels, iou_thr=0.7):
    """按类别分组的 NMS；空输入安全返回空索引。"""
    from torchvision.ops import batched_nms as _bnms
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return _bnms(boxes, scores, labels, iou_thr)


def iou_matrix(boxes_a, boxes_b):
    """计算 (Na, Nb) IoU 矩阵。"""
    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return torch.zeros((boxes_a.shape[0], boxes_b.shape[0]), device=boxes_a.device)
    x1 = torch.max(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = torch.max(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = torch.min(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = torch.min(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def letterbox(img, img_size, fill_value=114):
    """
    保持长宽比的 resize + 灰边填充 (cv2/numpy 实现)。
    img: (H, W, 3) uint8 RGB。返回 (img_size, img_size, 3) uint8, 以及 (scale, pad_x, pad_y)。
    """
    import cv2
    import numpy as np
    h0, w0 = img.shape[:2]
    scale = min(img_size / h0, img_size / w0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((img_size, img_size, 3), fill_value, dtype=img_resized.dtype)
    pad_x = (img_size - new_w) // 2
    pad_y = (img_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_resized
    return canvas, scale, pad_x, pad_y
