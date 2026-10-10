"""
伪标签过滤策略 (针对航拍小目标)
- 用与模型解码头一致的 anchor-free 解码, 避免原先 "直接取 reg 前 4 维当框" 的错误
- 置信度自适应阈值 (P2 分支更严格)
- 小目标框面积过滤
- 分头 NMS + 跨头 NMS 去重
"""
import torch
from .box_utils import batched_nms


def filter_pseudo_labels(preds, conf_thr=0.5, min_box_size=4,
                         iou_thr=0.7, per_head_thr=None,
                         strides=(4, 8, 16, 32), img_size=640):
    """
    对 teacher 输出的检测结果做过滤, 生成高质量伪标签。

    输入:
      preds: list of (cls_logits, reg_preds) per head, 与 MD-YOLO 输出对齐
             cls_logits: (B, nc, H, W); reg_preds: (B, 4, H, W)
      per_head_thr: 每个 head 单独的置信度阈值 (P2 更严格)
    输出:
      list of dict per image: {boxes: (N,4) xyxy 像素, scores: (N,), labels: (N,)}

    注意: 锚点网格直接由每个 head 的实际张量空间尺寸 (H, W) 推导,
          不再依赖 img_size 参数, 因此对任意输入分辨率 (如 320/640) 都正确。
    """
    if per_head_thr is None:
        per_head_thr = [0.6, 0.5, 0.4, 0.4]  # P2/P3/P4/P5

    B = preds[0][0].shape[0]
    final = []
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for hi, (cls, reg) in enumerate(preds):
            thr = per_head_thr[hi] if hi < len(per_head_thr) else conf_thr
            s = strides[hi]
            c = cls[b]
            r = reg[b]
            H, W = r.shape[-2], r.shape[-1]
            # 由实际特征图尺寸推导像素中心网格 (与解码/推理保持一致)
            gx = torch.arange(W, device=c.device).float() * s + s / 2.0
            gy = torch.arange(H, device=c.device).float() * s + s / 2.0
            gx, gy = torch.meshgrid(gx, gy, indexing='ij')
            n = H * W
            # anchor-free 解码 (gx,gy 已是像素中心, offset 单位为 stride)
            px = gx + (2.0 * r[0].sigmoid() - 1.0) * s
            py = gy + (2.0 * r[1].sigmoid() - 1.0) * s
            pw = (2.0 * r[2].sigmoid()) ** 2 * s
            ph = (2.0 * r[3].sigmoid()) ** 2 * s
            box = torch.stack([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], -1)
            box = box.reshape(-1, 4)
            sc, lb = c.sigmoid().reshape(n, -1).max(dim=-1)

            m = sc > thr
            box, sc, lb = box[m], sc[m], lb[m]
            if box.numel() > 0:
                w = box[:, 2] - box[:, 0]
                h = box[:, 3] - box[:, 1]
                keep = (w >= min_box_size) & (h >= min_box_size)
                box, sc, lb = box[keep], sc[keep], lb[keep]
            if box.numel() > 0:
                keep = batched_nms(box, sc, lb, iou_thr)
                box, sc, lb = box[keep], sc[keep], lb[keep]
            all_boxes.append(box)
            all_scores.append(sc)
            all_labels.append(lb)

        if all_boxes:
            box = torch.cat(all_boxes, dim=0)
            sc = torch.cat(all_scores, dim=0)
            lb = torch.cat(all_labels, dim=0)
            keep = batched_nms(box, sc, lb, iou_thr)
            box, sc, lb = box[keep], sc[keep], lb[keep]
            if box.numel() == 0:
                box, sc, lb = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        else:
            box, sc, lb = torch.zeros((0, 4)), torch.zeros(0), torch.zeros(0, dtype=torch.long)
        final.append({'boxes': box, 'scores': sc, 'labels': lb})
    return final


if __name__ == "__main__":
    # 模拟 teacher 输出 (anchor-free: cls=(B,nc,H,W), reg=(B,4,H,W))
    preds = [
        (torch.randn(2, 10, 160, 160), torch.randn(2, 4, 160, 160)),  # P2
        (torch.randn(2, 10, 80, 80), torch.randn(2, 4, 80, 80)),       # P3
        (torch.randn(2, 10, 40, 40), torch.randn(2, 4, 40, 40)),       # P4
        (torch.randn(2, 10, 20, 20), torch.randn(2, 4, 20, 20)),       # P5
    ]
    out = filter_pseudo_labels(preds)
    print(f"Filtered pseudo-labels per image: {len(out)}")
    print(f"  img0 boxes: {out[0]['boxes'].shape}, img1 boxes: {out[1]['boxes'].shape}")
