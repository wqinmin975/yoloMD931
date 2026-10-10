"""
MD-YOLO 检测损失 (anchor-free, YOLOv11 风格)
============================================
本模块实现文档"完整解决方案"第二部分所需的训练组件补齐:

1. 正样本分配: TaskAlignedAssigner (TAA)
   - 跨四个检测头 (P2/P3/P4/P5) 的所有 anchor 统一考虑,
     依据 alignment metric = cls_score^alpha * IoU^beta 为每个 GT 选取 top-k 锚点。
   - 取代原先"仅中心半径 2 网格为正样本"的朴素规则, 从根本上缓解
     密集小目标正样本不足 + 海量背景负样本压制分类梯度的问题。

2. 分类损失: VariFocal Loss (VFL)
   - 正样本按 IoU 质量加权 (target = IoU), 负样本按 focal 项 focally 降权,
     抑制简单背景负样本, 强化稀疏小目标正样本梯度。

3. 回归损失: WIoU v1 (可选 CIoU)
   - 以包围框距离代价调制 IoU, 对小目标/遮挡框更鲁棒。

训练与推理解码公式保持一致: px = gx + (2*sigmoid(tx)-1) * s。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .box_utils import make_anchor_grid, ciou_elementwise, iou_matrix


class DetectionLoss(nn.Module):
    def __init__(self, strides=(4, 8, 16, 32), img_size=640, nc=10,
                 box_gain=7.5, cls_gain=0.5,
                 assign_topk=10, assign_alpha=0.5, assign_beta=6.0,
                 box_loss_type='wiou', vfl_alpha=0.75, vfl_gamma=2.0):
        super().__init__()
        self.strides = tuple(strides)
        self.img_size = img_size
        self.nc = nc
        self.box_gain = box_gain
        self.cls_gain = cls_gain
        self.topk = int(assign_topk)
        self.alpha = float(assign_alpha)
        self.beta = float(assign_beta)
        self.box_loss_type = box_loss_type
        self.vfl_alpha = float(vfl_alpha)
        self.vfl_gamma = float(vfl_gamma)
        self.grids = make_anchor_grid(self.strides, img_size)

    @staticmethod
    def _decode_head(cls_map, reg_map, gx, gy, s, n):
        """解码单个头的所有 anchor 为像素 xyxy 框。与 box_utils._decode_one_head 一致。"""
        device = cls_map.device
        gx = gx.to(device)
        gy = gy.to(device)
        tx, ty, tw, th = reg_map[0], reg_map[1], reg_map[2], reg_map[3]
        px = gx + (2.0 * tx.sigmoid() - 1.0) * s
        py = gy + (2.0 * ty.sigmoid() - 1.0) * s
        pw = (2.0 * tw.sigmoid()) ** 2 * s
        ph = (2.0 * th.sigmoid()) ** 2 * s
        boxes = torch.stack([px - pw / 2.0, py - ph / 2.0,
                             px + pw / 2.0, py + ph / 2.0], dim=-1).reshape(-1, 4)
        return boxes

    @staticmethod
    def _wiou(box1, box2, gamma=2.0):
        """WIoU v1: LIoU = exp(-(dx^gamma+dy^gamma)/(Wg^gamma+Hg^gamma)) * IoU。返回 (N,) LIoU。"""
        x1 = torch.max(box1[:, 0], box2[:, 0])
        y1 = torch.max(box1[:, 1], box2[:, 1])
        x2 = torch.min(box1[:, 2], box2[:, 2])
        y2 = torch.min(box1[:, 3], box2[:, 3])
        inter = torch.clamp(x2 - x1, 0) * torch.clamp(y2 - y1, 0)
        area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
        area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
        union = area1 + area2 - inter + 1e-9
        iou = inter / union

        cx1 = (box1[:, 0] + box1[:, 2]) / 2.0
        cy1 = (box1[:, 1] + box1[:, 3]) / 2.0
        cx2 = (box2[:, 0] + box2[:, 2]) / 2.0
        cy2 = (box2[:, 1] + box2[:, 3]) / 2.0
        dx = torch.abs(cx1 - cx2)
        dy = torch.abs(cy1 - cy2)
        Wg = torch.max(box1[:, 2], box2[:, 2]) - torch.min(box1[:, 0], box2[:, 0])
        Hg = torch.max(box1[:, 3], box2[:, 3]) - torch.min(box1[:, 1], box2[:, 1])
        rwiou = torch.exp(-(dx.pow(gamma) + dy.pow(gamma)) /
                          (Wg.pow(gamma) + Hg.pow(gamma) + 1e-9))
        return rwiou * iou

    def _vfl(self, pred_logits, target):
        """VariFocal Loss (VarifocalNet, 归一化在 forward 中完成)。
        target: (Na, nc) IoU-aware (正样本位置为 IoU, 其余为 0)。
        正样本权重 = target(IoU 质量); 负样本权重 = p^gamma (focally 抑制易分背景)。"""
        pred = pred_logits.sigmoid()
        pos = (target > 0).float()
        focal = torch.where(pos > 0, target, pred.pow(self.vfl_gamma))
        bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
        return (self.vfl_alpha * focal * bce).sum()

    def forward(self, preds, targets):
        """
        preds:   list[(cls(B,nc,H,W), reg(B,4,H,W))]  四个头
        targets: list[ dict{'boxes':(M,4) xyxy 像素, 'labels':(M,)} ]
        返回: 标量损失 (含梯度)
        """
        device = preds[0][0].device
        B = preds[0][0].shape[0]

        cls_loss_sum = torch.zeros((), device=device)
        box_loss_sum = torch.zeros((), device=device)
        pos_total = 0

        for b in range(B):
            tbox = targets[b].get('boxes', torch.zeros((0, 4)))
            tlab = targets[b].get('labels', torch.zeros((0,), dtype=torch.long))
            if tbox.numel() > 0:
                tbox = tbox.to(device)
                tlab = tlab.to(device)

            # ---- 跨头收集所有 anchor 的预测框与分类 logits ----
            all_boxes, all_cls = [], []
            for hi, (cls, reg) in enumerate(preds):
                gx, gy, s, n = self.grids[hi]
                boxes = self._decode_head(cls[b], reg[b], gx, gy, s, n)   # (HW,4)
                cl = cls[b].reshape(self.nc, -1).T                        # (HW,nc)
                all_boxes.append(boxes)
                all_cls.append(cl)
            anc_boxes = torch.cat(all_boxes, dim=0)                        # (Na,4)
            anc_cls = torch.cat(all_cls, dim=0)                            # (Na,nc)
            Na = anc_boxes.shape[0]

            if tbox.numel() == 0:
                # 无 GT: 全为负样本, VFL 目标全 0
                loss_cls = self._vfl(anc_cls, torch.zeros_like(anc_cls))
                cls_loss_sum += loss_cls / max(Na, 1)
                continue

            # ---- TaskAlignedAssigner: 为每个 GT 选 top-k anchor ----
            ious = iou_matrix(anc_boxes, tbox)                 # (Na, M)
            cls_scores = anc_cls.sigmoid()
            gt_cls_scores = cls_scores[:, tlab]                # (Na, M)
            metric = gt_cls_scores.pow(self.alpha) * ious.pow(self.beta)
            topk = min(self.topk, Na)
            topk_idx = metric.topk(topk, dim=0).indices        # (topk, M)
            pos_mask = torch.zeros(Na, dtype=torch.bool, device=device)
            assign_gt = torch.full((Na,), -1, dtype=torch.long, device=device)
            for m in range(tbox.shape[0]):
                idx = topk_idx[:, m]
                pos_mask[idx] = True
                assign_gt[idx] = m

            # ---- 分类目标 (IoU-aware) ----
            target_full = torch.zeros(Na, self.nc, device=device)
            pos_idx = torch.where(pos_mask)[0]
            if pos_idx.numel() > 0:
                ag = assign_gt[pos_idx]
                target_full[pos_idx, tlab[ag]] = ious[pos_idx, ag].clamp(min=1e-4)
                # ---- 回归损失 (仅正样本) ----
                pb = anc_boxes[pos_idx]
                tb = tbox[ag]
                if self.box_loss_type == 'wiou':
                    liou = self._wiou(pb, tb)
                else:
                    liou = ciou_elementwise(pb, tb)
                box_loss_sum += (1.0 - liou).sum()
                pos_total += int(pos_idx.numel())

            loss_cls = self._vfl(anc_cls, target_full)
            denom = int(pos_idx.numel()) if pos_idx.numel() > 0 else Na
            cls_loss_sum += loss_cls / max(denom, 1)

        box_loss = box_loss_sum / max(pos_total, 1)
        cls_loss = cls_loss_sum / max(B, 1)
        loss = self.box_gain * box_loss + self.cls_gain * cls_loss
        return loss


if __name__ == "__main__":
    loss_fn = DetectionLoss(strides=(4, 8, 16, 32), img_size=640, nc=10, box_loss_type='wiou')
    preds = [
        (torch.randn(2, 10, 160, 160), torch.randn(2, 4, 160, 160)),
        (torch.randn(2, 10, 80, 80), torch.randn(2, 4, 80, 80)),
        (torch.randn(2, 10, 40, 40), torch.randn(2, 4, 40, 40)),
        (torch.randn(2, 10, 20, 20), torch.randn(2, 4, 20, 20)),
    ]
    targets = [
        {'boxes': torch.tensor([[100., 100., 140., 140.]]), 'labels': torch.tensor([0])},
        {'boxes': torch.zeros((0, 4)), 'labels': torch.zeros((0,), dtype=torch.long)},
    ]
    loss = loss_fn(preds, targets)
    loss.backward()
    print(f"DetectionLoss(TAA+VFL+WIoU) = {loss.item():.4f} (requires_grad={loss.requires_grad})")
