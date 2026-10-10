"""
SoftTeacher 半监督训练 MD-YOLO
- Student: MD-YOLO (用 MAE 预训练 backbone 初始化)
- Teacher: Student 的 EMA 影子模型
- 有标注数据: 正常监督损失
- 无标注数据: teacher 出伪标签 -> student 在伪标签上算无监督损失
- P2 分支伪标签阈值更严格
用法:
    python train_softteacher.py --config configs/semi_config.yaml
    # 或直接传参:
    python train_softteacher.py \
        --labeled_img /path/to/VisDrone2019-DET-train/images \
        --labeled_lbl /path/to/VisDrone2019-DET-train/labels \
        --unlabeled_dir /path/to/extra/unlabeled/images
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.md_yolo import MDYOLO
from models.ema import EMATeacher
from models.pseudo_label import filter_pseudo_labels
from models.detection_loss import DetectionLoss
from data.dataset import VisDroneLabeled, VisDroneUnlabeled, collate_flex
from utils.config import apply_config_to_args


# ============== SoftTeacher 训练器 ==============
class SoftTeacher:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. Student
        self.student = MDYOLO(num_classes=args.num_classes).to(self.device)
        # 加载 MAE 预训练 backbone
        if args.mae_ckpt and os.path.exists(args.mae_ckpt):
            print(f"[SoftTeacher] Loading MAE backbone from {args.mae_ckpt}")
            ckpt = torch.load(args.mae_ckpt, map_location=self.device)
            backbone_sd = ckpt.get('backbone', ckpt)
            missing, unexpected = self.student.load_state_dict(backbone_sd, strict=False)
            print(f"  missing keys (head/neck 随机初始化): {len(missing)}, "
                  f"unexpected: {len(unexpected)}")

        # 检测损失 (anchor-free, 自包含可训练)
        self.criterion = DetectionLoss(strides=self.student.head_strides,
                                      img_size=args.img_size, nc=args.num_classes)

        # 2. Teacher (EMA)
        self.teacher = EMATeacher(self.student,
                                  alpha_init=args.ema_init,
                                  alpha_final=args.ema_final,
                                  warmup_steps=args.ema_warmup,
                                  device=self.device)

        # 3. 优化器 (只优化 student)
        wd = getattr(args, 'weight_decay', args.wd)
        self.optimizer = SGD(self.student.parameters(), lr=args.lr,
                            momentum=0.937, weight_decay=wd)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        # 4. 数据
        self.labeled_loader = self._build_labeled_loader()
        self.unlabeled_loader = self._build_unlabeled_loader()

    def _build_labeled_loader(self):
        ds = VisDroneLabeled(self.args.labeled_img, self.args.labeled_lbl,
                             img_size=self.args.img_size, augment=True)
        return DataLoader(ds, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.workers, pin_memory=True,
                          drop_last=True, collate_fn=collate_flex)

    def _build_unlabeled_loader(self):
        ds = VisDroneUnlabeled(self.args.unlabeled_dir, img_size=self.args.img_size)
        return DataLoader(ds, batch_size=self.args.batch_size, shuffle=True,
                          num_workers=self.args.workers, pin_memory=True,
                          drop_last=True, collate_fn=collate_flex)

    def train_one_epoch(self, epoch):
        self.student.train()
        self.teacher.teacher.eval()

        # 无限迭代无标注 loader
        unlabel_iter = iter(self.unlabeled_loader)

        total_sup_loss = 0.0
        total_unsup_loss = 0.0
        for i, (imgs_l, targets_l) in enumerate(self.labeled_loader):
            imgs_l = imgs_l.to(self.device, non_blocking=True)

            # ---- 有标注分支: student 监督损失 ----
            preds_l = self.student(imgs_l)
            loss_sup = self.criterion(preds_l, targets_l)

            # ---- 无标注分支: teacher 出伪标签 ----
            try:
                imgs_u, _ = next(unlabel_iter)
            except StopIteration:
                unlabel_iter = iter(self.unlabeled_loader)
                imgs_u, _ = next(unlabel_iter)
            imgs_u = imgs_u.to(self.device, non_blocking=True)

            with torch.no_grad():
                preds_u_teacher = self.teacher.predict(imgs_u)
                pseudo = filter_pseudo_labels(
                    preds_u_teacher,
                    conf_thr=self.args.pseudo_conf,
                    min_box_size=self.args.min_box_size,
                    iou_thr=self.args.pseudo_nms,
                    per_head_thr=[0.6, 0.5, 0.4, 0.4],  # P2 最严格
                )

            # student 在无标注图上出预测, 用伪标签算无监督损失
            preds_u_student = self.student(imgs_u)
            # pseudo 已是 list[dict{'boxes','labels'}] 格式, 可直接作为无监督目标
            loss_unsup = self.criterion(preds_u_student, pseudo)
            loss_unsup = loss_unsup * self.args.unsup_w

            # ---- 总损失 ----
            loss = loss_sup + loss_unsup
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # EMA 更新 teacher
            global_step = (epoch - 1) * len(self.labeled_loader) + i
            self.teacher.update(global_step)

            total_sup_loss += float(loss_sup)
            total_unsup_loss += float(loss_unsup)

            if i % 20 == 0:
                print(f"  E{epoch} | it{i}/{len(self.labeled_loader)} | "
                      f"sup={loss_sup:.3f} unsup={loss_unsup:.3f} "
                      f"alpha={self.teacher._cosine_alpha(global_step):.4f}")

        return total_sup_loss / len(self.labeled_loader), \
               total_unsup_loss / len(self.labeled_loader)

    def save(self, epoch, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'student': self.student.state_dict(),
            'teacher': self.teacher.teacher.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)
        print(f"[SoftTeacher] saved -> {path}")

    def fit(self):
        print(f"[SoftTeacher] start training, device={self.device}")
        for epoch in range(1, self.args.epochs + 1):
            sup, unsup = self.train_one_epoch(epoch)
            self.scheduler.step()
            print(f"Epoch {epoch}/{self.args.epochs} | sup={sup:.4f} unsup={unsup:.4f}")
            if epoch % self.args.save_interval == 0:
                self.save(epoch, os.path.join(self.args.save_dir, f'ckpt_e{epoch}.pth'))
        # 最终
        self.save(self.args.epochs, os.path.join(self.args.save_dir, 'final.pth'))


def main(args):
    trainer = SoftTeacher(args)
    trainer.fit()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument('--labeled_img', type=str, required=True, help='VisDrone 有标注图像目录')
    p.add_argument('--labeled_lbl', type=str, required=True, help='VisDrone 标注目录 (YOLO txt)')
    p.add_argument('--unlabeled_dir', type=str, required=True, help='无标注航拍图像目录')
    p.add_argument('--mae_ckpt', type=str, default='./checkpoints/mae/mae_backbone_best.pth')
    p.add_argument('--save_dir', type=str, default='./checkpoints/softteacher')
    # 模型
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--img_size', type=int, default=640)
    # EMA
    p.add_argument('--ema_init', type=float, default=0.999)
    p.add_argument('--ema_final', type=float, default=0.9999)
    p.add_argument('--ema_warmup', type=int, default=2000)
    # 伪标签
    p.add_argument('--pseudo_conf', type=float, default=0.5)
    p.add_argument('--pseudo_nms', type=float, default=0.7)
    p.add_argument('--min_box_size', type=int, default=4)
    p.add_argument('--unsup_w', type=float, default=1.0, help='无监督损失权重')
    # 训练
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--save_interval', type=int, default=20)
    p.add_argument('--config', type=str, default=None, help='YAML 配置 (configs/semi_config.yaml)')
    args = p.parse_args()
    apply_config_to_args(args, args.config)
    main(args)
