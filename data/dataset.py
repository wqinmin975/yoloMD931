"""
数据集: VisDrone 有标注 + 无标注航拍图像
- VisDroneLabeled: 加载 VisDrone 标注 (YOLO 格式 txt), letterbox 保持长宽比
- VisDroneUnlabeled: 只加载图像, 用于半监督 / MAE 预训练
- ValDataset: 通用验证集 (任意数据集, 给定类别表), 供跨数据集泛化评估
"""
import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from models.box_utils import letterbox


class VisDroneLabeled(Dataset):
    """VisDrone2019 有标注数据集 (YOLO 格式)"""
    CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']

    def __init__(self, img_dir, label_dir, img_size=640, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.augment = augment
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        # 读取标注 (YOLO 格式: class cx cy w h, 归一化)
        label_path = os.path.join(self.label_dir,
                                  os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c, cx, cy, w, h = map(float, parts[:5])
                    x1 = (cx - w / 2) * w0
                    y1 = (cy - h / 2) * h0
                    x2 = (cx + w / 2) * w0
                    y2 = (cy + h / 2) * h0
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c))

        # Letterbox: 保持长宽比 resize + 灰边填充
        img, scale, pad_x, pad_y = letterbox(img, self.img_size)
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32) * scale
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        # 增强
        if self.augment:
            img, boxes = self._augment(img, boxes)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        return img, {'boxes': boxes, 'labels': labels, 'path': img_path}

    def _augment(self, img, boxes):
        # 随机水平翻转
        if random.random() < 0.5:
            img = img[:, ::-1, :].copy()
            if boxes.numel() > 0:
                w = img.shape[1]
                boxes[:, 0], boxes[:, 2] = w - boxes[:, 2], w - boxes[:, 0]
        # 随机亮度
        if random.random() < 0.5:
            alpha = random.uniform(0.7, 1.3)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        return img, boxes


class VisDroneUnlabeled(Dataset):
    """无标注航拍图像 (用于半监督 / MAE 预训练)"""
    def __init__(self, img_dir, img_size=640):
        self.img_dir = img_dir
        self.img_size = img_size
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, _, _, _ = letterbox(img, self.img_size)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        return img, img_path


class ValDataset(Dataset):
    """通用验证 / 测试集: 加载 YOLO 格式标注, 给定类别表。
    用于跨数据集泛化评估 (VisDrone / UAVDT 等)。"""
    def __init__(self, img_dir, label_dir, classes, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.classes = classes
        self.img_size = img_size
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                            if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]
        img, scale, pad_x, pad_y = letterbox(img, self.img_size)

        label_path = os.path.join(self.label_dir,
                                  os.path.basename(img_path).rsplit('.', 1)[0] + '.txt')
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    c, cx, cy, w, h = map(float, parts[:5])
                    x1 = (cx - w / 2) * w0 * scale + pad_x
                    y1 = (cy - h / 2) * h0 * scale + pad_y
                    x2 = (cx + w / 2) * w0 * scale + pad_x
                    y2 = (cy + h / 2) * h0 * scale + pad_y
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(c))
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
        return img, {'boxes': boxes, 'labels': labels, 'path': img_path}


def collate_flex(batch):
    """
    自定义 collate: 图像 stack 成 (B,3,H,W); 第二元素保持为 list
    (标注是变长 boxes, 不能 stack; 无标注则是路径字符串 list)。
    供 VisDroneLabeled / VisDroneUnlabeled 的 DataLoader 使用。
    """
    imgs = torch.stack([item[0] for item in batch], dim=0)
    rest = [item[1] for item in batch]
    return imgs, rest


if __name__ == "__main__":
    print("VisDrone classes:", VisDroneLabeled.CLASSES)
