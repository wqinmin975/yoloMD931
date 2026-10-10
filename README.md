# MD-YOLO Semi-Supervised 工程代码

**MD-YOLO: Multi-Scale Adaptive & Dual-Attention Enhanced YOLOv11**
+ **MAE 航拍域自监督骨干预训练**
+ **SoftTeacher 半监督训练框架**

## 目录结构
```
D:/ace/source/  (平铺工程根目录; 运行命令均为 `python xxx.py`, 非 `python -m`)
├── models/
│   ├── mac.py              # MAC 多尺度自适应下采样模块
│   ├── dafa.py             # DAFA 双注意力特征融合
│   ├── md_yolo.py          # MD-YOLO 整体模型 (P2/P3/P4/P5 四头, anchor-free)
│   ├── mae_pretrain.py     # MAE 预训练模型 (输入图像 patch 掩码 + decoder)
│   ├── ema.py              # EMA Teacher 影子模型 (cosine warmup)
│   ├── pseudo_label.py     # 伪标签过滤 (anchor-free 解码, P2 分支更严格)
│   ├── detection_loss.py   # anchor-free 检测损失 (CIoU + BCE)
│   └── box_utils.py        # 解码 / CIoU / NMS / letterbox 工具
├── data/
│   └── dataset.py          # VisDrone 有标注/无标注 + 通用 ValDataset (letterbox)
├── configs/
│   ├── mae_config.yaml     # MAE 预训练配置
│   └── semi_config.yaml   # SoftTeacher 半监督配置
├── utils/
│   └── config.py           # 扁平 YAML 配置加载 (零依赖)
├── train_mae.py            # 阶段1: MAE 预训练
├── train_softteacher.py    # 阶段2: 半监督训练
├── eval_cross_domain.py    # 阶段3: 跨数据集泛化评估 (zero-shot / few-shot)
└── README.md
```

## 训练流程 (两阶段)

### 阶段 1: MAE 航拍域骨干预训练
```bash
# 准备无标注航拍图像 (VisDrone 训练帧 + UAVDT 抽帧, 不使用 bbox)
python train_mae.py \
    --unlabeled_dir /path/to/unlabeled/images \
    --save_dir ./checkpoints/mae \
    --batch_size 16 --epochs 800 --mask_ratio 0.75
# 输出: checkpoints/mae/mae_backbone_best.pth
```

### 阶段 2: SoftTeacher 半监督训练
```bash
python train_softteacher.py \
    --labeled_img /path/to/VisDrone2019-DET-train/images \
    --labeled_lbl /path/to/VisDrone2019-DET-train/labels \
    --unlabeled_dir /path/to/extra/unlabeled/images \
    --mae_ckpt ./checkpoints/mae/mae_backbone_best.pth \
    --save_dir ./checkpoints/softteacher \
    --batch_size 16 --epochs 300
```

### 阶段 3: 跨数据集泛化测试 (拓展三)
```bash
# 1) Zero-shot: VisDrone 训练 -> UAVDT 直接测试
python eval_cross_domain.py \
    --weights ./checkpoints/softteacher/final.pth \
    --mode zero_shot \
    --val_img /path/to/UAVDT/images --val_lbl /path/to/UAVDT/labels \
    --classes car truck bus motor --map_src "0:0,1:1,2:2,3:3"

# 2) Few-shot: VisDrone 训练 -> UAVDT 5% 标注微调
python eval_cross_domain.py \
    --weights ./checkpoints/softteacher/final.pth \
    --mode few_shot --finetune_epochs 20 --finetune_frac 0.05 \
    --val_img /path/to/UAVDT/images --val_lbl /path/to/UAVDT/labels \
    --classes car truck bus motor --map_src "0:0,1:1,2:2,3:3"
```
评估指标: COCO 风格 mAP@0.5 与 mAP@0.5:0.95 (IoU 0.5:0.95:0.05)。

## 关键设计要点

| 模块 | 设计选择 | 论文叙事 |
|------|----------|----------|
| MAC 下采样 | 4 分支 + 门控融合, 替代 stride=2 | 保留小目标细粒度信息 |
| DAFA 融合 | 跨尺度对齐 + 通道/局部窗口空间注意力 | 抑制航拍背景干扰 |
| P2 检测头 | 160x160 高分辨率分支 | 专门检测 <32px 小目标 |
| MAE 预训练 | 航拍域内自监督, mask_ratio=0.75 | 替代 ImageNet, 缓解域偏移 |
| EMA Teacher | cosine warmup 0.999→0.9999 | 伪标签稳定 |
| 伪标签过滤 | P2 阈值 0.6, P5 阈值 0.4 | 抑制高分辨率分支噪声 |

## 工程替换建议
`models/detection_loss.py` 已实现自包含、可训练的 anchor-free 检测损失 (CIoU + BCE, 按目标尺寸分配到四个头), 不依赖 ultralytics, 可直接训练与复现。
追求极致精度时仍可替换为 `ultralytics.utils.loss.v8DetectionLoss` (含 TaskAlignedAssigner), 接口保持一致 (输入 `preds` + `targets`)。

## 消融实验建议
1. **MAE  ablation**: 随机初始化 / ImageNet 预训练 / 航拍域 MAE 预训练 三组对比
2. **半监督比例**: 10% / 30% / 50% / 100% 标注 vs 无标注
3. **伪标签阈值**: P2 分支 0.5/0.6/0.7 对比
4. **跨域泛化**: VisDrone→UAVDT zero-shot, mAP 衰减幅度 vs YOLOv11s baseline
