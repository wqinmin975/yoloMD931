# MD-YOLO 程序一致性检验与修复报告（修订版）

> 检验对象：`D:/ace/source` 下全部代码 vs 设计文档 `解决问题.docx`
> 环境：Python 3.13；`torch` 未安装，故**全部为静态审查 + `py_compile` 语法校验**（全通过），未做动态运行。
> 本轮对上一版报告列出的 P0 / P1 / P2 缺陷**逐项落地了代码修复**。

---

## 一、本轮修复清单（对应上一版优先级）

| 优先级 | 缺陷 | 修复动作 | 涉及文件 | 状态 |
|--------|------|----------|----------|------|
| **P0-A** | `detection_loss` 恒为 0 → 半监督不学习 | 新建自包含 **anchor-free 检测损失** `DetectionLoss`（CIoU + BCE，按目标尺寸分配到 P2–P5），接入 `train_softteacher`；伪标签（`boxes/labels`）直接作为无监督目标进入损失 | `models/detection_loss.py`（新）、`models/box_utils.py`（新）、`train_softteacher.py`、`models/md_yolo.py`（检测头改为 anchor-free） | ✅ 已修复 |
| **P0-B / 拓展三** | 跨数据集泛化"必做"未实现 | 新建 `eval_cross_domain.py`：COCO 风格 mAP（IoU 0.5:0.95）+ **zero-shot** 与 **few-shot** 两条路径；`data/dataset.py` 新增通用 `ValDataset` | `eval_cross_domain.py`（新）、`data/dataset.py` | ✅ 已修复 |
| **P0-C** | MAE 在 C5 特征图上掩码（平凡解） | 改为**在输入图像上做 patch 掩码**（SimMIM 风格）：被掩像素置 0 后再送入 backbone，编码器无法看到被掩内容，重建必须靠上下文 | `models/mae_pretrain.py`（重写） | ✅ 已修复 |
| **P1-D** | decoder `pos_embed` 冻结为 0 | 改为**可学习**位置编码（`trunc_normal_` 初始化） | `models/mae_pretrain.py` | ✅ 已修复 |
| **P1-E** | `cv2.resize` 拉伸方形 → 非方图框错位 | 改为 **letterbox**（保持长宽比 + 灰边填充），框按统一 `scale` 缩放并随填充 `pad` 偏移，框/图严格对齐 | `data/dataset.py`、`models/box_utils.py` | ✅ 已修复 |
| **P1-F** | MAE 优化器/ckpt 含未训练 neck/head | 优化器仅含 **backbone + encoder_proj + decoder**；保存只存 `get_backbone_state_dict()` | `train_mae.py`、`models/mae_pretrain.py` | ✅ 已修复 |
| **P2** | EMA "cosine warmup" 实为线性；死代码 | EMA 改为真正的 **cosine** 升温；删除 `SemiConcatDataset`/`semi_collate_fn`/`ConcatDataset` 等未定义/未使用代码，修正 `data/__init__.py` 失效导入 | `models/ema.py`、`data/dataset.py`、`data/__init__.py`、`models/pseudo_label.py`（错误解码一并修正） | ✅ 已修复 |
| **P2/死配置** | `configs/*.yaml` 未接入脚本 | 两个训练脚本新增 `--config` 参数 + 零依赖的扁平 YAML 加载器 `utils/config.py`；修正 docstring/README 中 `python -m md_yolo_semi...` 与实际平铺目录不符的命令 | `utils/config.py`（新）、`train_mae.py`、`train_softteacher.py`、`README.md`、`configs/mae_config.yaml` | ✅ 已修复 |

---

## 二、修复后架构一致性确认

- **网络结构**：MAC（4 分支 + 门控融合）、DAFA（通道 + 局部窗口空间双注意力）、P2/P3/P4/P5 四头、10 类（VisDrone）均与文档一致；检测头改为 anchor-free 与 YOLOv8/v11 一致（文档称"可替换为 YOLOv11 Detect"，此处实现等价简化版）。
- **数据流自洽**：backbone 输出通道 `backbone_channels[-1]=512`，与 MAE `embed_dim=512` 对齐；patch_size=32 与 C5 stride=32 对齐（grid 20×20），掩码/重建维度一致。
- **训练链路闭合**：`图像(letterbox) → backbone → neck(DAFA) → 四头 → DetectionLoss(监督/无监督)` 均可微分；伪标签经 `filter_pseudo_labels`（anchor-free 解码 + 分头阈值）后直接作为 `criterion` 目标；EMA Teacher 由 Student 影子更新。
- **语法校验**：全部 13 个 `.py` 经 `py_compile` 通过；修复了 `data/__init__.py` 因引用已删除符号导致的导入失败。

---

## 三、仍需诚实标注的方法学边界（不可省略）

1. **分配策略是启发式，非 SimOTA**：`DetectionLoss` 用"按目标尺寸选头 + 中心格邻域"作正样本，简单可训练，但**弱于** ultralytics 的 `TaskAlignedAssigner`。论文若要冲精度，应替换为官方 `v8DetectionLoss`（接口已对齐：`(preds, targets)`）。
2. **MAE 是 SimMIM 风格，非标准 patch-embedding MAE**：因 encoder 复用 CNN backbone（需完整特征图），无法做"随机可见 token 送 Transformer"。本修复在**输入图像**上掩码后再过 CNN，是文献中成立的 SimMIM 范式；但若审稿人要求"标准 MAE（ViT encoder）"，需另写 ViT 编码器，不能声称"标准 MAE"。
3. **跨域类映射需人工核对**：`eval_cross_domain.py` 的 `--map_src` 默认映射（VisDrone→UAVDT）仅为示例，需按实际标注语义确认；few-shot 用目标域头重初始化，属常规操作但会改变 head 随机种子。
4. **数值结论不可直接使用**：原文档中的 mAP 提升（如 38.5%→45.6%）**仍未在任何真实实验中被复现/验证**。以上仅为"代码可正确训练/评估"的修复，不等于"方法已验证有效"。

---

## 四、建议的验证命令（需有 torch + 数据）

```bash
# 1) 单元自检（无需数据，验证损失/解码形状与梯度）
python -m models.detection_loss
python -m models.pseudo_label
python -m models.mae_pretrain

# 2) 两阶段训练
python train_mae.py --unlabeled_dir /data/uav_unlabeled --save_dir ./checkpoints/mae
python train_softteacher.py --labeled_img /data/visdrone/images --labeled_lbl /data/visdrone/labels \
    --unlabeled_dir /data/uav_unlabeled --mae_ckpt ./checkpoints/mae/mae_backbone_best.pth

# 3) 跨数据集泛化
python eval_cross_domain.py --weights ./checkpoints/softteacher/final.pth \
    --mode zero_shot --val_img /data/uavdt/images --val_lbl /data/uavdt/labels \
    --classes car truck bus motor --map_src "3:0,4:0,5:1,8:2,9:3"
```

---

## 五、修复核验对照表（逐项 file:line 证据）

针对用户复核清单，**逐项在当前代码中核实**确认修复已落地（核验日期 2026-10-08，仍受限于无 torch 环境，未做动态运行）：

| # | 清单项（偏差 / 缺陷） | 核验结果 | 证据位置 |
|---|----------------------|----------|----------|
| 1 | MAE 在 C5 特征图掩码 → 改为输入图像 patch 掩码 | ✅ 已改 | `models/mae_pretrain.py:104-128`（`forward_encoder` 中 `imgs * (1.0 - mask_img)` 置 0 后送 backbone）；`:130-144` 仅对被掩 patch 算重建损失 |
| 2 | 跨数据集泛化（拓展三）未实现 | ✅ 已实现 | 新增 `eval_cross_domain.py`（zero-shot / few-shot + COCO mAP）；`data/dataset.py:112-159` 通用 `ValDataset` |
| 3 | `configs/*.yaml` 是死配置 | ✅ 已接入 | `utils/config.py`（零依赖扁平加载）；`train_mae.py:96-98`、`train_softteacher.py:195-197` 新增 `--config` 并经 `apply_config_to_args` 合并 |
| 4 | README `python -m md_yolo_semi...` 不符 | ✅ 已修正 | 运行命令改为 `python train_mae.py` / `python train_softteacher.py` / `python eval_cross_domain.py`（`README.md:37/46/58/65`）；目录树去除 `md_yolo_semi/` 包裹（`README.md:9`） |
| 5 | **P0-A**：`detection_loss` 恒为 0 → 不学习 | ✅ 已修复 | 旧占位函数删除；`models/detection_loss.py:17-142` 实现可训练 `DetectionLoss`；`train_softteacher.py:94`（监督）与 `:117`（无监督，伪标签入损失）调用；`box_utils.py` 提供 CIoU/NMS/解码支撑 |
| 6 | **P0-B**：跨域评估脚本缺失 | ✅ 同 #2 | `eval_cross_domain.py` |
| 7 | **P0-C**：MAE 掩码位置错误 | ✅ 同 #1 | `models/mae_pretrain.py` |
| 8 | **P1-D**：decoder `pos_embed` 冻结为 0 | ✅ 已改 | `models/mae_pretrain.py:28`（`nn.Parameter`）+ `:38`（`trunc_normal_` 初始化），`:46` 加进 decoder |
| 9 | **P1-E**：拉伸方形导致框错位 | ✅ 已修复 | `data/dataset.py:56-65`、`126-148` 用 `letterbox`（`box_utils.py:149-164`），框随 `scale`/`pad_x`/`pad_y` 对齐 |
| 10 | **P1-F**：MAE 优化器/ckpt 含 neck/head | ✅ 已改 | `train_mae.py:54-56` 仅 `backbone_parameters()+decoder`；`:73` 保存 `get_backbone_state_dict()`；`mae_pretrain.py:79-86` 定义 `backbone_parameters()`，`md_yolo.py:123-128` 定义 `get_backbone_state_dict()` |
| 11 | **P2**：EMA cosine 实为线性 | ✅ 已改 | `models/ema.py:45-51`（`(1-cos(π·frac))/2` 余弦升温） |
| 12 | **P2**：`SemiConcatDataset` 未定义 / 死代码 | ✅ 已清理 | 全仓 grep `SemiConcatDataset`/`semi_collate_fn`/`ConcatDataset` 已**零命中**；`data/__init__.py` 导入修正为 `ValDataset`；`pseudo_label.py` 重写使用 anchor-free 解码 |

> 残留一致性风险（非缺陷，需如实说明）：`DetectionLoss` 为启发式尺寸分配，弱于 SimOTA/TaskAlignedAssigner；MAE 为 SimMIM 风格（非标准 ViT-MAE）；跨域类映射默认值需按真实语义核对；文档中 mAP 数值仍未被真实实验复现。

---

## 六、结论

上一版报告中的 **P0-A（最严重，训练无效）、P0-B（拓展三未实现）、P0-C（MAE 方法学错误）、P1-D/E/F、P2** 均已在本轮落地为可运行的代码修复，并通过语法校验。**下一步必须在真实数据 + torch 环境下跑通训练与评估，并把分配器升级为 SimOTA 后，方可将性能结论写入论文。**
