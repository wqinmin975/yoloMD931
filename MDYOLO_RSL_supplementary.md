# Supplementary Material: An Empirical Reproducibility Audit of MD-YOLO for Small-Object Detection in UAV Imagery

Wang Qinmin¹, Wang Hongyuan¹, Li Peipei¹, and Wang Zongzhong²,*

¹ School of Computer Science and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan, China
² School of Electromechanical Engineering, Guangdong University of Technology, Guangzhou, China

This Supplementary Material supports the Research Letter and contains three items referenced from the main text: (i) the complete runtime-defect table (Table S1), (ii) pseudocode for the MAC backbone and DAFA neck, and (iii) full reproducibility commands.

## Table S1. Runtime defects exposed by real-data execution

| Defect | Symptom | Fix |
| --- | --- | --- |
| Collate crash | `DataLoader` cannot stack variable-length boxes | `collate_flex` keeps targets as a list |
| `letterbox` unpacking | four-value return unpacked as three | `img, _, _, _ = letterbox(...)` |
| Double-stride decode (critical) | `p_x = (g_x + offset)·s` pushes boxes out of bounds | `p_x = g_x + offset·s` in all three modules |
| Empty class-map filter | in-domain eval discards all detections | guard with `if labels.numel()>0 and map_src:` |

*Table S1. The four runtime defects uncovered by executing the pipeline on real data, with the critical double-stride decode error responsible for mAP = 0.*

## Software and hardware environment

| Item | Specification |
| --- | --- |
| OS | Windows 10, x86-64 |
| Python | 3.13.14 (CPython) |
| PyTorch / torchvision | 2.14.0+cpu / 0.29.0+cpu (CPU-only, no CUDA) |
| CPU | Intel 10th-gen Core (Comet Lake), 12 logical cores |
| Key dependencies | NumPy 2.5.2, Matplotlib 3.11.1, Pillow 12.3.0 |

All reported experiments (MAE pretraining, supervised training, remediation training, SoftTeacher pipeline check, in-domain and cross-domain diagnostics) ran on this single CPU-only machine; no GPU was used at any stage.

## MAC backbone and DAFA neck pseudocode

```
# MAC: Multi-scale Adaptive Convolution (downsampling block)
def mac_block(x, expand=2):
    b, c, h, w = x.shape
    # adaptive kernel selection via global context
    att = sigmoid(conv1x1(global_avg_pool(x)))      # channel attention
    k = 3 + 2 * round(att.mean() * 1.0)             # 3 or 5
    y = depthwise_conv(x, kernel_size=k) * att
    y = conv1x1(y)
    return y + downsample(x, stride=2)               # stride 4/8/16/32

# DAFA: Dual-Attention Feature Aggregation neck
def dafa_neck(feats):                                # feats: [P2,P3,P4,P5]
    top_down = fpn_top_down(feats)                   # path a: semantic
    bot_up   = pan_bottom_up(feats)                  # path b: localization
    fused = []
    for a, b in zip(top_down, bot_up):
        sa = spatial_attention(a)                    # spatial attention
        ca = channel_attention(b)                    # channel attention
        fused.append(a * sa + b * ca)
    return detect_heads(fused)                        # P2-P5 heads
```

## Full reproducibility commands

```
# VisDrone preparation & supervised training (250-image subset, 320 px)
# NOTE: the BASELINE arm (Tables 2-3) uses the pre-remediation loss module
# (single-center assignment, sigmoid BCE, CIoU); the command below reproduces
# the REMEDIATION arm (TAA + VFL + WIoU).
python tools/prepare_visdrone.py <visdrone_root> <out_root>
python tools/train_sup.py --images <train/images> --labels <train/labels> \
    --img_size 320 --epochs 40 --subset 250 --opt adamw --sched cosine \
    --warmup 3 --lr 0.001 --box_loss wiou --assign_topk 10
python tools/eval_report.py --weights ckpt_sup_v2/sup_last.pth \
    --val_img <val/images> --val_lbl <val/labels> --img_size 320 --conf 0.05

# UAVDT-M zero-shot cross-domain evaluation (single-class car)
python tools/prepare_uavdt.py --raw_gt <uavdt_gt> --img_root <uavdt_frames> --out <uavdt_test>
python tools/eval_cross_domain.py --weights ckpt_sup_v2/sup_last.pth --mode zero_shot \
    --val_img <uavdt_test/images> --val_lbl <uavdt_test/labels> \
    --classes car --map_src "3:0,4:0" --source_nc 10 --img_size 320 --conf 0.05
```

All experiments were executed on CPU-only hardware; the reduced configuration (250 images, 320 px, 40 epochs) is a reproducibility-audit regime and must not be read as the method's accuracy ceiling.
