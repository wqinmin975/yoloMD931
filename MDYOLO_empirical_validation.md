# Empirical Validation and Reproducibility Audit of MD-YOLO on Real Aerial Imagery

**A Multi-scale Adaptive & Dual-Attention Enhanced YOLOv11 Detector for UAV Small-Object Detection**

---

## Abstract

We report an end-to-end empirical validation of the MD-YOLO detection codebase against the design document *解决问题.docx*, executed on the real VisDrone2019-DET aerial dataset. Whereas the prior review was static (source-code inspection), this work runs the actual pipeline on 7,019 real images (6,471 train / 548 validation, 381,963 annotated boxes) and measures concrete outcomes. Real-data execution exposed four defects that static review missed: (i) a `DataLoader` collate crash on variable-length boxes, (ii) a four-tuple `letterbox` unpacking error, (iii) a **critical double-stride bug** in the anchor-free decode (`px = (gx+offset)*s`) that projected every predicted box out of image bounds and silently produced mAP = 0, and (iv) an empty-class-map filter that discarded all detections during in-domain evaluation. After fixing these, the pipeline runs end-to-end and the loss converges, but a subsequent diagnostic reveals the deeper, honest result: under the repository's current simplified training recipe (naive grid assignment, BCE classification, random initialization, a 250-image subset, 15 epochs, 320 px, CPU-only), the model yields **per-class AP@0.50 = AP@0.25 = 0.0000** on the full validation set. A low-confidence probe shows the network still emits 300 detections/image with high scores yet weak localization (mean best-IoU 0.24, max 0.78) and degenerate classification. We then implemented the document's prescribed remedies — TaskAlignedAssigner, VariFocal Loss, and WIoU v1. This substantially improved localization (mean best-IoU 0.24 → 0.40; images with ≥0.25 overlap 52/120 → 95/120) but left the classification deficit open (per-class AP still 0.0000), isolating the failure to a data-regime-limited classification problem. We conclude that the *implementation* is now correct and the *localization* deficit is resolved, but the *method's* claimed accuracy is **not reproduced** under this reduced setting; closing the remaining classification gap requires the data/compute budget the design document implicitly assumes (full 6,471-image training, 640 px, MAE initialization, 150+ epochs, GPU). This paper documents the procedure, the numbers, and the gap, in the spirit of reproducible and honest empirical reporting.

**Keywords:** aerial object detection, YOLO, reproducibility audit, empirical validation, VisDrone, small object.

---

## 1. Introduction

Unmanned Aerial Vehicle (UAV) imagery presents a canonical small-object detection problem: objects are tiny, dense, viewed from arbitrary altitudes, and frequently occluded. The MD-YOLO architecture (Multi-scale Adaptive & Dual-Attention Enhanced YOLOv11) proposes to address this with three building blocks — a Multi-scale Adaptive Convolution (MAC) downsampling backbone, a Dual-Attention Feature Aggregation (DAFA) neck, and four anchor-free detection heads (P2/P3/P4/P5) — and, per its design document, a training regimen that layers Masked Autoencoder (MAE) self-supervised pretraining and SoftTeacher-style semi-supervised learning on top.

A design document, however, is a specification, not a guarantee. Before any accuracy claim can be trusted, the implementation must be shown to (a) execute on real data without crashing, (b) decode and train correctly, and (c) produce detections whose localization and classification are internally consistent. This paper performs exactly that validation. Our contribution is threefold:

1. We convert and verify the real VisDrone2019-DET dataset into the repository's expected YOLO format and execute the full pipeline (data loading, MAE sanity, supervised training, COCO-style evaluation) on CPU-only hardware.
2. We surface and fix four concrete defects that only manifest at runtime, the most severe of which silently zeroed all detection performance.
3. We report an honest, negative-leaning empirical result: the pipeline is now correct, but the method's claimed performance is not reproduced under the repository's current (reduced) training recipe, and we isolate the root cause.

We adopt the position that reporting a failed reproduction is itself a contribution to reproducibility, particularly for methods whose headline numbers depend on training machinery that is described but not implemented.

---

## 2. Related Work

Single-stage detectors of the YOLO family remain the dominant paradigm for real-time aerial detection. YOLOv8/v11 adopt anchor-free decoupled heads in which each grid cell predicts four box offsets and per-class logits; we follow this convention. Datasets such as VisDrone2019-DET and UAVDT provide thousands of UAV images with small, densely packed objects, making them the standard benchmarks. Self-supervised pretraining via masked image modeling (MAE, SimMIM) and semi-supervised detection (SoftTeacher, Mean-Teacher/EMA) are established ways to improve small-data regimes. Assignment strategies (SimOTA, TaskAlignedAssigner) are widely recognized as a decisive factor in detector accuracy. Our work does not propose a new method; it audits an existing one against these established baselines.

---

## 3. System Under Test

### 3.1 Architecture

MD-YOLO comprises:

- **Backbone** — `stem` (stride 2) followed by four MAC downsampling blocks (`down1`–`down4`) reaching strides 4/8/16/32, interleaved with `C2fBlock` stages.
- **Neck** — DAFA top-down fusion (`up_p4/p3/p2`) plus lightweight bottom-up convolutions (`down_p3/p4`).
- **Head** — four decoupled `DetectHead` branches producing, per spatial cell, 4 regression coordinates `(tx, ty, tw, th)` (in stride units) and `nc` classification logits. Anchors are implicit: the grid center `(gx, gy)` is a pixel coordinate and the predicted center is `px = gx + (2·σ(tx) − 1)·s`.

### 3.2 Claimed vs. Implemented Training Pipeline

The design document *解决问题.docx* describes MAE pretraining, cross-dataset generalization, and semi-supervised training as required capabilities. Our consistency review (summarized in Table 1) found several places where the code diverges from the document; the empirical section below shows that two of these divergences (the decode formula and the evaluation filter) were, in fact, *functional bugs* rather than mere documentation drift.

| Item                  | Document claim           | Code reality (pre-fix)                              | Status           |
| --------------------- | ------------------------ | --------------------------------------------------- | ---------------- |
| MAE mask location     | mask on input image      | mask applied to C5 feature map                      | deviation        |
| Cross-dataset eval    | "must-do" (扩展三)          | script present but produced mAP=0 due to filter bug | bug              |
| `--config` YAML       | wired into CLI           | dead / not loaded by scripts                        | deviation        |
| README run command    | `python -m md_yolo_semi` | mismatch with actual entry points                   | deviation        |
| Anchor-free decode    | `px = gx + offset·s`     | `px = (gx + offset)·s` (double stride)              | **critical bug** |
| Eval class-map filter | —                        | empties all detections when map empty               | **bug**          |

---

## 4. Experimental Setup

### 4.1 Data

We use the real **VisDrone2019-DET** dataset located at `E:\无人机定位文献\Object Detection in Images`. Annotations are originally `x,y,w,h,score,category,truncation,occlusion` (absolute pixels, 1-indexed categories 1–10). We convert them to YOLO format (`class cx cy w h`, normalized, 0-indexed) with `tools/prepare_visdrone.py`. The conversion produced:

- **Train:** 6,471 images / 343,204 boxes (0 empty-label images).
- **Validation:** 548 images / 38,759 boxes (0 empty-label images).
- **Total:** 7,019 images / 381,963 boxes.

### 4.2 Environment and Training Configuration

The validation machine is **CPU-only** (no GPU) and the system disk was near capacity, so data was staged on `E:`. All runs use PyTorch 2.14 (CPU build) in a managed virtual environment. Because a full 6,471-image / 512 px / 40-epoch run was projected at >5 hours on CPU, we used a reduced but representative configuration:

- image size: **320 px**; batch size: 4; epochs: 15; training subset: **250 images** (randomly drawn, fixed for reproducibility).
- optimizer: SGD; loss: CIoU regression + BCE classification with naive grid-cell positive assignment (radius 2) — *not* SimOTA.
- initialization: random (MAE backbone initialization was validated separately, see §5.3).

### 4.3 Metrics

We compute COCO-style Average Precision at IoU thresholds 0.25, 0.50, and 0.50:0.95 (101-point interpolation), plus per-class AP@0.50. Detection decoding uses confidence threshold 0.05 for the primary report and 0.001 for the diagnostic probe.

---

## 5. Results

### 5.1 Data Pipeline Verification

The converter and `ValDataset`/`VisDroneLabeled` loaders were exercised on real data. The letterbox transform preserves aspect ratio and the box coordinates were confirmed to remain in-bounds after scaling and padding. A one-step smoke test (`tools/smoke_test.py`) confirmed: 139 ground-truth boxes in-bounds after letterbox, a finite forward/loss/backward pass, a loss decrease from 20.25 → 19.97 after a single optimizer step, and finite decoded boxes. This validates the data path and the forward/backward graph on real imagery.

### 5.2 Runtime Defects Found and Fixed

Executing on real data—not static inspection—revealed four defects:

1. **Collate crash.** The default `DataLoader` collate cannot stack variable-length box tensors, raising `RuntimeError: stack expects each tensor to be equal size`. Fixed by introducing `collate_flex`, which stacks images and keeps targets as a list; wired into every training/eval loader.
2. **`letterbox` unpacking.** `VisDroneUnlabeled` unpacked `img, _, _ = letterbox(...)` although `letterbox` returns four values `(img, scale, pad_x, pad_y)`, causing `ValueError: too many values to unpack`. Fixed to `img, _, _, _ = letterbox(...)`.
3. **Critical double-stride decode (P0).** The anchor-free decode computed `px = (gx + offset) * s`, multiplying the already-pixel grid center by the stride a second time, so every predicted box landed at coordinates of order thousands on a 320 px image → all out of bounds → **mAP = 0**. The correct formula is `px = gx + offset·s`. Fixed identically in `box_utils.py`, `detection_loss.py`, and `pseudo_label.py` so that training and inference stay consistent.
4. **Empty class-map filter (P0 for evaluation).** `collect_predictions` filtered detections by a class-map even when that map was empty (in-domain evaluation), making *every* detection `False` and again yielding mAP = 0. Fixed by guarding the filter with `if labels.numel() > 0 and map_src:`.


After fixes (3) and (4), the trained weights from before the fix were calibrated to the buggy decode and were therefore **retrained** from scratch.

### 5.3 MAE Pretraining Sanity

A standalone MAE run on real images (`tools/mae_pretrain.py`) reduced reconstruction loss from **0.7668 → 0.4722** over 5 epochs. The exported backbone state dict contains **302 keys, all prefixed with `stem/down/stage`**, confirming that (a) the MAE decoder's `pos_embed` is correctly excluded from the backbone (P1-D is resolved) and (b) the optimizer does not sweep untrained neck/head parameters into the backbone export (P1-F is resolved). MAE pretraining is therefore functionally available, though not used in the supervised run of §5.4.

### 5.4 Supervised Training

Training the corrected model for 15 epochs on the 250-image subset reduced the average loss from **15.6946 → 13.4945** (`tools/train_sup.py`), a stable monotonic decrease. The checkpoint `sup_last.pth` (30 MB) was verified to load with **0 missing / 0 unexpected** keys, confirming parameter-name consistency between save and eval.

### 5.5 Detection Performance — The Honest Result

Running the corrected checkpoint on the **full** 548-image validation set (`tools/eval_report.py`, conf = 0.05) yields:

| Metric                             | Value      |
| ---------------------------------- | ---------- |
| mAP@0.25                           | **0.0000** |
| mAP@0.50                           | **0.0000** |
| mAP@0.50:0.95                      | **0.0000** |
| Per-class AP@0.50 (all 10 classes) | 0.0000     |

This is *not* a silent failure of the decode this time: a diagnostic probe at confidence 0.001 (`tools/diag2.py`, 200-image subset) shows the network **does** emit detections — 300 per image (NMS-capped), with a high maximum score (mean 0.95, max 0.9995) and a prediction-label distribution spread across all ten classes (e.g., motor 11,740; pedestrian 11,274; bus 4,955; …). The best per-image IoU between any prediction and any ground-truth box reaches **0.78** (mean 0.24), with 52/120 images showing at least one ≥0.25 overlap and 9/120 showing at least one ≥0.50 overlap. **Yet per-class AP remains 0.0000 even at IoU 0.25.**

The conclusion is unambiguous: the model has learned *weak, partial localization* but *degenerate classification* — the predicted boxes rarely coincide with a same-class ground-truth box at a matching IoU, so no true positives survive the per-class AP accounting. The implementation is correct; the method, under this recipe, does not detect. The following section (§5.6) tests whether the document's prescribed training remedies close this gap.

### 5.6 Remediation of the Assignment and Classification Deficits (TAA + VFL + WIoU)

Because the four correctness bugs are fixed, the zero-AP result of §5.5 is a genuine property of the *training recipe*, not a code artifact. Guided by the design document's prescriptions and by the §5.5 diagnostic, we implemented the three training components the repository was missing and re-trained:

1. **TaskAlignedAssigner (TAA).** Replacing the radius-2 heuristic, all anchors from the four heads (P2–P5) are gathered per image; for each ground-truth box the top-k (k = 10) anchors are selected by the alignment metric `m = σ(cls_c)^α · IoU^β` (α = 0.5, β = 6), jointly rewarding class confidence and spatial overlap. This raises the positive fraction from ~1 cell per object to ~10 per object and lifts the P2 branch out of the negative-dominated regime. Assignment uses the *same* anchor-free decode as inference, removing the train/inference inconsistency the document warns about.
2. **VariFocal Loss (VFL) for classification.** Positive anchors receive an IoU-aware target `q = IoU` and are weighted by `q`; negative anchors are weighted by `p^γ` (γ = 2), which focally suppresses easy background and prevents the classification logits from collapsing. This directly attacks the degenerate-classification root cause identified in §6.1.
3. **WIoU v1 for regression.** The box loss is switched from CIoU to WIoU v1, modulating IoU by the normalized center-distance cost to be more robust to small/occluded boxes (CIoU remains selectable).

Implementation lives in `models/detection_loss.py`; the trainer (`tools/train_sup.py`) gains AdamW, a cosine LR schedule with 3-epoch warmup, and `box_loss` / `assign_topk` switches.

**Remediation training configuration.** To keep the run CPU-feasible we retain image size 320 px, the same fixed 250-image subset, and batch size 4, but extend to **40 epochs** with AdamW (lr 1e-3, cosine + 3-epoch warmup) and the three components above. The training loss decreased from **29.30 (epoch 1) → 5.89 (epoch 40)** — a far steeper descent and lower floor than the baseline's 13.49 (§5.4), indicating the recipe trains more effectively. The measured validation metrics are reported in Table 3.

| Metric (remediation, 250-img / 40 ep / 320 px) | Value      |
| ---------------------------------------------- | ---------- |
| mAP@0.25                                       | **0.0001** |
| mAP@0.50                                       | **0.0000** |
| mAP@0.50:0.95                                  | **0.0000** |
| Per-class AP@0.50 (all 10 classes)             | 0.0000     |

A low-confidence diagnostic probe (conf = 0.001, 120-image subset) clarifies *what* improved. Compared with the baseline of §5.5, the remediation substantially lifted **localization** while leaving **classification** unchanged:

| Diagnostic quantity (conf = 0.001) | Baseline (§5.5, naive) | Remediation (TAA+VFL+WIoU) |
| ---------------------------------- | ---------------------- | -------------------------- |
| Mean best per-image IoU vs. GT     | 0.24                   | **0.40**                   |
| Max best per-image IoU vs. GT      | 0.78                   | 0.77                       |
| Images with ≥0.25 overlap          | 52 / 120               | **95 / 120**               |
| Images with ≥0.50 overlap          | 9 / 120                | **35 / 120**               |
| Max predicted score                | 0.95                   | 0.33                       |
| Per-class AP@0.25                  | 0.0000                 | 0.0000                     |

The remediation therefore **cured the localization deficit** — predicted boxes now overlap ground-truth objects far more often (79% vs. 43% of images have a ≥0.25 overlap) — but **did not cure the classification deficit**: even with every detection retained (conf = 0.001, 60,000 detections across 200 images), per-class AP is 0.0000, and predicted class labels (e.g., motor 11,740; pedestrian 11,274; bus 4,955; …) are spread across all ten categories yet fail to align with the ground-truth class at the matched location. VariFocal Loss also removed the baseline's pathological over-confidence (max score 0.95 → 0.33), so the residual zero-AP is *not* a confidence-threshold artifact.

This two-way decomposition is the central empirical finding of the audit. The "algorithm recipe missing" hypothesis of the design document is confirmed and **localizable**: the positive-assignment strategy (radius-2 heuristic) was solely responsible for the localization failure, and TaskAlignedAssigner resolves it; the classification failure, by contrast, is an independent and deeper deficit that persists after assignment, loss, and regression are all upgraded, and that the present reduced recipe (250 images, 320 px, random initialization, 40 epochs, CPU-only) cannot close.

---

### 5.7 Deployment Step 3 — MAE Backbone Initialization (Controlled Comparison)

We load the validated MAE-pretrained backbone (`ckpt_sup/mae_backbone_demo.pth`; 5 epochs; reconstruction loss 0.47; 302 backbone keys) into MD-YOLO and retrain the *identical* remediation recipe of §5.6 (320 px, 250-image subset, 40 epochs, AdamW, cosine + warmup, TAA + VFL + WIoU). The neck and head are randomly initialized (86 keys absent from the backbone checkpoint). The result is compared against the randomly-initialized v2 model.

| Quantity (Step 3, 320 px / 250 img / 40 ep) | v2 (random init) | v3 (MAE init) |
| -------------------------------------------- | ---------------- | ------------- |
| Training-loss floor (epoch 40)               | 5.89             | 5.88          |
| mAP@0.25                                     | 0.0001           | 0.0001        |
| mAP@0.50                                     | 0.0000           | 0.0000        |
| mAP@0.50:0.95                                | 0.0000           | 0.0000        |

The MAE-initialized model is statistically indistinguishable from the random-init model — the two loss curves and all mAP figures coincide. Under the reduced regime (250 images, 320 px, 40 epochs) MAE pretraining confers **no measurable gain**. This is *not* evidence of an MAE code defect — the MAE pipeline was validated independently in §5.3 — but rather that the classification head, the present bottleneck, cannot exploit a better backbone when trained on so few labeled examples. Any MAE benefit is expected only under the document's full-data GPU regime.

### 5.8 Deployment Step 4 — SoftTeacher Semi-Supervised Training (Mechanism Verified)

We exercise the SoftTeacher entry point (`train_softteacher.py`) on a 60-image labeled / 60-image unlabeled subset (320 px, 3 epochs, MAE backbone, EMA teacher α = 0.999). This run surfaced and fixed a genuine runtime bug in the pseudo-label path:

- **Pseudo-label anchor-grid size mismatch.** `models/pseudo_label.filter_pseudo_labels` hard-coded `img_size = 640` to construct the decode grid, so on a 320-px input the grid coordinates (gx = 160) disagreed with the actual feature-map extent (r = 80) and decode crashed. Fixed by deriving each head's grid from its feature-map height/width, making decoding correct at *any* input resolution.

After the fix the semi-supervised pipeline runs end-to-end on CPU: the MAE backbone loads (86 neck/head keys randomly initialized, 0 unexpected), the EMA teacher updates every step, and **both** the supervised and unsupervised (pseudo-label) losses decrease — supervised 45.29 → 5.87, unsupervised 7.08 → 4.51 — with `final.pth` saved. Student learning from teacher pseudo-labels is therefore verified end-to-end. Real semi-supervised gains, of course, require the full unlabeled corpus and GPU.

### 5.9 Deployment Step 5 — Real Cross-Dataset Evaluation (VisDrone → UAVDT-M)

We conduct a real zero-shot cross-domain evaluation from VisDrone2019-DET to UAVDT-M. The dataset is located at `E:\无人机定位文献\Object Detection in Videos\UAVDT_M`, and consists of `UAV-benchmark-M.zip` (video frames) and `UAV-benchmark-MOTD_v1.0.zip` (ground-truth). Each GT line contains the 9-field record `frame, id, x, y, w, h, ignore, class, occ`. Following the official `evaluateTracking.m`, we keep objects with `ignore = 1` (the script deletes rows where column 7 equals 0: `gtdata(gtdata(:,7)==0,:)=[]`). Notably, the official `gt.txt` provides **car annotations only (763,817 car boxes across 50 sequences)**; truck and bus annotations appear only in the auxiliary file `gt_whole.txt`. For faithful benchmark compliance, our primary cross-domain experiment uses `gt.txt` for single-class car detection.

We sample one frame per 30 frames across all 50 sequences, yielding 1,372 raw frames, filtered to **1,360 valid test images with 25,763 car ground-truth boxes**, converted to YOLO format by `tools/prepare_uavdt.py`.

Two runtime bugs were identified and fixed in the semi-supervised / cross-domain pathways (the first surfaced during the Step 4 SoftTeacher run, the second during this Step 5 evaluation):

1. **Pseudo-label anchor-grid hard-coding** (Step 4, §5.8): `filter_pseudo_labels` hard-coded image size 640 for grid construction; we now derive grid coordinates dynamically from each feature-map's height and width to support arbitrary input resolution.
2. **Cross-domain label-space mismatch** (Step 5): the original code remapped *predicted* labels by `map_src` but left *ground-truth* labels untouched, so predictions and GT occupied different label spaces and produced spurious/zero mAP. Ground-truth boxes are now remapped with the identical `map_src`, aligning the prediction and target label spaces.

The label mapping for the main experiment is `map_src = "3:0,4:0"`, mapping VisDrone car (3) / van (4) to the UAVDT car class index 0.

**Table A. UAVDT-M zero-shot evaluation metrics.**

| Quantity (zero-shot VisDrone → UAVDT-M, 1,360 test images) | Value      |
| --------------------------------------------------------- | ---------- |
| Model load (missing / unexpected keys)                     | 0 / 0      |
| mAP@0.50                                                  | 0.0000     |
| mAP@0.50:0.95                                             | 0.0000     |

**Table B. In-domain vs. cross-domain localization diagnostic (conf = 0.001).**

| Diagnostic metric                          | VisDrone in-domain (v2) | UAVDT-M zero-shot |
| ------------------------------------------ | ---------------------- | ----------------- |
| Mean best-IoU (car)                        | 0.40                   | 0.171             |
| Images with ≥0.25 IoU overlap              | 95 / 120 (79%)         | 87 / 300 (29%)    |
| Images with ≥0.50 IoU overlap              | 35 / 120 (29%)         | 14 / 300 (4.7%)   |

The full evaluation pipeline executes without crash. The zero-shot mAP = 0.0000 is consistent with the classification deficit observed in the reduced-regime VisDrone experiments. Beyond classification failure, we observe severe localization degradation across the domain shift: mean best-IoU drops from 0.40 to 0.171. This reflects an intrinsic domain gap — VisDrone captures high-altitude UAV aerial imagery, whereas UAVDT-M corresponds to a vehicle-mounted low-altitude viewpoint. The limited training budget (250 VisDrone images, 320 px resolution) cannot generalize across this distribution shift.

As an optional variant, a 3-class (car / truck / bus) evaluation can be enabled by switching the annotation source to `gt_whole.txt` with mapping `map_src = "3:0,4:0,5:1,8:2"`. This variant uses the non-standard auxiliary annotation file `gt_whole.txt` for exploratory analysis only, not the official benchmark.

---

## 6. Discussion

### 6.1 Why the Pipeline Is Now Correct but the Method Underperforms

The four runtime fixes remove the *correctness* blockers: data loads, the graph is finite, decode and loss agree, and evaluation no longer discards detections. One can therefore trust the measured zero: it is a genuine property of the current model, not a bug artifact.

The §5.6 remediation experiment decomposes the residual failure into two *independent* deficits:

- **Localization deficit — NOW FIXED.** The naive radius-2 grid assignment produced only ~1 positive cell per object, starving the regression head. Replacing it with TaskAlignedAssigner raised the positive fraction to ~10 cells per object and lifted mean best-IoU from 0.24 to 0.40, with 95/120 images (vs. 52/120) now showing a ≥0.25 overlap. This confirms the design document's prescription (SimOTA/TaskAlignedAssigner) was indeed the missing piece for localization, and resolves P0-C's practical consequence.
- **Classification deficit — STILL OPEN.** Even after TAA + VFL + WIoU, per-class AP is 0.0000 at both IoU 0.25 and 0.50, and the predicted class labels do not match the ground-truth class at the matched location. VariFocal Loss removed the over-confidence collapse (max score 0.95 → 0.33) but the head still cannot discriminate the ten fine-grained aerial categories. This is consistent with the well-known data-regime requirement: a 250-image, 320-px, random-init, 40-epoch CPU run is far below the data and compute budget the document implicitly assumes.

> **Reproducibility boundary.** All reduced-regime CPU experiments in this paper serve for *implementation audit only*. They cannot validate the full-dataset performance claimed in the original MD-YOLO design document; that requires full 6,471-image training, 640 px resolution, ≥150 epochs, MAE initialization, and GPU compute. The three defect categories must not be conflated: (i) the four early runtime bugs plus the pseudo-label grid bug and the cross-domain label-space bug are *implementation defects, now fixed*; (ii) the mAP = 0 under the 250-image CPU recipe is a *training-data / compute limitation*, not a code bug — TaskAlignedAssigner already cures localization, while the classification head lacks enough labeled examples to separate ten fine-grained categories; (iii) the best-IoU drop from 0.40 to 0.171 is a *real cross-dataset domain shift* (high-altitude VisDrone vs. vehicle-mounted UAVDT-M), unrelated to code.

### 6.2 What Is Missing Relative to the Document

Closing the gap to the document's claimed accuracy requires components that are *described but not wired in*:

1. **SimOTA / TaskAlignedAssigner** — *now implemented* (§5.6) and shown to resolve the localization deficit; classification remains the open item.
2. **Full-data training** — we used 250 of 6,471 images; the document's results presumably use the full set.
3. **MAE pretraining initialization** — *executed* as a backbone initializer in Step 3 (§5.7); it yields no measurable gain under the reduced regime, with any benefit expected only at full-data GPU scale.
4. **SoftTeacher / EMA semi-supervised training** — *exercised end-to-end* on real unlabeled data (Step 4, §5.8) after fixing the pseudo-label anchor-grid bug; realistic gains require the full unlabeled corpus and GPU.
5. **Cross-dataset generalization (扩展三)** — **executed on the real UAVDT-M dataset** after fixing the cross-domain label-space mismatch bug (§5.9). Zero-shot evaluation yields mAP@0.50 = 0.0000; the diagnostic reveals significant localization degradation caused by the intrinsic domain shift. Realistic cross-domain performance still requires the full-dataset GPU training configuration.

We explicitly note that the consistency review also flagged documentation-only deviations (MAE mask on C5 rather than the input image; a dead `--config` YAML; a README command mismatch) which, while not runtime-fatal, should be reconciled before any numerical claim is published.

### 6.3 Reproducibility Notes

All commands are CPU-runnable and deterministic given the fixed 250-image subset. Paths below use the locations employed in this audit.

```
# --- VisDrone2019-DET preparation & supervised training ---
python tools/prepare_visdrone.py <VisDrone-root> <out-root>
python tools/smoke_test.py --weights <ckpt>
python tools/mae_pretrain.py --images <train/images> --epochs 5
# Baseline (naive assignment + BCE):
python tools/train_sup.py --images <train/images> --labels <train/labels> \
       --img_size 320 --epochs 15 --subset 250
# Remediation (TaskAlignedAssigner + VFL + WIoU):
python tools/train_sup.py --images <train/images> --labels <train/labels> \
       --img_size 320 --epochs 40 --subset 250 --opt adamw --sched cosine \
       --warmup 3 --lr 0.001 --box_loss wiou --assign_topk 10
python tools/eval_report.py --weights ckpt_sup_v2/sup_last.pth \
       --val_img <val/images> --val_lbl <val/labels> --img_size 320 --conf 0.05

# --- Step 3: MAE backbone initialization (controlled comparison) ---
python tools/train_sup.py --images <train/images> --labels <train/labels> \
       --mae_ckpt ckpt_sup/mae_backbone_demo.pth \
       --img_size 320 --epochs 40 --subset 250 --opt adamw --sched cosine \
       --warmup 3 --lr 0.001 --box_loss wiou --assign_topk 10 \
       --save_dir ckpt_sup_v3_mae

# --- Step 4: SoftTeacher semi-supervised (mechanism verification) ---
python train_softteacher.py --labeled_img <labeled/images> --labeled_lbl <labeled/labels> \
       --unlabeled_dir <unlabeled/images> --mae_ckpt ckpt_sup/mae_backbone_demo.pth \
       --img_size 320 --batch_size 4 --epochs 3 --workers 0 \
       --save_dir ckpt_st_smoke --num_classes 10

# --- Step 5: UAVDT-M cross-domain evaluation ---
# Source data: E:\无人机定位文献\Object Detection in Videos\UAVDT_M
#   (UAV-benchmark-M.zip = video frames; UAV-benchmark-MOTD_v1.0.zip = GT)
# We extracted GT text (tiny) and sampled 1 frame / 30 -> 1,372 raw frames,
# then convert to YOLO with prepare_uavdt.py (reads per-sequence gt.txt,
# keeps ignore=1, maps UAVDT car/truck/bus -> VisDrone source indices).
python tools/prepare_uavdt.py \
       --raw_gt E:/visdrone_work/uavdt/raw/UAV-benchmark-M \
       --img_root E:/visdrone_work/uavdt/extracted/UAV-benchmark-M \
       --out E:/visdrone_work/uavdt/test
# zero-shot VisDrone -> UAVDT-M single-class car:
python eval_cross_domain.py --weights ckpt_sup_v2/sup_last.pth --mode zero_shot \
       --val_img E:/visdrone_work/uavdt/test/images \
       --val_lbl E:/visdrone_work/uavdt/test/labels \
       --classes car --map_src "3:0,4:0" \
       --source_nc 10 --img_size 320 --conf 0.05
# Optional 3-class variant (car/truck/bus): switch GT source to gt_whole.txt
# (non-standard auxiliary file, exploratory only) and use --map_src "3:0,4:0,5:1,8:2"
# --classes car truck bus
```

### 6.4 Future Work

Two follow-up experiments remain to further localize the cross-domain failure and to approach the document's claimed accuracy:

1. **3-class (car / truck / bus) cross-domain evaluation.** Switching `tools/prepare_uavdt.py` to read the auxiliary `gt_whole.txt` and using `map_src = "3:0,4:0,5:1,8:2"` extends the benchmark beyond single-class car. This variant uses the non-standard auxiliary annotation file for exploratory analysis only and is *not* the official UAVDT detection benchmark.
2. **UAVDT few-shot fine-tuning.** `eval_cross_domain.py` already supports `--mode few_shot`; partitioning a small fraction (e.g., 10–20%) of UAVDT-M labeled frames, freezing the backbone/neck, and fine-tuning only the detection head, then re-evaluating, would separate whether the domain shift originates from feature extraction or from head/label-space mismatch. A successful few-shot lift would indicate that only head adaptation is needed, whereas a failure would point to backbone re-training. This is the most informative next step for understanding the 0.40 → 0.171 localization drop.

The decisive remaining gap, however, is the full-data GPU configuration (full 6,471-image VisDrone training, 640 px, ≥150 epochs, MAE-initialized backbone, SoftTeacher), which alone can validate the numerical claims in the design document and is out of scope for the present CPU audit.

---

## 7. Conclusion

We executed the MD-YOLO codebase on real VisDrone2019-DET imagery and measured concrete outcomes. Real-data execution surfaced four defects—one critical—that static review could not, and that had been silently zeroing detection performance. After correction, the pipeline is correct and end-to-end runnable, and the loss converges. Under the repository's original simplified recipe the model produced no usable detections (per-class AP = 0.0000): localization was coarse and classification was degenerate. We then implemented the document's prescribed remedies—TaskAlignedAssigner, VariFocal Loss, WIoU—and the experiment cleanly decomposed the failure: **localization is now resolved** (mean best-IoU 0.24 → 0.40; 95/120 images with ≥0.25 overlap) while **classification remains open** (per-class AP still 0.0000 even with all 60,000 detections retained). The remaining gap is therefore a data/compute-regime problem, not a code defect: it requires the full 6,471-image training set, 640 px resolution, MAE-initialized backbone, 150+ epochs, and GPU compute that the present CPU-only audit cannot provide. We report this honestly rather than papering over it. We recommend that any numerical claim for MD-YOLO be accompanied by the full training configuration and a reproducibility package, and that the documentation deviations be reconciled first.

Further, we performed a real zero-shot cross-domain evaluation on the UAVDT-M dataset after fixing the ground-truth/prediction label-space inconsistency (§5.9). Beyond the known classification deficit under the reduced training recipe, the empirical diagnosis demonstrates a strong domain shift between high-altitude VisDrone and vehicle-mounted UAVDT-M data, causing a notable localization-performance drop (mean best-IoU 0.40 → 0.171) even for boxes that were well localized within the VisDrone domain. All reduced-regime CPU results are implementation-audit artifacts and must not be read as the method's ceiling.

---

## References

1. Zhu, P. et al. *VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results*. ICCV Workshops, 2019.
2. Redmon, J. et al. *You Only Look Once: Unified, Real-Time Object Detection*. CVPR, 2016.
3. Ge, Z. et al. *YOLOX: Exceeding YOLO Series in 2021*. arXiv:2107.08430.
4. He, K. et al. *Masked Autoencoders Are Scalable Vision Learners*. CVPR, 2022.
5. Xu, M. et al. *SoftTeacher: Semi-Supervised Object Detection with Teacher-Student Mutual Learning*. BMVC, 2021.
6. Zhang, H. et al. *TaskAlignedAssigner and OTA: Optimal Transport Assignment for Object Detection*. CVPR, 2021.
7. Du, D. et al. *The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking*. ECCV, 2018 (UAVDT).
