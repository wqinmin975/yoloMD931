# An Empirical Reproducibility Audit of MD-YOLO for Small-Object Detection in UAV Imagery

Qinmin Wang¹, Zhenzhen Hu², Hongyuan Wang¹, and Yuming Sun¹,*

¹ School of Computer Science and Artificial Intelligence, Guangdong University of Science and Technology, Dongguan, China
² Chengdu University of Information Technology, Chengdu, China
*Corresponding author. Email: sunyuming@gdust.edu.cn*

*ORCID iDs:* Qinmin Wang, 0009-0006-0447-0482; Zhenzhen Hu, 0000-0003-2455-0691; Hongyuan Wang, 0000-0007-4278-771X; Yuming Sun, 0009-0003-7222-2390.

**Abstract.** We present an end-to-end empirical reproducibility audit of MD-YOLO, a multi-scale adaptive, dual-attention enhanced YOLOv11 variant for UAV small-object detection. Previous assessment of this codebase relied purely on static source-code inspection; in this work, we convert the official VisDrone2019-DET split (6,471 training / 548 validation images, 7,019 total, 381,963 boxes) and run the complete detection pipeline on this real aerial corpus. Real-data execution uncovered four runtime defects missed by static review, including a critical double-stride decoding bug that silently yielded $\text{mAP}=0$. After bug fixes, the pipeline runs end-to-end with converged training loss. Nevertheless, under a reduced training recipe (250 training images, 320 px input resolution, 40 epochs, CPU-only hardware), the model achieves per-class AP@0.50 = 0.0000. Diagnostic decomposition demonstrates that the proposed remedies (TaskAlignedAssigner + VariFocal Loss + WIoU) resolve localization deficiency (mean best-IoU increases from 0.24 to 0.40). Nevertheless, classification performance remains poor, stemming from a constrained data-compute regime rather than software defects. Further zero-shot cross-domain evaluation on the UAVDT-M dataset reveals severe domain degradation, with mean best-IoU dropping from 0.40 to 0.27. We honestly report these negative experimental observations and clearly separate three non-overlapping categories of failure, to support reproducible UAV object-detection research.

**Keywords:** aerial object detection; YOLO; reproducibility audit; empirical validation; VisDrone; UAVDT; small object; domain shift

## 1. Introduction

Unmanned aerial vehicle (UAV) remote sensing has become a cornerstone of modern earth observation, supporting traffic monitoring, disaster response, and precision agriculture (Xia et al. 2018; Cheng et al. 2023). Yet the small ground-sample distance and arbitrary viewing geometry of aerial platforms make object detection a canonical small-object problem: targets are tiny, densely packed, captured from varying altitudes, and frequently occluded (Cheng et al. 2023). Building on the YOLO family of detectors (Redmon et al. 2016; Jocher et al. 2023; Khanam et al. 2024), the MD-YOLO architecture (Multi-scale Adaptive & Dual-Attention Enhanced YOLOv11; Zhou and Gong 2026) targets this setting with a multi-scale adaptive convolution (MAC) backbone, a dual-attention feature aggregation (DAFA) neck, and four anchor-free detection heads (P2–P5), and—per its design specification—a training regimen layering masked autoencoder (MAE) pretraining (He et al. 2022) and SoftTeacher-style semi-supervised learning (Xu et al. 2021).

A design document is a specification, not a guarantee. Before any accuracy claim can be trusted, the implementation must be shown to (i) execute on real data, (ii) decode and train correctly, and (iii) produce detections whose localization and classification are internally consistent. Previous assessment of this codebase—including a prior static assessment—relied purely on source-code inspection. This paper fills this gap by carrying out empirical validation on real-world aerial data. Our contributions are: (1) we convert and verify the real VisDrone2019-DET dataset and execute the full pipeline on CPU-only hardware; (2) we surface and fix four runtime defects, the most severe of which silently zeroed all detection performance; and (3) we report an honest, negative-leaning result—the pipeline is now correct and localization is resolved, but the method's claimed accuracy is not reproduced under the reduced recipe, and we isolate the root cause as a classification data-regime limitation. We further conduct a real zero-shot cross-domain evaluation on UAVDT-M that quantifies a severe domain shift. We argue that reporting unsuccessful reproductions constitutes a valuable contribution to reproducible research. This is particularly relevant for methods whose headline performance relies on training components described in the technical specification yet not fully integrated or validated under the complete proposed training recipe (Pineau et al. 2020).

## 2. Materials and Methods

### 2.1 System under test

MD-YOLO comprises a `stem` (stride 2) and four MAC downsampling blocks (`down1`–`down4`) reaching strides 4/8/16/32, a DAFA top-down/bottom-up neck, and four decoupled `DetectHead` branches. Each head predicts, per spatial cell, four box offsets (t_x, t_y, t_w, t_h) in stride units and n_c class logits. Anchors are implicit; the predicted center is decoded as

p_x = g_x + (2σ(t_x) − 1)·s,   p_y = g_y + (2σ(t_y) − 1)·s,   p_w = s·e^{t_w},   p_h = s·e^{t_h}      (1)

where g_x, g_y are the grid-center pixel coordinates, s the feature-map stride, and σ the logistic sigmoid function. The design specification also prescribes MAE pretraining, cross-dataset generalization, and SoftTeacher semi-supervised training.

### 2.2 Datasets

We use the real **VisDrone2019-DET** dataset (Du et al. 2019): 6,471 training images / 343,204 boxes and 548 validation images / 38,759 boxes (381,963 boxes total). Annotations use 1-indexed category labels (converted to 0-based model indices via category - 1, giving n_c = 10) and raw absolute-pixel (x, y, w, h) boxes (converted to normalized YOLO format); ignored regions (category 0) and out-of-bounds boxes are excluded. For the cross-domain study we use **UAVDT-M**, derived from the Unmanned Aerial Vehicle Detection and Tracking (UAVDT) benchmark (Du et al. 2018). Its detection ground-truth annotations adopt the official nine-field format (frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, ignore_flag, occlusion, object_category). For the car-only detection file `gt.txt` the object_category field is a −1 placeholder (the file is already restricted to cars), whereas the seventh field (1-indexed) is the official ignore flag under which rows with ignore_flag = 0 are excluded by the evaluator (evaluateTracking.m: `gtdata(gtdata(:,7)==0,:)=[]`); in the released `gt.txt` this flag is uniformly 1, so no rows are dropped. We therefore treat every retained box as a car and map it to the source-domain car index, while the full multi-class labels (car/truck/bus) are available in the auxiliary `gt_whole.txt` (object_category ∈ {1, 2, 3}). The official `gt.txt` provides 763,817 car boxes across 50 sequences (truck and bus appear only in `gt_whole.txt`). We sample every 30 frames across all sequences, yielding 1,360 valid test images with 25,763 car ground-truth boxes, converted to YOLO format. All diagnostic subsets are deterministic: the in-domain diagnostic uses the first 120 validation images and the zero-shot diagnostic uses the first 300 entries from the frame-sampled UAVDT-M list (no random sampling is performed).

### 2.3 Environment and reduced training configuration

Validation runs on a CPU-only machine. Because a full-dataset training run (6 471 images at practical aerial-detection resolutions and schedules) is far beyond CPU feasibility, we adopt a representative reduced-size configuration: image size 320 px, batch size 4, the first 250 images of the sorted training list (fixed for reproducibility), 40 epochs, AdamW (lr 1e-3) with cosine schedule and 3-epoch warmup. The baseline uses the original assignment—each ground-truth object is placed on a single pyramid level selected by its size and only its center cell is marked positive (≈1 positive per object)—with sigmoid BCE classification and CIoU box regression; the remediation replaces these with TaskAlignedAssigner (TAA; Feng et al. 2021; top-k = 10, alignment metric m = σ(c)^α · IoU^β defined in Equation (2), where σ(c) is the classification score and IoU (intersection over union) the localization quality of the matched anchor; TOOD originally sets α = 1.0; we adopt α = 0.5 to dampen noisy classification scores under limited data, placing relatively greater emphasis on localization quality), VariFocal Loss (VFL; Zhang et al. 2021; Equation (3)), and WIoU v1 (Tong et al. 2023; Equation (4)):

m = σ(c)^α · IoU^β      (2)

L_VFL = − α_v·q·[q log p + (1 − q) log(1 − p)] − α_v·p^γ·log(1 − p)      (3)

L_WIoU = 1 − exp(−(Δx^γ + Δy^γ)/(W_g^γ + H_g^γ))·IoU      (4)

where in Equation (3) the first term applies to positive anchors with soft target q = IoU, the second term to negatives down-weighted by p^γ (γ = 2), and α_v = 0.75 is a global weight; this matches the original VarifocalNet formulation. In Equation (4), (Δx, Δy) is the center distance between prediction and ground truth, (W_g, H_g) the smallest enclosing-box dimensions, and γ = 2.

### 2.4 Metrics

We report COCO-style average precision (AP; averaged over classes, denoted mAP; Lin et al. 2014) at IoU = 0.25, 0.50, and 0.50:0.95 (101-point interpolation) plus per-class AP@0.50. A low-confidence diagnostic probe (conf = 0.001, applied after class-aware NMS at IoU 0.7 with top-300 retention, i.e. at most 300 detections per image) retains essentially all detections to measure the best per-image IoU between any prediction and any ground-truth box.

### 2.5 Implementation corrections

Real-data execution—not static inspection—revealed four defects, the most severe being a double-stride decode error: the erroneous form p_x^* = (g_x + offset)·s applies the stride twice—g_x is already a pixel coordinate, so only the offset should be scaled (p_x = g_x + offset·s)—projecting every predicted box out of image bounds and silently producing $\text{mAP}=0$. The other three defects are a variable-length collate crash in the DataLoader, incorrect unpacking of the four-tuple returned by the letterbox resize, and an empty class-map filter that silently cleared all predictions; the complete defect list with symptoms and fixes is provided as Table S1 in the Supplementary Material. All defects were corrected identically across training, loss, and inference.

## 3. Results

### 3.1 Pipeline verification and training

The data converter and loaders were exercised on real data; a one-step smoke test confirmed finite forward/loss/backward and in-bounds decoded boxes. A standalone MAE run reduced reconstruction loss from 0.7668 to 0.4722 over 5 epochs, exporting 302 backbone keys (prefixes `stem/down/stage`), confirming the backbone export path. We note that this MAE reconstruction experiment is a sanity check for backbone weights, not full downstream detection training. A short 15-epoch supervised run of the corrected model reduced average loss from 15.69 to 13.49; this run served as an integration smoke test only and is not used for any accuracy table—Tables 2 and 3 report the 40-epoch checkpoints described in §3.2. The checkpoint loads with 0 missing / 0 unexpected keys.

### 3.2 Honest detection performance and the localization–classification decomposition

Running the corrected checkpoint on the full 548-image validation set yields per-class AP@0.50 = AP@0.25 = 0.0000 (Table 2). A diagnostic probe shows the network still emits 300 detections/image (max score 0.95) with weak localization (mean best-IoU 0.24, max 0.78), isolating a degenerate classification rather than a silent decode failure.

| Metric (full validation, conf = 0.05) | Value |
| --- | --- |
| mAP@0.25 | 0.0000 |
| mAP@0.50 | 0.0000 |
| mAP@0.50:0.95 | 0.0000 |
| Per-class AP@0.50 (all 10 classes) | 0.0000 |

*Table 2. Validation performance (full 548-image set) of the corrected 40-epoch baseline—original single-center assignment, BCE, CIoU—before remediation, under the reduced recipe.*

Implementing the prescribed remediation (TaskAlignedAssigner + VariFocal Loss + WIoU) lowered training loss from 29.30 to 5.89 and sharply improved localization while leaving classification unchanged (Table 3). The decomposition is the central finding: localization is resolved (mean best-IoU 0.24 → 0.40; 95/120 images with ≥0.25 overlap vs. 52/120) but classification remains open (per-class AP still 0.0000 even with every emitted detection retained at the diagnostic threshold, where the maximum predicted score is only 0.33). VariFocal Loss also removed pathological over-confidence (the maximum predicted score decreased from 0.95 to 0.33). We emphasize that the best per-image IoU diagnostic is class-agnostic—the maximum overlap between any predicted and any ground-truth box—so its improvement reflects localization only; because predicted classes remain wrong, per-class AP stays at 0.

| Diagnostic quantity (conf = 0.001) | Baseline | Remediation |
| --- | --- | --- |
| Mean best per-image IoU vs. ground truth (GT) | 0.24 | 0.40 |
| Images with ≥0.25 overlap | 52/120 | 95/120 |
| Images with ≥0.50 overlap | 9/120 | 35/120 |
| Max predicted score | 0.95 | 0.33 |
| Per-class AP@0.25 | 0.0000 | 0.0000 |
| Per-class AP@0.50 | 0.0000 | 0.0000 |

*Table 3. Localization–classification decomposition: baseline (original) vs. remediation (TAA+VFL+WIoU), evaluated on a fixed n = 120 diagnostic image subset under 250-image / 320-px / 40-epoch training (both arms use 40-epoch checkpoints).*

Two controlled follow-ups confirm the classification gap is regime-limited, not a residual bug: (i) loading the MAE-pretrained backbone gives a loss floor of 5.88 vs. 5.89 for random init and identical mAP—no measurable gain under this reduced regime (this does not speak to full-recipe performance); (ii) the SoftTeacher semi-supervised path runs end-to-end after fixing a pseudo-label anchor-grid bug (hard-coded 640-px grid changed to a dynamic per-feature-map grid), with supervised loss 45.3 → 5.9 and unsupervised 7.1 → 4.5. The SoftTeacher path validates pipeline correctness only; it is not included in the accuracy comparisons of Tables 2 and 3.

### 3.3 Zero-shot cross-domain evaluation (VisDrone → UAVDT-M)

We evaluate the VisDrone-trained model zero-shot on UAVDT-M car detection. The class bridge `map_src = 3:0, 4:0` maps the 0-based VisDrone classes car (3) and van (4) onto the single UAVDT car class; the reported cross-domain metrics therefore refer to this merged vehicle class, not to a strict car-only evaluation, and predictions and ground truth are remapped consistently. The pipeline executes without runtime error (0/0 key mismatch for the model checkpoint), yet mAP@0.50 = 0.0000, consistent with the classification degradation observed above. Beyond classification, a severe localization degradation appears across the domain shift (Table 4, Figure 1): under the same class-agnostic protocol, mean best-IoU drops from 0.40 to 0.27, and images with ≥0.25 / ≥0.50 overlap fall from 79% / 29% to 49.7% / 15.0%. This reflects an intrinsic domain gap between two UAV datasets—differences in flight altitude, scene type, camera parameters, and capture conditions (VisDrone versus UAVDT-M)—that the limited training budget cannot bridge.

| Diagnostic metric | VisDrone in-domain | UAVDT-M zero-shot |
| --- | --- | --- |
| mAP@0.50 (merged vehicle class) | 0.0000 | 0.0000 |
| Mean best-IoU (class-agnostic) | 0.40 | 0.27 |
| Images with ≥0.25 IoU overlap | 95/120 (79%) | 149/300 (49.7%) |
| Images with ≥0.50 IoU overlap | 35/120 (29%) | 45/300 (15.0%) |

*Table 4. In-domain vs. cross-domain localization diagnostic (conf = 0.001), under a single class-agnostic protocol: the best per-image IoU is the maximum overlap between any predicted box and any ground-truth box, evaluated with the same remediated (v2) checkpoint on both datasets. In-domain figures use the n = 120 VisDrone diagnostic subset; zero-shot figures use the first 300 images of the frame-sampled UAVDT-M list.*

## 4. Discussion

After correcting four runtime defects, zero-valued AP metrics reflect model behaviour under limited training resources instead of implementation bugs. Our remediation experiment separates two independent failure modes: localization performance can be substantially improved using TaskAlignedAssigner (raising positive anchor cells from ~1 to ~10 per ground-truth object), whereas classification performance remains poor when trained with only 250 samples, 320-pixel resolution and limited epochs. This limitation arises from insufficient training data rather than code defects. From the perspective of UAV remote sensing, it underscores that aerial detectors demand large, domain-specific annotated corpora before their claimed accuracy can be realized in operational settings. Cross-domain evaluation further demonstrates a third independent effect: intrinsic domain shift degrades localization even for samples well-localized within the source dataset.

For reproducible research practice, we explicitly distinguish three mutually independent failure sources: (i) runtime and cross-mapping software defects, which are fully resolved; (ii) zero-AP metrics under the small CPU training subset caused by limited data-compute budget—intra-domain localization capacity has already been restored, but classification remains open; (iii) a further localization drop under cross-domain evaluation (mean best-IoU from 0.40 to 0.27), which represents a genuine dataset domain gap and is distinct from failure (ii). All results obtained under this CPU-only reduced training regime serve solely for code auditing and should not be interpreted as the upper-bound performance achievable for MD-YOLO. To achieve the performance claimed in the original design specification requires the full 6 471-image training set, a higher input resolution, longer training schedules, MAE backbone initialization, SoftTeacher semi-supervised learning, and GPU compute—the design specification prescribes these components without fixing a complete numerical recipe—and lies outside the scope of this audit. Exploratory future work includes a small-scale data/compute scaling study (e.g. 500–1 000 training images or higher input resolution) to verify the data-regime explanation, 3-class UAVDT evaluation (car/truck/bus), and few-shot fine-tuning on target-domain UAVDT annotations.

Full reproducibility commands, pseudo-code and supplementary diagnostic tables are provided in Supplementary Material.

## Declaration of Competing Interests

The authors report there are no competing interests to declare.

## Acknowledgments

This research was supported by Guangdong University of Science and Technology under Grant No. GKY-2025BSQDK-46. The authors thank the VisDrone and UAVDT benchmark providers for public datasets.

## Data and Code Availability

The MD-YOLO system under test is publicly available as source code at https://github.com/wqinmin975/yoloMD931 (commit: 451fe37055b2bdee8b5baf59f246f14856dc2524; accessed: 2026-10-10). The audit-specific artifacts of this work—bug-fix patches, dataset-conversion scripts, and exact experiment commands and configuration files—are included in the Supplementary Material and will be released in a public repository upon manuscript acceptance (repository URL, commit hash, and code-access date to be completed post-acceptance). The converted YOLO-format VisDrone2019-DET annotations, frame-sampled UAVDT-M image list, and deterministic diagnostic subsets (first 250 sorted training images for model training; first 120 validation images and first 300 frame-sampled UAVDT-M images for diagnostics; no random sampling is used) are fully reproducible via the conversion scripts provided in the Supplementary Material. All experiments were executed on CPU-only hardware; complete environment specifications including operating-system, Python, PyTorch version, CPU model, and full dependency list are documented within the Supplementary Material.

## References

Cheng, G., X. Yuan, X. Yao, K. Yan, Q. Zeng, X. Xie, and J. Han. 2023. "Towards Large-Scale Small Object Detection: Survey and Benchmarks." IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (1): 134–157.

Du, D., Y. Qi, H. Yu, Y. Yang, K. Duan, G. Li, W. Zhang, Q. Huang, and Q. Tian. 2018. "The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking." In Proceedings of the European Conference on Computer Vision (ECCV), 370–386.

Du, D., P. Zhu, L. Wen, X. Bian, H. Ling, Q. Hu, Q. Nie, H. Cheng, X. Liu, S. Lyu, et al. 2019. "VisDrone-DET2019: The Vision Meets Drone Object Detection in Image Challenge Results." In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), 213–226. doi:10.1109/ICCVW.2019.00030.

Feng, C., Y. Zhong, Y. Gao, M. R. Scott, and W. Huang. 2021. "TOOD: Task-Aligned One-Stage Object Detection." In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 3490–3499.

He, K., X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick. 2022. "Masked Autoencoders Are Scalable Vision Learners." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 16000–16009.

Jocher, G., A. Chaurasia, and J. Qiu. 2023. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics.

Khanam, R., and M. Hussain. 2024. "YOLOv11: An Overview of the Key Architectural Enhancements." arXiv:2410.17725.

Lin, T.-Y., M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick. 2014. "Microsoft COCO: Common Objects in Context." In Proceedings of the European Conference on Computer Vision (ECCV), 740–755. doi:10.1007/978-3-319-10602-1_48.

Zhou, W., and G. Gong. 2026. "MD-YOLO: A Multi-Scale Adaptive and Dual-Attention Enhanced YOLOv11 for Small Object Detection." Electronics 15 (10): 2099. https://doi.org/10.3390/electronics15102099. Source code of the implementation audited in this work: https://github.com/wqinmin975/yoloMD931 (commit: 451fe37055b2bdee8b5baf59f246f14856dc2524; accessed: 2026-10-10).

Pineau, J., P. Vincent-Lamarre, K. Sinha, V. Larivière, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and H. Larochelle. 2020. "Improving Reproducibility in Machine Learning Research (Dagstuhl Seminar 19242)." arXiv:2003.12206.

Redmon, J., S. Divvala, R. Girshick, and A. Farhadi. 2016. "You Only Look Once: Unified, Real-Time Object Detection." In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 779–788. doi:10.1109/CVPR.2016.91.

Tong, Z., Y. Chen, Z. Xu, and R. Yu. 2023. "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism." arXiv:2301.10051.

Xia, G.-S., X. Bai, J. Ding, Z. Zhu, S. Belongie, J. Luo, M. Datcu, M. Pelillo, and L. Zhang. 2018. "DOTA: A Large-Scale Dataset for Object Detection in Aerial Images." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 3974–3983. doi:10.1109/CVPR.2018.00418.

Xu, M., Z. Zhang, H. Hu, J. Wang, L. Wang, F. Wei, X. Bai, and Z. Liu. 2021. "End-to-End Semi-Supervised Object Detection with Soft Teacher." In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 3060–3069.

Zhang, H., Y. Wang, F. Dayoub, and N. Sünderhauf. 2021. "VarifocalNet: An IoU-Aware Dense Object Detector." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 8514–8523.
