"""
把真实 VisDrone2019-DET 标注转换为模型期望的 YOLO 格式。

真实 VisDrone 标注格式（每行 8 列，逗号分隔）:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category>,<truncation>,<occlusion>
- 坐标为绝对像素 (x_left, y_top, w, h)
- category 取值 1..10 (1-indexed), 对应:
    1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
    6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor
- 模型内部类别为 0-indexed (10 类), 故 category-1。

输出 YOLO 格式（每行 5 列，空格分隔，归一化）:
    <class> <cx> <cy> <w> <h>

用法:
    python tools/prepare_visdrone.py \
        --root E:/visdrone_work \
        --splits VisDrone2019-DET-train VisDrone2019-DET-val
"""
import os
import cv2
import argparse


def convert_split(root, split):
    split_dir = os.path.join(root, split)
    img_dir = os.path.join(split_dir, "images")
    ann_dir = os.path.join(split_dir, "annotations")
    lbl_dir = os.path.join(split_dir, "labels")
    if not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
        print(f"[skip] {split}: 找不到 images/ 或 annotations/ (实际内容见 {split_dir})")
        return 0, 0
    os.makedirs(lbl_dir, exist_ok=True)

    img_files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    n_img = len(img_files)
    n_box = 0
    n_empty = 0
    for i, fn in enumerate(img_files):
        base = os.path.splitext(fn)[0]
        ann_path = os.path.join(ann_dir, base + '.txt')
        img_path = os.path.join(img_dir, fn)
        # 读取图像尺寸 (绝对坐标需要原始 W,H 归一化)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[warn] 无法读取图像: {img_path}")
            continue
        h0, w0 = img.shape[:2]

        out_lines = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 8:
                        continue
                    try:
                        x, y, w, h = (float(parts[0]), float(parts[1]),
                                      float(parts[2]), float(parts[3]))
                        score = float(parts[4])
                        cat = int(float(parts[5]))
                    except ValueError:
                        continue
                    # 过滤: 非法类别 / 零宽高 / (可选) 忽略区域
                    if cat < 1 or cat > 10:
                        continue
                    if w <= 0 or h <= 0:
                        continue
                    # VisDrone DET 中 score 列在 GT 里通常无意义; 这里保留全部有效框
                    cx = (x + w / 2.0) / w0
                    cy = (y + h / 2.0) / h0
                    wn = w / w0
                    hn = h / h0
                    if cx <= 0 or cy <= 0 or cx >= 1 or cy >= 1:
                        continue  # 越界框丢弃
                    out_lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
                    n_box += 1
        if not out_lines:
            n_empty += 1
        with open(os.path.join(lbl_dir, base + '.txt'), 'w') as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))
        if (i + 1) % 500 == 0:
            print(f"  {split}: {i+1}/{n_img} images, {n_box} boxes so far")
    print(f"[done] {split}: {n_img} images, {n_box} boxes, {n_empty} empty-label images")
    return n_img, n_box


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='E:/visdrone_work')
    p.add_argument('--splits', type=str, nargs='+',
                   default=['VisDrone2019-DET-train', 'VisDrone2019-DET-val'])
    args = p.parse_args()
    total_img = total_box = 0
    for s in args.splits:
        ni, nb = convert_split(args.root, s)
        total_img += ni
        total_box += nb
    print(f"\n=== 总计: {total_img} 图像, {total_box} 框 (YOLO 格式已写入各 split 的 labels/) ===")


if __name__ == '__main__':
    main()
