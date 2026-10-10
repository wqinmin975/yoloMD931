"""
将真实 UAVDT (UAV-benchmark-M) 标注 + 抽样帧转换为跨域评估用的 YOLO 格式测试集。

UAVDT DET GT 格式 (9 字段, 官方 UAVDT-M 检测基准 gt.txt / gt_whole.txt):
  frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height,
  ignore_flag, occlusion, object_category
  - 第 7 字段 (1-indexed, 0-indexed=6) ignore_flag: 官方 evaluator 约定 ignore==0 的行不参与评测
    (evaluateTracking.m: gtdata(gtdata(:,7)==0,:)=[]); 在发布的 gt.txt 中该标志恒为 1, 故不丢弃任何行。
  - 第 9 字段 object_category: gt.txt 为 car-only 检测文件, 该字段恒为 -1 (占位符), 真实类别隐含为 car;
    gt_whole.txt 中则为真实类别 1=car / 2=truck / 3=bus / 4=others。
  - 第 8 字段 occlusion: 0/1/2/3 遮挡程度, 非类别。
  因此本脚本: ① 遇 object_category==-1 视为 car(1); ② 丢弃 ignore_flag==0 的行; ③ 仅保留 car/truck/bus。

跨域对齐: 标签文件里写入"源域 (VisDrone) 对齐"的类别索引,
再由 eval_cross_domain 的 map_src 把预测与 GT 同时桥接到目标域 (UAVDT 0/1/2):
  UAVDT car(1)  -> VisDrone car(3)  -> target 0
  UAVDT truck(2)-> VisDrone truck(5)-> target 1
  UAVDT bus(3)  -> VisDrone bus(8)  -> target 2
  (VisDrone van(4) 也映射到 target 0, 由 map_src 处理)

用法:
  python tools/prepare_uavdt.py \
      --raw_gt E:/visdrone_work/uavdt/raw/UAV-benchmark-M \
      --img_root E:/visdrone_work/uavdt/extracted/UAV-benchmark-M \
      --out E:/visdrone_work/uavdt/test
"""
import os
import argparse
import glob
from PIL import Image


# UAVDT 原生 class -> VisDrone 源域索引 (供 map_src 桥接)
UAVDT2SRC = {1: 3, 2: 5, 3: 8}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--raw_gt', required=True, help='含 Mxxxx/gt/gt.txt 的根目录')
    p.add_argument('--img_root', required=True, help='含 Mxxxx/img1/*.jpg 的解压根目录')
    p.add_argument('--out', required=True, help='输出 test/{images,labels}')
    return p.parse_args()


def main():
    a = parse_args()
    img_out = os.path.join(a.out, 'images')
    lbl_out = os.path.join(a.out, 'labels')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    # 收集所有序列
    seq_dirs = sorted(glob.glob(os.path.join(a.img_root, 'M*')))
    n_img = 0
    n_box = 0
    n_seq = 0

    for seq_path in seq_dirs:
        seq = os.path.basename(seq_path)
        gt_path = os.path.join(a.raw_gt, seq, 'gt', 'gt.txt')
        if not os.path.exists(gt_path):
            print(f"  [skip] 无 GT: {seq}")
            continue
        # 读 GT: frame -> list of (src_class, x,y,w,h)
        gt_by_frame = {}
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 9:
                    continue
                fid = int(parts[0])
                x, y, w, h = (float(parts[2]), float(parts[3]),
                              float(parts[4]), float(parts[5]))
                ignore_flag = int(parts[6])       # 第7字段(1-indexed): 官方 ignore 标志; ==0 表示该行不参与评测
                cls = int(parts[8])                # 第9字段 object_category (gt.txt 恒为 -1 占位)
                if ignore_flag == 0:
                    continue                       # 官方约定: 丢弃 ignore==0 的行 (gt.txt 恒为 1, 不丢弃)
                if cls == -1:
                    cls = 1                        # gt.txt car-only: 占位 -1 -> car(1)
                if cls not in UAVDT2SRC:
                    continue                       # 仅保留 car/truck/bus (others 跳过)
                src_cls = UAVDT2SRC[cls]
                gt_by_frame.setdefault(fid, []).append((src_cls, x, y, w, h))

        n_seq += 1
        img_dir = os.path.join(seq_path, 'img1')
        for img_name in sorted(os.listdir(img_dir)):
            if not img_name.lower().endswith('.jpg'):
                continue
            fid = int(img_name.replace('img', '').replace('.jpg', ''))
            boxes = gt_by_frame.get(fid)
            if not boxes:
                continue                        # 仅保留含有效目标的帧
            img_path = os.path.join(img_dir, img_name)
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
            except Exception as e:
                print(f"  [warn] 读图失败 {img_path}: {e}")
                continue
            # 写 YOLO 标签 (src 索引, cx cy w h 归一化)
            stem = f"{seq}_{fid:06d}"
            with open(os.path.join(lbl_out, stem + '.txt'), 'w') as lf:
                for (sc, x, y, w, h) in boxes:
                    x2 = x + w
                    y2 = y + h
                    cx = ((x + x2) / 2.0) / W
                    cy = ((y + y2) / 2.0) / H
                    nw = w / W
                    nh = h / H
                    if nw <= 0 or nh <= 0:
                        continue
                    lf.write(f"{sc} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                    n_box += 1
            # 放到 test/images (用唯一 stem 避免跨序列重名); 优先硬链接, 失败则复制
            dst = os.path.join(img_out, stem + '.jpg')
            if not os.path.exists(dst):
                try:
                    os.link(img_path, dst)
                except OSError:
                    import shutil
                    shutil.copy(img_path, dst)
            n_img += 1

    print(f"[done] sequences={n_seq} images={n_img} boxes={n_box} -> {a.out}")


if __name__ == '__main__':
    main()
