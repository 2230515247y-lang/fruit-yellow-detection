# -*- coding: utf-8 -*-
"""fruit_yellow 预标注生成与质量抽检（prelabel.py）

合并来源（核心算法逻辑未改动，保留原中文注释）：
  - prelabel_all.py   -> stage prelabel：yolo11x-seg(COCO) 高精度框 ∪ FastSAM-s 严格过滤框，IoU>0.5 去重合并
  - yolo2xany.py      -> stage yolo2xany：YOLO txt -> X-AnyLabeling 0.4.43 labelme 风格 json（只新增 json，不改 txt）
  - check_prelabel.py -> stage check：每类框数分布统计 + 随机抽样画框可视化
仅把脚本内写死的绝对路径改为命令行参数（默认指向 D:/software/yolo）。

用法：
  python scripts/prelabel.py --stage prelabel
  python scripts/prelabel.py --stage yolo2xany
  python scripts/prelabel.py --stage check
"""
import argparse
import os
import sys

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow 预标注生成与质量抽检")
    p.add_argument("--stage", default="prelabel", choices=["prelabel", "yolo2xany", "check"],
                   help="prelabel=生成全量预标注初稿（yolo11x-seg ∪ FastSAM-s）；yolo2xany=YOLO txt 转 X-AnyLabeling json；check=初稿质量抽检与抽样可视化")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo）")
    p.add_argument("--dataset-dir", default=None, help="fruit_yellow 数据根（默认 <root>/datasets/fruit_yellow）")
    p.add_argument("--aux", default=None, help="中间产物目录（txt 指纹、抽检可视化，默认 <root>/aux）")
    p.add_argument("--seg-model", default=None, help="分割模型权重（默认 <root>/yolo11x-seg.pt）")
    return p

ARGS, _EXTRA = _build_parser().parse_known_args()


ROOT = ARGS.root
FY_DIR = ARGS.dataset_dir or os.path.join(ROOT, "datasets", "fruit_yellow")
AUX = ARGS.aux or os.path.join(ROOT, "aux")
SEG_MODEL = ARGS.seg_model or os.path.join(ROOT, "yolo11x-seg.pt")
os.makedirs(AUX, exist_ok=True)

if ARGS.stage == "prelabel":
    # -*- coding: utf-8 -*-
    """全量预标注初稿生成
    策略：yolo11x-seg(COCO) 高精度框  ∪  FastSAM-s 严格过滤框，IoU>0.5 去重合并
    输出：D:\\software\\yolo\\datasets\\fruit_yellow\\prelabel\\  图片 + 同名 YOLO txt
    （类 id 按 CLASSES 顺序，供后续划分与 data.yaml 复用）
    """
    import os
    import shutil
    import sys
    import time
    import cv2
    import numpy as np
    from ultralytics import YOLO, FastSAM

    DS = os.path.join(FY_DIR, "raw_photos")
    OUTD = os.path.join(FY_DIR, "prelabel")
    YOLO_SEG = SEG_MODEL
    FS = os.path.join(ROOT, "FastSAM-s.pt")
    CLASSES = ["apple_yellow", "lemon", "mango", "pear_yellow", "pomelo"]
    IMG_EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prelabel_all.log")


    def log(msg):
        line = str(msg)
        print(line, flush=True)
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


    def mask_stats(bgr, mb):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        sel = mb > 0
        if sel.sum() == 0:
            return 0.0, 999.0
        ym = (h >= 10) & (h <= 45) & (s >= 45) & (v >= 60)
        return float((ym & sel).sum()) / float(sel.sum()), float(s[sel].std())


    def iou(a, b):
        xx1, yy1 = max(a[0], b[0]), max(a[1], b[1])
        xx2, yy2 = min(a[2], b[2]), min(a[3], b[3])
        iw, ih = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0


    def main():
        if os.path.exists(LOG):
            os.remove(LOG)
        os.makedirs(OUTD, exist_ok=True)
        yseg = YOLO(YOLO_SEG)
        fsam = FastSAM(FS)
        total_img = 0
        total_box = 0
        t0 = time.time()
        for ci, cls in enumerate(CLASSES):
            src = os.path.join(DS, cls)
            if not os.path.isdir(src):
                log(f"[WARN] 目录不存在: {src}")
                continue
            files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMG_EXT)
            log(f"[INFO] {cls}: {len(files)} 张")
            for f in files:
                p = os.path.join(src, f)
                img = cv2.imread(p)
                if img is None:
                    log(f"[FAIL] 读取失败 {p}")
                    continue
                H, W = img.shape[:2]
                stem = os.path.splitext(f)[0]
                out_img = os.path.join(OUTD, f"{cls}__{stem}{os.path.splitext(f)[1].lower()}")
                out_txt = os.path.join(OUTD, f"{cls}__{stem}.txt")
                try:
                    boxes = []

                    # A: yolo11x-seg
                    ra = yseg(p, device=0, imgsz=1280, conf=0.15, iou=0.6, verbose=False)[0]
                    if ra.boxes is not None and ra.masks is not None:
                        for b, m in zip(ra.boxes.xyxy.cpu().numpy(), ra.masks.data.cpu().numpy()):
                            mb = (m > 0.5).astype(np.uint8)
                            if mb.shape != (H, W):
                                mb = cv2.resize(mb, (W, H), interpolation=cv2.INTER_NEAREST)
                            if int(mb.sum()) < 0.0025 * H * W:
                                continue
                            boxes.append([float(v) for v in b])

                    # B: FastSAM-s 严格过滤补漏
                    rb = fsam(p, device=0, retina_masks=True, imgsz=1024, conf=0.25, iou=0.9, verbose=False)[0]
                    if rb.boxes is not None and rb.masks is not None:
                        bb = rb.boxes.xyxy.cpu().numpy()
                        ms = rb.masks.data.cpu().numpy()
                        for b, m in zip(bb, ms):
                            mb = (m > 0.5).astype(np.uint8)
                            if mb.shape != (H, W):
                                mb = cv2.resize(mb, (W, H), interpolation=cv2.INTER_NEAREST)
                            a = int(mb.sum())
                            if not (0.005 * H * W <= a <= 0.6 * H * W):
                                continue
                            x1, y1, x2, y2 = [float(v) for v in b]
                            if x2 - x1 <= 0 or y2 - y1 <= 0:
                                continue
                            if not (0.5 <= (x2 - x1) / (y2 - y1) <= 2.0):
                                continue
                            cnts, _ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not cnts:
                                continue
                            c = max(cnts, key=cv2.contourArea)
                            per = cv2.arcLength(c, True)
                            area = cv2.contourArea(c)
                            circ = 4 * np.pi * area / (per * per) if per > 0 else 0.0
                            if circ < 0.40:
                                continue
                            yr, sstd = mask_stats(img, mb)
                            if yr < 0.45 or sstd > 75:
                                continue
                            cand = [x1, y1, x2, y2]
                            if any(iou(cand, e) > 0.5 for e in boxes):
                                continue
                            boxes.append(cand)

                    # 写 txt
                    lines = []
                    for b in boxes:
                        x1, y1, x2, y2 = b
                        xc = max(0.0, min(1.0, (x1 + x2) / 2.0 / W))
                        yc = max(0.0, min(1.0, (y1 + y2) / 2.0 / H))
                        bw = max(0.0, min(1.0, (x2 - x1) / W))
                        bh = max(0.0, min(1.0, (y2 - y1) / H))
                        lines.append(f"{ci} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                    shutil.copy2(p, out_img)
                    with open(out_txt, "w", encoding="utf-8") as fh:
                        fh.write("\n".join(lines) + ("\n" if lines else ""))
                    total_img += 1
                    total_box += len(boxes)
                    log(f"[OK] {out_img} boxes={len(boxes)}")
                except Exception as e:
                    log(f"[ERR] {p} -> {e}")
        log(f"[DONE] 图片={total_img} 框={total_box} 耗时={time.time()-t0:.1f}s")


    if __name__ == "__main__":
        main()

if ARGS.stage == "yolo2xany":
    # -*- coding: utf-8 -*-
    """YOLO txt -> X-AnyLabeling(labelme 风格) json
    只新增 .json，绝不改动/删除原 .txt
    JSON 结构严格对齐 X-AnyLabeling 0.4.43 的 LabelFile.save()
    """
    import os
    import json
    import hashlib
    import collections
    from PIL import Image

    PD = os.path.join(FY_DIR, "prelabel")
    VERSION = "0.4.43"
    EXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    TMP = AUX


    def load_classes():
        # classes.txt 含 UTF-8 BOM，按 utf-8-sig 读取以剥离
        with open(os.path.join(PD, "classes.txt"), encoding="utf-8-sig") as fh:
            return [ln.strip().lstrip("\ufeff") for ln in fh if ln.strip()]


    def main():
        classes = load_classes()
        print("类别表:", classes)
        imgs = sorted(f for f in os.listdir(PD) if os.path.splitext(f)[1].lower() in EXT)
        md5_before = {}
        stat = collections.OrderedDict()
        boxes_total = 0
        n_skip_exist = 0
        errors = []

        for f in imgs:
            stem, ext = os.path.splitext(f)
            tp = os.path.join(PD, stem + ".txt")
            jp = os.path.join(PD, stem + ".json")
            p = os.path.join(PD, f)
            if not os.path.exists(tp):
                errors.append((f, "缺同名 txt"))
                continue
            # 记录 txt 指纹
            with open(tp, "rb") as fh:
                raw = fh.read()
            md5_before[f] = hashlib.md5(raw).hexdigest()

            if os.path.exists(jp):
                n_skip_exist += 1  # 本次任务自产的 json，允许覆盖重写

            im = Image.open(p)
            W, H = im.size
            im.close()

            shapes = []
            for ln in raw.decode("utf-8").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                ps = ln.split()
                if len(ps) != 5:
                    errors.append((f, "非法行: " + ln[:50]))
                    continue
                cid = int(ps[0])
                xc, yc, bw, bh = (float(v) for v in ps[1:])
                x1 = max(0.0, min(float(W), (xc - bw / 2.0) * W))
                y1 = max(0.0, min(float(H), (yc - bh / 2.0) * H))
                x2 = max(0.0, min(float(W), (xc + bw / 2.0) * W))
                y2 = max(0.0, min(float(H), (yc + bh / 2.0) * H))
                if x2 <= x1:
                    x2 = min(float(W), x1 + 1.0)
                if y2 <= y1:
                    y2 = min(float(H), y1 + 1.0)
                shapes.append({
                    "label": classes[cid],
                    "text": "",
                    "points": [[round(x1, 2), round(y1, 2)], [round(x2, 2), round(y2, 2)]],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {},
                })

            data = {
                "version": VERSION,
                "flags": {},
                "shapes": shapes,
                "imagePath": f,
                "imageData": None,
                "imageHeight": H,
                "imageWidth": W,
            }
            with open(jp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)

            cls_name = f.split("__")[0]
            c = stat.setdefault(cls_name, [0, 0])
            c[0] += 1
            c[1] += len(shapes)
            boxes_total += len(shapes)

        print("已生成 json: %d 张 | 已存在跳过: %d" % (len(imgs) - n_skip_exist, n_skip_exist))
        print("框总数(写入): %d" % boxes_total)
        for k, v in stat.items():
            print("   %-14s 图=%d 框=%d" % (k, v[0], v[1]))
        print("异常:", len(errors), errors[:10])

        with open(os.path.join(TMP, "txt_md5_before.json"), "w", encoding="utf-8") as fh:
            json.dump(md5_before, fh, ensure_ascii=False, indent=1)
        print("txt 指纹已存: txt_md5_before.json")


    if __name__ == "__main__":
        main()

if ARGS.stage == "check":
    # -*- coding: utf-8 -*-
    """预标注初稿质量抽检：统计每类框数分布 + 随机抽样画框可视化"""
    import os
    import random
    import statistics
    import cv2
    import numpy as np

    PD = os.path.join(FY_DIR, "prelabel")
    OUT = os.path.join(AUX, "check")
    CLASSES = ["apple_yellow", "lemon", "mango", "pear_yellow", "pomelo"]
    os.makedirs(OUT, exist_ok=True)
    random.seed(7)

    stats = {}
    zero = {}
    for cls in CLASSES:
        files = sorted(f for f in os.listdir(PD) if f.startswith(cls + "__") and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
        counts = []
        for f in files:
            stem = os.path.splitext(f)[0]
            tp = os.path.join(PD, stem + ".txt")
            n = 0
            if os.path.exists(tp):
                with open(tp, encoding="utf-8") as fh:
                    n = len([ln for ln in fh if ln.strip()])
            counts.append(n)
        stats[cls] = counts
        zero[cls] = sum(1 for c in counts if c == 0)
        print(f"{cls:14s} 图={len(files):3d} 框总数={sum(counts):4d} 均值={sum(counts)/max(1,len(counts)):.1f} 中位={statistics.median(counts):.0f} 最多={max(counts)} 零框图={zero[cls]}")

    # 抽样可视化
    for cls in CLASSES:
        files = sorted(f for f in os.listdir(PD) if f.startswith(cls + "__") and os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
        for f in random.sample(files, min(2, len(files))):
            p = os.path.join(PD, f)
            img = cv2.imread(p)
            if img is None:
                continue
            H, W = img.shape[:2]
            tp = os.path.join(PD, os.path.splitext(f)[0] + ".txt")
            vis = img.copy()
            n = 0
            if os.path.exists(tp):
                with open(tp, encoding="utf-8") as fh:
                    for ln in fh:
                        parts = ln.split()
                        if len(parts) != 5:
                            continue
                        _, xc, yc, bw, bh = parts
                        xc, yc, bw, bh = float(xc) * W, float(yc) * H, float(bw) * W, float(bh) * H
                        x1, y1 = int(xc - bw / 2), int(yc - bh / 2)
                        x2, y2 = int(xc + bw / 2), int(yc + bh / 2)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        n += 1
            cv2.imwrite(os.path.join(OUT, f"{cls}__{os.path.splitext(f)[0]}__n{n}.jpg"), vis)
    print("OUT:", OUT)
