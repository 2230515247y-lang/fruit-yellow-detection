# -*- coding: utf-8 -*-
"""fruit_yellow 多口径评估入口（evaluate.py）

合并来源（核心算法逻辑未改动，保留原中文注释）：
  - map_eval2.py    -> stage coco：pycocotools COCOeval 统一口径 mAP（GT 由 YOLO labels 换算，预测需 --pred-dir）
  - val_check.py    -> stage val ：ultralytics 官方 val 口径复核指定 split 的 mAP50 与每类 AP50
  - validate_map.py -> stage geom：已解出坐标与原图标签的几何交叉验证（依赖 solve_src3.py / sift_solve.py）
  - metrics_all.py  -> stage metrics：四批原图 lap512/长边/短边/MP/bpp/pHash 指标与近重复对
仅把脚本内写死的绝对路径改为命令行参数（默认指向 D:/software/yolo）。

用法：
  python scripts/evaluate.py --stage val   --model D:\software\yolo\runs\fruit_yellow_v1\weights\best.pt
  python scripts/evaluate.py --stage coco  --pred-dir <预测txt目录> --split test
  python scripts/evaluate.py --stage geom
  python scripts/evaluate.py --stage metrics --metrics-src D:\Desktop\水果图片
"""
import argparse
import os
import sys

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow 多口径评估（COCO mAP / 官方 val / 几何交叉验证 / 原图质量指标）")
    p.add_argument("--stage", default="val", choices=["val", "coco", "geom", "metrics"],
                   help="val=ultralytics 官方 val 口径；coco=pycocotools COCOeval 统一口径 mAP；geom=拼接图标签几何交叉验证；metrics=原图质量量化指标")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo）")
    p.add_argument("--dataset-dir", default=None, help="fruit_yellow 数据根（默认 <root>/datasets/fruit_yellow）")
    p.add_argument("--aux", default=None, help="中间产物目录（val 运行目录、几何校验缓存，默认 <root>/aux）")
    p.add_argument("--model", default=None, help="val stage：待评估权重路径（必填）")
    p.add_argument("--data", default=None, help="val stage：data.yaml（默认 <dataset-dir>/dataset/data.yaml）")
    p.add_argument("--split", default="test", help="评估划分（默认 test）")
    p.add_argument("--tag", default="run", help="本次评估标签，用于输出目录/日志（默认 run）")
    p.add_argument("--pred-dir", default="", help="coco stage：预测 txt 目录（每图 <stem>.txt，格式 cls score x1 y1 x2 y2）")
    p.add_argument("--conf", type=float, default=0.001, help="val stage 置信度阈值（默认 0.001）")
    p.add_argument("--iou", type=float, default=0.7, help="val stage NMS IoU（默认 0.7）")
    p.add_argument("--imgsz", type=int, default=640, help="val stage 推理分辨率（默认 640）")
    p.add_argument("--workers", type=int, default=0, help="val stage DataLoader 进程数（默认 0）")
    p.add_argument("--metrics-src", default="D:/Desktop/水果图片",
                   help="metrics stage：四批原图根目录（默认 D:/Desktop/水果图片）")
    return p

ARGS, _EXTRA = _build_parser().parse_known_args()


ROOT = ARGS.root
FY_DIR = ARGS.dataset_dir or os.path.join(ROOT, "datasets", "fruit_yellow")
DS_DIR = os.path.join(FY_DIR, "dataset")
AUX = ARGS.aux or os.path.join(ROOT, "aux")
os.makedirs(AUX, exist_ok=True)

if ARGS.stage == "val":
    # -*- coding: utf-8 -*-
    """用 ultralytics 官方 val 口径复核 test split 的 mAP50"""
    import argparse
    from ultralytics import YOLO

    a = ARGS

    m = YOLO(a.model)
    r = m.val(data=ARGS.data or os.path.join(DS_DIR, 'data.yaml'),
              split=ARGS.split, conf=ARGS.conf, iou=ARGS.iou, imgsz=ARGS.imgsz, workers=ARGS.workers,
              verbose=False, plots=False,
              project=os.path.join(AUX, 'val_runs'),
              name=a.tag, exist_ok=True)

    print(f'[val:{a.tag}] mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}')
    for i, c in enumerate(r.box.ap_class_index):
        print(f'  AP50[{r.names[c]:<12}] = {r.box.ap50[i]:.4f}')

if ARGS.stage == "coco":
    # -*- coding: utf-8 -*-
    """
    统一口径 mAP 评估（pycocotools COCOeval, 默认参数 = IoU 0.5:0.05:0.95, maxDets 1/10/100）
    - GT : YOLO 格式 labels (cls cx cy w h 归一化) -> COCO bbox 像素
    - Pred: (cls score x1 y1 x2 y2 像素) -> COCO bbox
    AP@0.5 取 precision[iouIdx=0, :, :, area='all', maxDet=100]
    用法: python map_eval2.py --pred-dir <dir> --tag <name>
    """
    import argparse
    import os
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    DATASET = DS_DIR
    NAMES = ['apple_yellow', 'lemon', 'mango', 'pear_yellow', 'pomelo']
    EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


    def find_img(stem, img_dir):
        for e in EXTS:
            p = os.path.join(img_dir, stem + e)
            if os.path.exists(p):
                return p
        return None


    def build_gt(img_dir, gt_dir):
        images, anns, file_ids = [], [], {}
        stems = sorted(f[:-4] for f in os.listdir(gt_dir) if f.lower().endswith('.txt'))
        for i, stem in enumerate(stems, start=1):
            p = find_img(stem, img_dir)
            if p is None:
                print(f'[WARN] image not found: {stem}')
                continue
            w, h = Image.open(p).size
            file_ids[i] = stem
            images.append({'id': i, 'file_name': os.path.basename(p), 'width': w, 'height': h})
            with open(os.path.join(gt_dir, stem + '.txt')) as f:
                for line in f:
                    v = line.split()
                    if len(v) < 5:
                        continue
                    c, cx, cy, bw, bh = map(float, v[:5])
                    anns.append({
                        'id': len(anns) + 1, 'image_id': i, 'category_id': int(c) + 1,
                        'bbox': [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                        'area': bw * w * bh * h, 'iscrowd': 0,
                    })
        cats = [{'id': i + 1, 'name': n} for i, n in enumerate(NAMES)]
        return images, anns, cats, file_ids


    def load_preds(pred_dir, file_ids):
        dets, missing, empty = [], [], []
        cls_cnt = {}
        for i, stem in file_ids.items():
            fp = os.path.join(pred_dir, stem + '.txt')
            if not os.path.exists(fp):
                missing.append(stem)
                continue
            rows = 0
            with open(fp) as f:
                for line in f:
                    v = line.split()
                    if len(v) < 6:
                        continue
                    c, s, x1, y1, x2, y2 = map(float, v[:6])
                    dets.append({'image_id': i, 'category_id': int(c) + 1,
                                 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s})
                    cls_cnt[int(c)] = cls_cnt.get(int(c), 0) + 1
                    rows += 1
            if rows == 0:
                empty.append(stem)
        return dets, missing, empty, cls_cnt


    def main():
        a = ARGS

        img_dir = os.path.join(DATASET, 'images', ARGS.split)
        gt_dir = os.path.join(DATASET, 'labels', ARGS.split)
        images, anns, cats, file_ids = build_gt(img_dir, gt_dir)
        dets, missing, empty, cls_cnt = load_preds(a.pred_dir, file_ids)

        print(f'[eval:{a.tag}] gt_images={len(images)} gt_boxes={len(anns)} pred_boxes={len(dets)}')
        if missing:
            print(f'[eval:{a.tag}] MISSING pred for {len(missing)} imgs: {missing}')
        if empty:
            print(f'[eval:{a.tag}] EMPTY pred files: {len(empty)}')
        print(f'[eval:{a.tag}] pred cls dist: {dict(sorted(cls_cnt.items()))}')

        coco_gt = COCO()
        coco_gt.dataset = {'images': images, 'annotations': anns, 'categories': cats}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)

        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate()
        ev.accumulate()

        prec = ev.eval['precision']          # [T, R, K, A, M]
        ap50 = prec[0, :, :, 0, -1]          # IoU=0.5, area=all, maxDet=100
        per_cls = {}
        for k, n in enumerate(NAMES):
            v = ap50[:, k]
            v = v[v > -1]
            per_cls[n] = float(v.mean()) if v.size else float('nan')
        mAP50 = float(np.nanmean(list(per_cls.values())))

        print(f'\n===== {a.tag} =====')
        print(f'mAP50    = {mAP50:.4f}')
        for n in NAMES:
            print(f'  AP50[{n:<12}] = {per_cls[n]:.4f}')

        return mAP50


    if __name__ == '__main__':
        main()

if ARGS.stage == "geom":
    # -*- coding: utf-8 -*-
    """交叉验证：已解出的 (源图 stem, L, x0, y0) 是否能解释该格标签几何；并诊断 10 个未解格"""
    import os, sys, json, collections
    import numpy as np, cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import solve_src3 as S
        import sift_solve as SS
    except ImportError as _e:
        print('[SKIP] 本 stage 依赖 solve_src3.py / sift_solve.py（未随本次收拢复制），已跳过：%r' % (_e,))
        sys.exit(0)

    TEMP = AUX
    FY = FY_DIR
    RP = os.path.join(FY, 'raw_photos')
    PRE = os.path.join(FY, 'prelabel')
    SYN = os.path.join(FY, 'prelabel_synth')
    CELL, CANVAS = 480, 960
    SRC = json.load(open(os.path.join(TEMP, 'synth_src_final.json'), encoding='utf-8'))


    def prep_json(cls, stem, which):
        base = PRE if which == 'prelabel' else SYN
        p = os.path.join(base, cls, stem + '.json')
        if not os.path.exists(p):
            return None
        d = json.load(open(p, encoding='utf-8'))
        out = []
        for s in d.get('shapes', []):
            (x1, y1), (x2, y2) = s['points']
            out.append((s['label'], float(x1), float(y1), float(x2), float(y2)))
        return out


    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0


    print('=== 缓存坐标幅度 vs 当前文件尺寸（抽查已解出的 mango/pear 源） ===')
    tgt = {'mango': ['mango_009', 'mango_015', 'mango_017', 'mango_025', 'mango_026', 'mango_040', 'mango_042', 'mango_046'],
           'pear_yellow': ['pear_yellow_010', 'pear_yellow_029', 'pear_yellow_032', 'pear_yellow_035',
                           'pear_yellow_039', 'pear_yellow_040', 'pear_yellow_046', 'pear_yellow_048']}
    for cat, stems in tgt.items():
        for it in S.INDEX[cat]:
            st = os.path.splitext(it['file'])[0]
            if st in stems:
                mx = max((b[0] + b[2]) for b in it['all']) if it['all'] else 0
                my = max((b[1] + b[3]) for b in it['all']) if it['all'] else 0
                p = os.path.join(RP, cat, st + '.jpg')
                im = S.imread_u(p) if os.path.exists(p) else None
                print('  %-16s cache=%dx%d boxmax=(%.0f,%.0f) root=%s nbox=%d' % (
                    it['file'], it['w'], it['h'], mx, my,
                    ('%dx%d' % (im.shape[1], im.shape[0])) if im is not None else 'NA', len(it['all'])))

    print()
    print('=== 标签几何交叉验证（源图 prelabel json 框 -> 映射回格内 与 该格标签比 IoU） ===')
    stat = collections.defaultdict(collections.Counter)
    bad = []
    for stem, r in sorted(SRC.items()):
        for k, c in r['cells'].items():
            if c['state'] != 'ok':
                continue
            sstem = os.path.splitext(os.path.basename(c['file']))[0]
            sb = prep_json(c['cls'], sstem, 'prelabel')
            lab = prep_json(c['cls'], stem, 'prelabel_synth')
            if sb is None or lab is None:
                stat[c['method']]['nojson'] += 1
                continue
            L, x0, y0 = c['L'], c['x0'], c['y0']
            sc = CELL / float(L)
            cxo = 0 if k in ('0', '2') else CELL
            cyo = 0 if k in ('0', '1') else CELL
            mapped = []
            for (lb, x1, y1, x2, y2) in sb:
                mx1, my1 = cxo + (x1 - x0) * sc, cyo + (y1 - y0) * sc
                mx2, my2 = cxo + (x2 - x0) * sc, cyo + (y2 - y0) * sc
                ccx, ccy = (mx1 + mx2) / 2, (my1 + my2) / 2
                if cxo <= ccx < cxo + CELL and cyo <= ccy < cyo + CELL:
                    mapped.append((lb, mx1, my1, mx2, my2))
            celllabs = [(l, x1, y1, x2, y2) for (l, x1, y1, x2, y2) in lab
                        if cxo <= (x1 + x2) / 2 < cxo + CELL and cyo <= (y1 + y2) / 2 < cyo + CELL
                        and l == c['cls']]
            hmatch = 0
            for (l, x1, y1, x2, y2) in celllabs:
                best = max([iou((x1, y1, x2, y2), (a, b, cc, d)) for (_, a, b, cc, d) in mapped] or [0])
                hmatch += (best >= 0.5)
            rate = hmatch / len(celllabs) if celllabs else -1
            stat[c['method']][('hit' if rate >= 0.8 else 'weak')] += 1
            if rate < 0.8:
                bad.append((stem, k, c['cls'], sstem, c['method'], rate, len(celllabs), len(mapped), c['diff']))
    print('  统计:', {m: dict(v) for m, v in stat.items()})
    print('  弱匹配明细(%d):' % len(bad))
    for b in bad:
        print('    %-18s|%s %-13s src=%-18s %s rate=%.2f labs=%d mapped=%d diff=%.2f' % b)

    print()
    print('=== 10 个未解格：放宽阈值找最接近候选 ===')
    for stem, r in sorted(SRC.items()):
        for k, c in r['cells'].items():
            if c['state'] == 'ok' or c['cls'] != 'mango':
                continue
            img = S.imread_u(os.path.join(SYN, stem + '.jpg'))
            cell_pix = img[(0 if k in ('0', '1') else CELL):(0 if k in ('0', '1') else CELL) + CELL,
                           (0 if k in ('0', '2') else CELL):(0 if k in ('0', '2') else CELL) + CELL]
            res = []
            g = cv2.cvtColor(cell_pix, cv2.COLOR_BGR2GRAY)
            kp1, d1 = SS.SIFT.detectAndCompute(g, None)
            if d1 is None:
                print('  %s|%s 无特征' % (stem, k)); continue
            d1 = d1[:400]
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            import math
            for sub, path in SS.POOLS['mango']:
                kp2, d2 = SS.feats(path)
                if d2 is None:
                    continue
                ms = bf.match(d1, d2)
                if len(ms) < 10:
                    continue
                src = np.float32([kp1[x.queryIdx].pt for x in ms]).reshape(-1, 1, 2)
                dst = np.float32([kp2[x.trainIdx].pt for x in ms]).reshape(-1, 1, 2)
                M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if M is None:
                    continue
                sc_ = math.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
                if sc_ <= 0 or abs(math.degrees(math.atan2(M[1, 0], M[0, 0]))) > 3:
                    continue
                L = int(round(480 * sc_)); x0, y0 = int(round(M[0, 2])), int(round(M[1, 2]))
                loc = SS.best_local(path, L, x0, y0, cell_pix, rng=4)
                if loc:
                    res.append((loc[0], sub, os.path.basename(path), loc[2], loc[3], loc[4], len(ms)))
            res.sort()
            print('  %-18s|%s 候选%d' % (stem, k, len(res)))
            for t in res[:3]:
                print('       d=%.2f %-14s %-22s L=%d (%d,%d) m=%d' % (t[0], t[1], t[2], t[3], t[4], t[5], t[6]))

if ARGS.stage == "metrics":
    # -*- coding: utf-8 -*-
    """四批水果图片统一量化指标计算：lap512 / 长边 / 短边 / MP / bpp / pHash"""
    import os, json, math
    import numpy as np
    from PIL import Image
    import cv2

    BASE = ARGS.metrics_src
    BATCHES = {"芒果": "芒果", "梨子": "梨子", "柠檬": "柠檬", "黄苹果": "黄苹果"}
    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    OUT = AUX


    def phash64(gray):
        img = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(img)
        low = dct[:8, :8].flatten()
        vals = np.delete(low, 0)  # 去 DC
        med = np.median(vals)
        bits = (vals > med).astype(np.uint8)
        return bits


    def ham(a, b):
        return int(np.count_nonzero(a != b))


    def calc(path):
        size = os.path.getsize(path)
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode
            gray = np.array(im.convert("L"))
        long_side, short_side = max(w, h), min(w, h)
        mp = w * h / 1e6
        bpp = size / (w * h)
        # lap512：缩放到长边 512 后计算 Laplacian 方差
        scale = 512.0 / long_side
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        g512 = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        lap = float(cv2.Laplacian(g512, cv2.CV_64F).var())
        return {
            "file": os.path.basename(path),
            "ext": os.path.splitext(path)[1].lower().lstrip("."),
            "bytes": size,
            "w": w, "h": h,
            "long_side": long_side, "short_side": short_side,
            "mp": round(mp, 3), "bpp": round(bpp, 5),
            "lap512": round(lap, 2),
            "mode": mode,
            "ph": phash64(gray),
        }


    result = {}
    for name, folder in BATCHES.items():
        d = os.path.join(BASE, folder)
        files = [f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS]
        files.sort(key=lambda x: (len(x), x))
        items = []
        for f in files:
            p = os.path.join(d, f)
            try:
                r = calc(p)
                r["path"] = p
                items.append(r)
            except Exception as e:
                items.append({"file": f, "path": p, "error": str(e)})
        result[name] = items
        print(name, len(items))

    # 序列化（phash 转 list）
    ser = {}
    for k, items in result.items():
        ser[k] = []
        for it in items:
            it2 = dict(it)
            if "ph" in it2 and isinstance(it2["ph"], np.ndarray):
                it2["ph"] = it2["ph"].tolist()
            ser[k].append(it2)

    with open(os.path.join(OUT, "metrics_all_batches.json"), "w", encoding="utf-8") as f:
        json.dump(ser, f, ensure_ascii=False, indent=1)

    # 近重复对（批内）
    dup_report = {}
    for k, items in result.items():
        ok = [x for x in items if "ph" in x]
        pairs = []
        for i in range(len(ok)):
            for j in range(i + 1, len(ok)):
                d = ham(ok[i]["ph"], ok[j]["ph"])
                if d <= 14:
                    pairs.append({"a": ok[i]["file"], "b": ok[j]["file"], "ham": d})
        pairs.sort(key=lambda x: x["ham"])
        dup_report[k] = pairs
    with open(os.path.join(OUT, "dup_pairs.json"), "w", encoding="utf-8") as f:
        json.dump(dup_report, f, ensure_ascii=False, indent=1)

    for k, v in dup_report.items():
        print(k, "近重复对(<=14)：", len(v), "其中<=8：", sum(1 for p in v if p["ham"] <= 8))
