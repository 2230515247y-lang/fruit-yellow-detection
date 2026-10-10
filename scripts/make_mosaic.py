# -*- coding: utf-8 -*-
"""fruit_yellow 拼接图合成与标注现场构建（make_mosaic.py）

合并来源（核心算法逻辑未改动，保留原中文注释）：
  - mosaic_v5.py            -> stage mosaic：真实照片区域拼接合成（索引召回 + 检测补标 + 硬性质检）
  - holdout_mosaic_v2.py    -> stage holdout：仅用 dataset/images/val|test 内源图，标签由 GT 几何映射，零漏标
  - synth_prelabel_build.py -> stage prelabel-synth：把 raw_photos/<类>/拼接 顶层图复制进 prelabel_synth 并生成 X-AnyLabeling json
仅把脚本内写死的绝对路径改为命令行参数（默认指向 D:/software/yolo）。

用法：
  python scripts/make_mosaic.py --stage mosaic --target-cat lemon --num 12
  python scripts/make_mosaic.py --stage holdout --num-mosaic 8
  python scripts/make_mosaic.py --stage prelabel-synth
"""
import argparse
import os
import sys

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow 拼接图合成（真实照片区域拼接 + 零漏标）")
    p.add_argument("--stage", default="mosaic", choices=["mosaic", "holdout", "prelabel-synth"],
                   help="mosaic=检测器辅助的 v5 式合成；holdout=仅用 val/test 源图的 holdout 合成；prelabel-synth=把 raw_photos 拼接图搬入 prelabel_synth 标注现场")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo）")
    p.add_argument("--dataset-dir", default=None, help="fruit_yellow 数据根（默认 <root>/datasets/fruit_yellow）")
    p.add_argument("--aux", default=None, help="中间产物目录（检测器索引缓存 frame_index3.json、可视化输出，默认 <root>/aux）")
    p.add_argument("--seg-model", default=None, help="分割模型权重（默认 <root>/yolo11x-seg.pt）")
    p.add_argument("--target-cat", default="apple_yellow", help="mosaic stage：目标类别（该类的框在每格中占多数）")
    p.add_argument("--num", type=int, default=12, help="mosaic stage：生成拼接图数量（默认 12）")
    p.add_argument("--num-mosaic", type=int, default=8, help="holdout stage：每个 split 生成拼接图数量（默认 8）")
    p.add_argument("--holdout-out", default=None, help="holdout stage 输出目录（默认 <dataset-dir>/prelabel_synth_holdout）")
    return p

ARGS, _EXTRA = _build_parser().parse_known_args()


ROOT = ARGS.root
FY_DIR = ARGS.dataset_dir or os.path.join(ROOT, "datasets", "fruit_yellow")
DS_DIR = os.path.join(FY_DIR, "dataset")
AUX = ARGS.aux or os.path.join(ROOT, "aux")
SEG_MODEL = ARGS.seg_model or os.path.join(ROOT, "yolo11x-seg.pt")
HOLDOUT_OUT = ARGS.holdout_out or os.path.join(FY_DIR, "prelabel_synth_holdout")
os.makedirs(AUX, exist_ok=True)

if ARGS.stage == "mosaic":
    # -*- coding: utf-8 -*-
    """合成 v5（正式版）：真实照片区域拼接 + 零漏标
    相对 v4b 的修正：
     1. 索引召回放宽 conf 0.18（质量过滤仍用于挑锚框）
     2. 窗口内"所有完整水果框"一律写入标签（不再按尺寸丢弃）
     3. 框需距窗口边 ≥12px，避免被羽化边缘切虚
     4. 生成后用检测器复核，凡置信度≥0.35 且与已写标签 IoU<0.30 的框 => 自动补标
     5. 硬性质检：复核不通过（存在未解释的检出框）则重生成，最多 3 次
    用法: python mosaic_v5.py <target_cat> [N]
    """
    import os, sys, cv2, json, random, numpy as np
    from ultralytics import YOLO

    BASE = AUX
    SRC = os.path.join(FY_DIR, 'raw_photos')
    CATS = ['apple_yellow', 'lemon', 'mango', 'pear_yellow', 'pomelo']
    CLS = {c: i for i, c in enumerate(CATS)}
    FRUIT = {46, 47, 49}
    TAGMAP = {'apple_yellow': 'apple', 'lemon': 'lemon', 'mango': 'mango',
              'pear_yellow': 'pear', 'pomelo': 'pomelo'}
    TARGET = ARGS.target_cat
    N = ARGS.num
    OUT = os.path.join(BASE, 'mosaic_v5_' + TARGET)
    TAG = TAGMAP[TARGET]
    CACHE = os.path.join(BASE, 'frame_index3.json')
    os.makedirs(OUT, exist_ok=True)
    random.seed(abs(hash(TARGET)) % 99991 + 31)
    CANVAS = 960
    CELL = CANVAS // 2
    FEATHER = 10
    EDGE_KEEP = 14


    def imread_u(p, flag=cv2.IMREAD_COLOR):
        buf = np.fromfile(p, dtype=np.uint8)
        return cv2.imdecode(buf, flag) if buf.size else None


    def imwrite_u(p, img):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if ok:
            buf.tofile(p)


    def fill_holes(mk):
        ff = mk.astype(np.uint8).copy()
        m2 = np.zeros((mk.shape[0] + 2, mk.shape[1] + 2), np.uint8)
        cv2.floodFill(ff, m2, (0, 0), 255)
        return cv2.bitwise_or(mk.astype(np.uint8), cv2.bitwise_not(ff))


    def quality(mk, img):
        area = int(mk.sum())
        if area <= 0:
            return None
        ys, xs = np.where(mk)
        bw = int(xs.max() - xs.min() + 1); bh = int(ys.max() - ys.min() + 1)
        if bw < 10 or bh < 10:
            return None
        good = True
        if not (0.6 <= bw / float(bh) <= 1.7) or area / float(bw * bh) < 0.68:
            good = False
        cnts = cv2.findContours(mk.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if cnts:
            peri = cv2.arcLength(max(cnts, key=cv2.contourArea), True)
            if 4 * np.pi * area / (peri * peri + 1e-6) < 0.70:
                good = False
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        if float(np.median(hsv[:, :, 1][mk])) < 50:
            good = False
        return (int(xs.min()), int(ys.min()), bw, bh, good)


    def build_index():
        det = YOLO(SEG_MODEL)
        data = {}
        for cat in CATS:
            d = os.path.join(SRC, cat)
            items = []
            for f in sorted(x for x in os.listdir(d)
                            if x.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
                            and os.path.isfile(os.path.join(d, x))):
                img = imread_u(os.path.join(d, f))
                if img is None:
                    continue
                h, w = img.shape[:2]
                r = det(img, device='0', conf=0.18, iou=0.45, verbose=False)[0]
                if r.boxes is None or len(r.boxes) == 0 or r.masks is None:
                    continue
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                masks = r.masks.data.cpu().numpy()
                allb, goodb = [], []
                for i, c in enumerate(cls):
                    if c not in FRUIT:
                        continue
                    mk = masks[i]
                    if mk.shape[:2] != (h, w):
                        mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)
                    mk = fill_holes(mk > 0.5)
                    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    mk = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_CLOSE, ker, 2)
                    q = quality(mk > 0, img)
                    if q is None:
                        continue
                    allb.append([q[0], q[1], q[2], q[3], float(cf[i])])
                    if q[4] and q[0] > 1 and q[1] > 1 and q[0] + q[2] < w - 2 and q[1] + q[3] < h - 2 \
                            and max(q[2], q[3]) >= 0.09 * min(w, h):
                        goodb.append([q[0], q[1], q[2], q[3]])
                if allb:
                    items.append({'file': f, 'w': w, 'h': h, 'all': allb, 'good': goodb})
            data[cat] = items
            print('%-14s 有检出图 %d' % (cat, len(items)), flush=True)
        with open(CACHE, 'w', encoding='utf-8') as fp:
            json.dump(data, fp)
        return data


    if os.path.exists(CACHE):
        with open(CACHE, encoding='utf-8') as fp:
            index = json.load(fp)
        print('索引缓存命中')
    else:
        index = build_index()

    usable = {c: [it for it in index[c] if it['good']] for c in CATS}
    print('可做锚的图:', {c: len(usable[c]) for c in CATS}, flush=True)
    cache_img = {}


    def get_img(cat, f):
        k = cat + '/' + f
        if k not in cache_img:
            cache_img[k] = imread_u(os.path.join(SRC, cat, f))
        return cache_img[k]


    def pick_cell(cat):
        cands = usable.get(cat, [])
        if not cands:
            return None, None, None
        for _ in range(100):
            it = random.choice(cands)
            img = get_img(cat, it['file'])
            if img is None:
                continue
            h, w = img.shape[:2]
            bx, by, bw, bh = random.choice(it['good'])
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            L = int(max(bw, bh) * random.uniform(1.6, 2.2))
            L = min(L, w, h, CANVAS)
            if L < 300:
                L = min(300, w, h)
            if L < 240:
                continue
            x0 = int(min(max(0, cx - L / 2), w - L))
            y0 = int(min(max(0, cy - L / 2), h - L))
            inside, bad = [], False
            for b in it['all']:
                px, py, pw, ph = b[0], b[1], b[2], b[3]
                if px + pw <= x0 or px >= x0 + L or py + ph <= y0 or py >= y0 + L:
                    continue
                if px < x0 + EDGE_KEEP or py < y0 + EDGE_KEEP or \
                   px + pw > x0 + L - EDGE_KEEP or py + ph > y0 + L - EDGE_KEEP:
                    bad = True
                    break
                inside.append((px - x0, py - y0, pw, ph))
            if bad or not inside:
                continue
            return img[y0:y0 + L, x0:x0 + L], L, inside
        return None, None, None


    def blend(canvas, patch, x0, y0):
        p = cv2.resize(patch, (CELL, CELL), interpolation=cv2.INTER_AREA)
        a = np.ones((CELL, CELL), np.float32)
        f = FEATHER
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        a[:f, :] *= ramp[:, None]; a[-f:, :] *= ramp[::-1, None]
        a[:, :f] *= ramp[None, :]; a[:, -f:] *= ramp[::-1][None, :]
        roi = canvas[y0:y0 + CELL, x0:x0 + CELL].astype(np.float32)
        canvas[y0:y0 + CELL, x0:x0 + CELL] = (p.astype(np.float32) * a[:, :, None]
                                              + roi * (1 - a[:, :, None])).astype(np.uint8)


    def base_bg():
        c = random.choice([c for c in CATS if index[c]])
        it = random.choice(index[c])
        im = get_img(c, it['file'])
        if im is None:
            return np.full((CANVAS, CANVAS, 3), 205, np.uint8)
        bg = cv2.resize(im, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(bg, (61, 61), 0)


    det = YOLO(SEG_MODEL)
    others = [c for c in CATS if c != TARGET]


    def iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1])
        x1, y1 = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        iw, ih = max(0, x1 - x0), max(0, y1 - y0)
        inter = iw * ih
        u = a[2] * a[3] + b[2] * b[3] - inter
        return inter / u if u > 0 else 0


    def detect_boxes(img):
        r = det(img, device='0', conf=0.30, iou=0.45, verbose=False)[0]
        out = []
        if r.boxes is not None and len(r.boxes):
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            xy = r.boxes.xyxy.cpu().numpy()
            for i, c in enumerate(cl):
                if c in FRUIT and cf[i] >= 0.35:
                    out.append((float(xy[i][0]), float(xy[i][1]), float(xy[i][2] - xy[i][0]), float(xy[i][3] - xy[i][1])))
        return out


    final = []
    for n in range(1, N + 1):
        name = 'mosaic%s_%02d' % (TAG, n)
        for attempt in range(3):
            canvas = np.clip(base_bg().astype(np.float32) * random.uniform(0.92, 1.08), 0, 255).astype(np.uint8)
            plan = [TARGET, TARGET] + [random.choice(others), random.choice(others)]
            random.shuffle(plan)
            cells = [(0, 0), (CELL, 0), (0, CELL), (CELL, CELL)]
            labels, cellcat, seen = [], [], set()
            for k, cat in enumerate(plan):
                got = None
                for _t in range(20):
                    patch, L, boxes = pick_cell(cat)
                    if patch is None:
                        continue
                    key = (cat, tuple(boxes[0]))
                    if key in seen:
                        continue
                    seen.add(key); got = (patch, L, boxes); break
                if got is None:
                    continue
                patch, L, boxes = got
                x0, y0 = cells[k]
                blend(canvas, patch, x0, y0)
                sc = CELL / float(L)
                cellcat.append((k, cat, x0, y0, sc))
                for (px, py, pw, ph) in boxes:
                    gx = min(max(x0 + px * sc, x0), CANVAS - max(10.0, pw * sc))
                    gy = min(max(y0 + py * sc, y0), CANVAS - max(10.0, ph * sc))
                    gw = max(10.0, pw * sc); gh = max(10.0, ph * sc)
                    labels.append([CLS[cat], gx, gy, gw, gh])
            # 检测补标
            added = 0
            for b in detect_boxes(canvas):
                if all(iou(b, (l[1], l[2], l[3], l[4])) < 0.30 for l in labels):
                    # 归属到所在格，用该格目录类别
                    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                    cat = None
                    for (k, c, x0, y0, sc) in cellcat:
                        if x0 <= cx <= x0 + CELL and y0 <= cy <= y0 + CELL:
                            cat = c; break
                    if cat is None:
                        continue
                    labels.append([CLS[cat], b[0], b[1], b[2], b[3]])
                    added += 1
            # 硬质检：补标后应无未匹配检出框
            residual = sum(1 for b in detect_boxes(canvas)
                           if all(iou(b, (l[1], l[2], l[3], l[4])) < 0.30 for l in labels))
            if residual == 0 or attempt == 2:
                break
        imwrite_u(os.path.join(OUT, name + '.jpg'), canvas)
        with open(os.path.join(OUT, name + '.txt'), 'w') as fp:
            for c, x, y, w, h in labels:
                fp.write('%d %.6f %.6f %.6f %.6f\n' % (c, (x + w / 2) / CANVAS, (y + h / 2) / CANVAS, w / CANVAS, h / CANVAS))
        vis = canvas.copy()
        for c, x, y, w, h in labels:
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
            cv2.putText(vis, CATS[c].split('_')[0][:4], (int(x), int(y) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        ok, buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 93])
        buf.tofile(os.path.join(OUT, name + '_vis.jpg'))
        final.append((name, len(labels), added, residual))
        print('%s 格=%d 标签=%d 补标=%d 残余未匹配=%d (第%d次)' % (name, len(cellcat), len(labels), added, residual, attempt + 1), flush=True)

    print('\n汇总: 图数=%d 标签总数=%d 平均标签=%.1f 补标总数=%d' %
          (len(final), sum(f[1] for f in final), np.mean([f[1] for f in final]), sum(f[2] for f in final)))
    print('OUT:', OUT)

if ARGS.stage == "holdout":
    # -*- coding: utf-8 -*-
    """holdout 拼接图生成 v2：仅用 dataset/images/val 或 test 内源图，2x2 / 960x960，风格对齐 mosaic_v5
    标签全部由源图 GT 标签几何映射而来（无检测器参与，标签精确）
    TierA = v5 式聚合裁切（无缝羽化拼接）；TierB = 整图成格（保证 4 格齐全、零漏标）
    """
    import os, json, random, collections
    import numpy as np
    import cv2

    DS = DS_DIR
    OUT = HOLDOUT_OUT
    TEMP = AUX
    CATS = ['apple_yellow', 'lemon', 'mango', 'pear_yellow', 'pomelo']
    CANVAS, CELL, FEATHER = 960, 480, 10
    N_MOSAIC = ARGS.num_mosaic
    SEED = 20261007
    VIS_DIR = os.path.join(TEMP, 'holdout_vis')


    def imread_u(p, flag=cv2.IMREAD_COLOR):
        buf = np.fromfile(p, dtype=np.uint8)
        return cv2.imdecode(buf, flag) if buf.size else None


    def imwrite_u(p, img, q=93):
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            buf.tofile(p)


    def load_pool(split):
        idir = os.path.join(DS, 'images', split)
        ldir = os.path.join(DS, 'labels', split)
        pool = collections.defaultdict(list)
        for f in sorted(os.listdir(idir)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                continue
            stem = os.path.splitext(f)[0]
            cat = next((c for c in CATS if stem.startswith(c)), None)
            if cat is None:
                print('  [警告] 源图 stem 无法归属类别，跳过:', f); continue
            img = imread_u(os.path.join(idir, f))
            if img is None:
                print('  [警告] 读取失败，跳过:', f); continue
            h, w = img.shape[:2]
            boxes = []
            with open(os.path.join(ldir, stem + '.txt'), encoding='utf-8') as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    p = line.split()
                    k = int(p[0]); cx, cy, bw, bh = map(float, p[1:5])
                    c = CATS[k]
                    if c != cat:
                        print('  [警告] 标签类别与目录类别不一致:', stem, c)
                    boxes.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h))
            pool[cat].append({'file': f, 'stem': stem, 'img': img, 'w': w, 'h': h, 'boxes': boxes})
        return pool


    def tier_a(item, rnd):
        """v5 式：环绕某框的方形裁切，框全部完整且留边"""
        W, H, boxes = item['w'], item['h'], item['boxes']
        order = list(range(len(boxes))); rnd.shuffle(order)
        for ai in order:
            _, ax, ay, aw, ah = boxes[ai]
            for factor in (2.3, 2.1, 1.9, 1.7, 1.5, 1.35):
                for _ in range(4):
                    L = int(max(aw, ah) * factor * rnd.uniform(0.97, 1.03))
                    L = min(L, W, H)
                    if L < 220:
                        continue
                    cx = ax + aw / 2.0 + rnd.uniform(-0.05, 0.05) * L
                    cy = ay + ah / 2.0 + rnd.uniform(-0.05, 0.05) * L
                    x0 = int(min(max(0, cx - L / 2.0), W - L))
                    y0 = int(min(max(0, cy - L / 2.0), H - L))
                    sc = CELL / float(L)
                    m = max(14.0, 12.0 / sc)
                    inside, bad = [], False
                    for (c, px, py, pw, ph) in boxes:
                        if px + pw <= x0 or px >= x0 + L or py + ph <= y0 or py >= y0 + L:
                            continue
                        if px < x0 + m or py < y0 + m or px + pw > x0 + L - m or py + ph > y0 + L - m:
                            bad = True; break
                        inside.append((c, px - x0, py - y0, pw, ph))
                    if bad or not inside:
                        continue
                    return {'mode': 'crop-A', 'x0': x0, 'y0': y0, 'cw': L, 'ch': L,
                            'sx': sc, 'sy': sc, 'inside': inside}
        return None


    def tier_b(item):
        """整图/大窗成格：保证源图内所有 GT 框完整可见、零漏标"""
        W, H, boxes = item['w'], item['h'], item['boxes']
        ux0 = min(b[1] for b in boxes); uy0 = min(b[2] for b in boxes)
        ux1 = max(b[1] + b[3] for b in boxes); uy1 = max(b[2] + b[4] for b in boxes)
        uw, uh = ux1 - ux0, uy1 - uy0
        for m in (16.0, 10.0, 6.0, 3.0, 0.0):
            L = int(np.ceil(max(uw, uh) + 2 * m))
            if L > min(W, H) or L < 120:
                continue
            x0 = int(min(max(0, (ux0 + ux1) / 2.0 - L / 2.0), W - L))
            y0 = int(min(max(0, (uy0 + uy1) / 2.0 - L / 2.0), H - L))
            inside, bad = [], False
            for (c, px, py, pw, ph) in boxes:
                if px < x0 or py < y0 or px + pw > x0 + L or py + ph > y0 + L:
                    bad = True; break
                inside.append((c, px - x0, py - y0, pw, ph))
            if bad or not inside:
                continue
            return {'mode': 'crop-B', 'x0': x0, 'y0': y0, 'cw': L, 'ch': L,
                    'sx': CELL / float(L), 'sy': CELL / float(L), 'inside': inside}
        # 兜底：整图（含非正方，等比独立缩放）
        inside = [(c, px, py, pw, ph) for (c, px, py, pw, ph) in boxes]
        return {'mode': 'full', 'x0': 0, 'y0': 0, 'cw': W, 'ch': H,
                'sx': CELL / float(W), 'sy': CELL / float(H), 'inside': inside}


    def blend(canvas, patch, x0, y0):
        p = cv2.resize(patch, (CELL, CELL), interpolation=cv2.INTER_AREA)
        a = np.ones((CELL, CELL), np.float32)
        f = FEATHER
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        a[:f, :] *= ramp[:, None]; a[-f:, :] *= ramp[::-1, None]
        a[:, :f] *= ramp[None, :]; a[:, -f:] *= ramp[::-1][None, :]
        roi = canvas[y0:y0 + CELL, x0:x0 + CELL].astype(np.float32)
        canvas[y0:y0 + CELL, x0:x0 + CELL] = (p.astype(np.float32) * a[:, :, None]
                                              + roi * (1 - a[:, :, None])).astype(np.uint8)


    def base_bg(pool, rnd):
        c = rnd.choice([cat for cat in CATS if pool[cat]])
        it = rnd.choice(pool[c])
        bg = cv2.resize(it['img'], (CANVAS, CANVAS), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(bg, (61, 61), 0)


    os.makedirs(OUT, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    meta = {'val': [], 'test': []}
    mode_stat = collections.Counter()

    with open(os.path.join(OUT, 'classes.txt'), 'w', encoding='utf-8-sig') as fp:
        fp.write('\n'.join(CATS) + '\n')

    for si, split in enumerate(('val', 'test')):
        rnd = random.Random(SEED + si)
        pool = load_pool(split)
        print('\n== %s 源图池: %s  框数=%s' % (
            split, {c: len(pool[c]) for c in CATS},
            {c: sum(len(x['boxes']) for x in pool[c]) for c in CATS}))
        queues = {}
        for c in CATS:
            q = list(pool[c]); rnd.shuffle(q); queues[c] = q
        usage = collections.Counter()
        outdir = os.path.join(OUT, split)
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'classes.txt'), 'w', encoding='utf-8-sig') as fp:
            fp.write('\n'.join(CATS) + '\n')

        for n in range(1, N_MOSAIC + 1):
            order = sorted(CATS, key=lambda c: (usage[c], rnd.random()))
            chosen = order[:4]
            canvas = np.clip(base_bg(pool, rnd).astype(np.float32) * rnd.uniform(0.92, 1.08), 0, 255).astype(np.uint8)
            cells = [(0, 0), (CELL, 0), (0, CELL), (CELL, CELL)]
            rnd.shuffle(cells)
            shapes, used = [], []
            for k, cat in enumerate(chosen):
                # 队列轮转取图（池耗尽则循环复用，保证 4 格齐全）
                q = queues[cat]
                it, plan = None, None
                for _ in range(3):
                    if not q:
                        q = list(pool[cat]); rnd.shuffle(q); queues[cat] = q
                    cand = q.pop(0)
                    plan = tier_a(cand, rnd) or tier_b(cand)
                    if plan is not None:
                        it = cand; break
                if plan is None:
                    print('  [警告] %s 无可用成格方案' % cat); continue
                patch = it['img'][plan['y0']:plan['y0'] + plan['ch'], plan['x0']:plan['x0'] + plan['cw']]
                px0, py0 = cells[k]
                blend(canvas, patch, px0, py0)
                cell_boxes = []
                for (c, bx, by, bw, bh) in plan['inside']:
                    gx, gy = px0 + bx * plan['sx'], py0 + by * plan['sy']
                    gw, gh = bw * plan['sx'], bh * plan['sy']
                    shapes.append({'label': c, 'text': '',
                                   'points': [[round(gx, 2), round(gy, 2)],
                                              [round(min(gx + gw, float(CANVAS)), 2),
                                               round(min(gy + gh, float(CANVAS)), 2)]],
                                   'group_id': None, 'shape_type': 'rectangle', 'flags': {}})
                    cell_boxes.append(c)
                mode_stat[plan['mode']] += 1
                used.append({'cell': k, 'cat': cat, 'src': it['stem'], 'file': it['file'],
                             'src_size': [it['w'], it['h']], 'mode': plan['mode'],
                             'crop': [plan['x0'], plan['y0'], plan['cw'], plan['ch']],
                             'scale': [round(plan['sx'], 4), round(plan['sy'], 4)],
                             'boxes_in_cell': dict(collections.Counter(cell_boxes))})
            usage.update([u['cat'] for u in used])
            name = '%s_synth_%03d' % (split, n)
            imwrite_u(os.path.join(outdir, name + '.jpg'), canvas)
            js = {'version': '0.4.43', 'flags': {}, 'shapes': shapes,
                  'imagePath': name + '.jpg', 'imageData': None,
                  'imageHeight': CANVAS, 'imageWidth': CANVAS, 'text': ''}
            with open(os.path.join(outdir, name + '.json'), 'w', encoding='utf-8') as fp:
                json.dump(js, fp, ensure_ascii=False)
            vis = canvas.copy()
            for s in shapes:
                (a, b), (c2, d2) = s['points']
                cv2.rectangle(vis, (int(a), int(b)), (int(c2), int(d2)), (0, 0, 255), 2)
                cv2.putText(vis, s['label'].split('_')[0][:4], (int(a), int(b) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            imwrite_u(os.path.join(VIS_DIR, name + '_vis.jpg'), vis)
            cnt = collections.Counter(s['label'] for s in shapes)
            meta[split].append({'name': name, 'path': os.path.join(outdir, name + '.jpg'),
                                'cells': used, 'total_boxes': len(shapes),
                                'per_class': {c: cnt.get(c, 0) for c in CATS}})
            print('  %s 格=%d 框=%d 分布=%s 模式=%s 源=%s' % (
                name, len(used), len(shapes), {c: cnt.get(c, 0) for c in CATS},
                [u['mode'] for u in used], [u['src'] for u in used]), flush=True)

    with open(os.path.join(TEMP, 'holdout_meta.json'), 'w', encoding='utf-8') as fp:
        json.dump(meta, fp, ensure_ascii=False, indent=1)
    print('\n格模式统计:', dict(mode_stat))
    print('meta ->', os.path.join(TEMP, 'holdout_meta.json'))

if ARGS.stage == "prelabel-synth":
    # -*- coding: utf-8 -*-
    """为 60 张多类同框拼接图建立 prelabel_synth 标注现场
    1) 只复制 raw_photos\<类>\拼接 顶层 12 张 jpg（跳过 _backup_旧版/_backup_修复前/素材 等子目录）
    2) 同名 YOLO txt -> X-AnyLabeling 0.4.43 labelme 风格 json（矩形像素坐标）
    3) 写入 classes.txt
    只读源目录：不移动/不修改/不删除 raw_photos 下任何文件
    """
    import os
    import json
    import shutil
    import hashlib
    import collections
    from PIL import Image

    _FYROOT = FY_DIR
    SRC_ROOT = os.path.join(_FYROOT, "raw_photos")
    DST = os.path.join(_FYROOT, "prelabel_synth")
    CATS = ["apple_yellow", "lemon", "mango", "pear_yellow", "pomelo"]
    SUB = "拼接"
    CLASSES = ["apple_yellow", "lemon", "mango", "pear_yellow", "pomelo"]
    VERSION = "0.4.43"
    EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    TMP = AUX


    def md5(p):
        with open(p, "rb") as fh:
            return hashlib.md5(fh.read()).hexdigest()


    def main():
        os.makedirs(DST, exist_ok=True)

        src_txt_md5 = {}          # 源 txt 指纹（复核源目录未被改动）
        copied = []
        made_json = []
        errors = []
        stat = collections.OrderedDict()

        for cat in CATS:
            sdir = os.path.join(SRC_ROOT, cat, SUB)
            if not os.path.isdir(sdir):
                errors.append((cat, "缺拼接目录"))
                continue
            imgs = sorted(f for f in os.listdir(sdir)
                          if os.path.isfile(os.path.join(sdir, f))
                          and os.path.splitext(f)[1].lower() in EXT)
            subdirs = sorted(d for d in os.listdir(sdir)
                             if os.path.isdir(os.path.join(sdir, d)))
            print("[%s] 顶层图片 %d 张 | 已排除子目录: %s" % (cat, len(imgs), subdirs))

            for f in imgs:
                stem, ext = os.path.splitext(f)
                sp = os.path.join(sdir, f)
                stxt = os.path.join(sdir, stem + ".txt")
                if not os.path.exists(stxt):
                    errors.append((f, "缺同名 txt"))
                    continue
                src_txt_md5[(cat, f)] = md5(stxt)

                dp = os.path.join(DST, f)
                shutil.copy2(sp, dp)          # 覆盖式复制，脚本可重跑不翻倍
                copied.append(f)

                raw = open(stxt, "rb").read()
                im = Image.open(dp)
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
                    if not (0 <= cid < len(CLASSES)):
                        errors.append((f, "越界 class_id: %d" % cid))
                        continue
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
                        "label": CLASSES[cid],
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
                jp = os.path.join(DST, stem + ".json")
                with open(jp, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                made_json.append(stem + ".json")

                c = stat.setdefault(cat, [0, 0])
                c[0] += 1
                c[1] += len(shapes)

        with open(os.path.join(DST, "classes.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(CLASSES) + "\n")

        total_boxes = sum(v[1] for v in stat.values())
        print("复制图片: %d 张 | 生成 json: %d 个 | classes.txt: 1" % (len(copied), len(made_json)))
        print("框总数: %d" % total_boxes)
        for k, v in stat.items():
            print("   %-14s 图=%d 框=%d" % (k, v[0], v[1]))
        print("异常: %d %s" % (len(errors), errors[:10]))

        with open(os.path.join(TMP, "synth_src_txt_md5.json"), "w", encoding="utf-8") as fh:
            json.dump({"%s/%s" % k: v for k, v in src_txt_md5.items()}, fh,
                      ensure_ascii=False, indent=1)
        print("源 txt 指纹已存: synth_src_txt_md5.json")


    if __name__ == "__main__":
        main()
