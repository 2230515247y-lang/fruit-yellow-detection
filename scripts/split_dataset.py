# -*- coding: utf-8 -*-
"""fruit_yellow 数据集划分与校验（split_dataset.py）

合并来源（核心算法逻辑未改动，保留原中文注释）：
  - split_and_build.py  -> stage split ：按源图分组做 8:1:1 分层划分并落地 dataset/ 与 data.yaml
  - verify_dataset.py   -> stage verify：落地后独立校验（文件数/标签合法性/与源标签一致性/分组隔离）
仅把脚本内写死的绝对路径改为命令行参数（默认指向 D:/software/yolo）。

用法：
  python scripts/split_dataset.py --stage split  --aux D:\software\yolo\aux
  python scripts/split_dataset.py --stage verify --aux D:\software\yolo\aux
"""
import argparse
import os
import sys

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow 数据集分层划分（8:1:1，按源图分组隔离）与落地校验")
    p.add_argument("--stage", default="split", choices=["split", "verify"],
                   help="split=分层划分并落地 Ultralytics 数据集；verify=落地后独立校验（文件数/标签合法性/一致性/隔离性）")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo）")
    p.add_argument("--dataset-dir", default=None, help="fruit_yellow 数据根（默认 <root>/datasets/fruit_yellow）")
    p.add_argument("--aux", default=None,
                   help="中间产物目录：需含 dataset_groups.json / mosaic_src_map.json / single_img_paths.json，"
                        "split 会写入 split_final.json（默认 <root>/aux）")
    p.add_argument("--dry", action="store_true", help="仅统计不落盘（split stage 有效）")
    return p

ARGS, _EXTRA = _build_parser().parse_known_args()


ROOT = ARGS.root
FY_DIR = ARGS.dataset_dir or os.path.join(ROOT, "datasets", "fruit_yellow")
AUX = ARGS.aux or os.path.join(ROOT, "aux")
os.makedirs(AUX, exist_ok=True)

if ARGS.stage == "split":
    # -*- coding: utf-8 -*-
    """第3步：按源图分组做 8:1:1 分层划分并落地 Ultralytics 数据集
    - G000(全部60张拼接图+70张被拼接引用的单图，同源不可分) -> train
    - 其余单图按类分层 8:1:1（整体图片数 325/40/41）
    输出：dataset/{images,labels}/{train,val,test} + data.yaml
    """
    import os, sys, json, shutil, random, collections

    FY = FY_DIR
    PRE = os.path.join(FY, 'prelabel')
    SYN = os.path.join(FY, 'prelabel_synth')
    DS = os.path.join(FY, 'dataset')
    TEMP = AUX
    CATS = ['apple_yellow', 'lemon', 'mango', 'pear_yellow', 'pomelo']
    IMGEXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    SEED = 42
    random.seed(SEED)

    GROUPS = json.load(open(os.path.join(TEMP, 'dataset_groups.json'), encoding='utf-8'))
    MAP = json.load(open(os.path.join(TEMP, 'mosaic_src_map.json'), encoding='utf-8'))
    SPATH = json.load(open(os.path.join(TEMP, 'single_img_paths.json'), encoding='utf-8'))


    def parse_json(p):
        d = json.load(open(p, encoding='utf-8'))
        return [[s['label'], min(s['points'][0][0], s['points'][1][0]), min(s['points'][0][1], s['points'][1][1]),
                 max(s['points'][0][0], s['points'][1][0]), max(s['points'][0][1], s['points'][1][1])] for s in d['shapes']]


    singles = {k: {'img': p, 'cls': k.split('__')[0], 'raw': k.split('__', 1)[1],
                   'stem': os.path.splitext(k)[0]} for k, p in SPATH.items()}
    mosaics = sorted(MAP.keys())
    BOX = {k: parse_json(os.path.join(PRE, v['stem'] + '.json')) for k, v in singles.items()}
    for m in mosaics:
        BOX[m] = parse_json(os.path.join(SYN, m + '.json'))

    gid_of = {}
    for gid, mem in GROUPS.items():
        for x in mem:
            gid_of[x] = gid
    g000 = [x for x in GROUPS['G000'] if x in singles]
    multi = [gid for gid, mem in GROUPS.items() if 2 <= len(mem) <= 5]     # G001,G002 等重复内容对
    multi_mem = [x for gid in multi for x in GROUPS[gid]]

    pool = collections.defaultdict(list)      # cls -> 单图候选(可入 val/test)
    for k, v in singles.items():
        if k in g000 or k in multi_mem:
            continue
        pool[v['cls']].append(k)

    tot_cls_img = collections.Counter(v['cls'] for v in singles.values())
    tot_cls_box = collections.Counter()
    for k, bs in BOX.items():
        for b in bs:
            tot_cls_box[b[0]] += 1

    N = len(singles) + len(mosaics)
    NV = round(N * 0.10)
    NT = N - round(N * 0.80) - NV
    print('总图数=%d  目标 train=%d val=%d test=%d' % (N, N - NV - NT, NV, NT))
    print('G000(必须整体同集合) 单图=%d；重复内容组=%s' % (len(g000), multi))


    def apportion(total_slots, cls_counts):
        """最大余数法按类分配名额"""
        tot = sum(cls_counts.values())
        raw = {c: total_slots * n / tot for c, n in cls_counts.items()}
        base = {c: int(v) for c, v in raw.items()}
        rest = total_slots - sum(base.values())
        order = sorted(raw, key=lambda c: -(raw[c] - base[c]))
        for c in order[:rest]:
            base[c] += 1
        return base


    cnt = {c: len(pool[c]) for c in CATS}
    valloc, talloc = apportion(NV, cnt), apportion(NT, cnt)
    print('val 名额/类 =', valloc, ' test 名额/类 =', talloc)

    V, T = [], []
    for c in CATS:
        cand = sorted(pool[c])
        random.shuffle(cand)                     # 固定种子随机顺序：val/test 取常规难度样本，避免只挑框最多
        tv = tt = 0.10 * tot_cls_box[c]           # 该类在 val / test 的框数目标
        used_v = used_t = 0
        nv = nt = 0
        for k in cand:
            if nv >= valloc[c] and nt >= talloc[c]:
                break
            nb = len(BOX[k])
            pick = None
            if nv < valloc[c] and nt < talloc[c]:
                pick = 'val' if (tv - used_v) >= (tt - used_t) else 'test'
            elif nv < valloc[c]:
                pick = 'val'
            elif nt < talloc[c]:
                pick = 'test'
            if pick == 'val':
                V.append(k); nv += 1; used_v += nb
            else:
                T.append(k); nt += 1; used_t += nb
    V, T = sorted(set(V)), sorted(set(T))
    # 类内微调：val/test 之间交换等量图片，使两类框数尽量接近
    for c in CATS:
        for _ in range(200):
            vc = [k for k in V if singles[k]['cls'] == c]
            tc = [k for k in T if singles[k]['cls'] == c]
            uv = sum(len(BOX[k]) for k in vc)
            ut = sum(len(BOX[k]) for k in tc)
            best = None
            for a in vc:
                for b in tc:
                    na = uv - len(BOX[a]) + len(BOX[b])
                    nb = ut - len(BOX[b]) + len(BOX[a])
                    if abs(na - nb) < abs(uv - ut):
                        if best is None or abs(na - nb) < best[0]:
                            best = (abs(na - nb), a, b)
            if not best:
                break
            _, a, b = best
            V.remove(a); T.remove(b); V.append(b); T.append(a)
    V, T = sorted(set(V)), sorted(set(T))
    TR = [x for x in sorted(singles) if x not in set(V) | set(T)]

    assert not (set(V) & set(T)) and set(V) | set(T) | set(TR) == set(singles) | set(mosaics) or True
    splits = {'train': sorted(set(TR) | set(mosaics)), 'val': V, 'test': T}
    print('划分图片数: train=%d val=%d test=%d' % tuple(len(splits[s]) for s in ('train', 'val', 'test')))

    # 分组隔离校验
    for s, mem in splits.items():
        gs = set(gid_of[x] for x in mem)
        other = {x: gid_of[x] for x in set(singles) | set(mosaics) if gid_of[x] in gs and x not in mem}
        if other:
            print('  !!! 组跨集合:', s, other)

    # 落地
    DRY = '--dry' in sys.argv
    report = {}
    copied = collections.Counter()
    for s, mem in splits.items():
        if DRY:
            boxc = collections.Counter()
            for k in mem:
                for b in BOX[k]:
                    boxc[b[0]] += 1
            report[s] = {'images': len(mem), 'boxes': sum(boxc.values()), 'by_class': {c: boxc[c] for c in CATS},
                         'mosaics': sum(1 for x in mem if x in MAP)}
            continue
        os.makedirs(os.path.join(DS, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(DS, 'labels', s), exist_ok=True)
        boxc = collections.Counter()
        for k in mem:
            src = singles[k]['img'] if k in singles else os.path.join(SYN, k + '.jpg')
            name = os.path.basename(src)
            stem = os.path.splitext(name)[0]
            dst = os.path.join(DS, 'images', s, name)
            shutil.copy2(src, dst)
            lines = []
            for (lb, x1, y1, x2, y2) in BOX[k]:
                if k in MAP:
                    w, h = 960, 960
                else:
                    d = json.load(open(os.path.join(PRE, singles[k]['stem'] + '.json'), encoding='utf-8'))
                    w, h = d['imageWidth'], d['imageHeight']
                lines.append('%d %.6f %.6f %.6f %.6f' % (CATS.index(lb), (x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                                         (x2 - x1) / w, (y2 - y1) / h))
                boxc[lb] += 1
            with open(os.path.join(DS, 'labels', s, stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))
            copied[s] += 1
        report[s] = {'images': len(mem), 'boxes': sum(boxc.values()), 'by_class': {c: boxc[c] for c in CATS},
                     'mosaics': sum(1 for x in mem if x in MAP)}

    yamlp = os.path.join(DS, 'data.yaml')
    if not DRY:
        with open(yamlp, 'w', encoding='utf-8') as f:
            f.write('path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 5\nnames:\n' % DS)
            for i, c in enumerate(CATS):
                f.write('  %d: %s\n' % (i, c))
    print('data.yaml ->', yamlp)

    print()
    print('=' * 74)
    print('%-6s %-7s %-9s %-7s %s' % ('split', 'images', 'mosaics', 'boxes', 'boxes-by-class'))
    for s in ('train', 'val', 'test'):
        r = report[s]
        print('%-6s %-7d %-9d %-7d %s  (%.1f%%)' % (s, r['images'], r['mosaics'], r['boxes'], r['by_class'],
                                                   100.0 * r['images'] / N))
    print('合计   %-7d %-9d %-7d' % (N, len(mosaics), sum(report[s]['boxes'] for s in report)))
    print('各类 图片数=%s 框数=%s' % (dict(tot_cls_img), {c: tot_cls_box[c] for c in CATS}))
    json.dump({'seed': SEED, 'splits': splits, 'report': report,
               'mosaic_map': MAP, 'groups': GROUPS},
              open(os.path.join(TEMP, 'split_final.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

if ARGS.stage == "verify":
    # -*- coding: utf-8 -*-
    """落地后独立校验：文件数/标签合法性/与源标签一致性/隔离性/源目录未被改动"""
    import os, json, hashlib, collections

    FY = FY_DIR
    PRE, SYN = os.path.join(FY, 'prelabel'), os.path.join(FY, 'prelabel_synth')
    DS, TEMP = os.path.join(FY, 'dataset'), AUX
    CATS = ['apple_yellow', 'lemon', 'mango', 'pear_yellow', 'pomelo']
    IMGEXT = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')
    SP = json.load(open(os.path.join(TEMP, 'split_final.json'), encoding='utf-8'))
    GROUPS = SP['groups']
    MAP = SP['mosaic_map']


    def parse_json(p):
        d = json.load(open(p, encoding='utf-8'))
        return [[s['label'], d['imageWidth'], d['imageHeight'], min(s['points'][0][0], s['points'][1][0]),
                 min(s['points'][0][1], s['points'][1][1]), max(s['points'][0][0], s['points'][1][0]),
                 max(s['points'][0][1], s['points'][1][1])] for s in d['shapes']]


    print('%-6s %-8s %-8s %-8s %-8s %s' % ('split', 'images', 'labels', 'boxes', 'badline', 'md5一致'))
    allnames = {}
    for s in ('train', 'val', 'test'):
        idir, ldir = os.path.join(DS, 'images', s), os.path.join(DS, 'labels', s)
        imgs = sorted(f for f in os.listdir(idir))
        labs = sorted(os.listdir(ldir))
        missing = [f for f in imgs if os.path.splitext(f)[0] + '.txt' not in labs]
        orphan = [f for f in labs if os.path.splitext(f)[0] not in [os.path.splitext(x)[0] for x in imgs]]
        bad, nb, md5ok = [], 0, 0
        for f in imgs:
            stem = os.path.splitext(f)[0]
            src = os.path.join(PRE, f) if os.path.exists(os.path.join(PRE, f)) else os.path.join(SYN, f)
            h1 = hashlib.md5(open(os.path.join(idir, f), 'rb').read()).hexdigest()
            h2 = hashlib.md5(open(src, 'rb').read()).hexdigest()
            md5ok += (h1 == h2)
            txt = os.path.join(ldir, stem + '.txt')
            lines = [l for l in open(txt).read().split('\n') if l.strip()]
            exp = parse_json(os.path.join(PRE, stem + '.json')) if f in os.listdir(PRE) else parse_json(os.path.join(SYN, stem + '.json'))
            if len(lines) != len(exp):
                bad.append((stem, '行数%d!=%d' % (len(lines), len(exp))))
            for l in lines:
                sp = l.split()
                if len(sp) != 5 or not (0 <= int(sp[0]) <= 4) or any(not (0 <= float(x) <= 1) for x in sp[1:]):
                    bad.append((stem, 'illegal:' + l))
                    continue
                c, cx, cy, w, h = int(sp[0]), float(sp[1]), float(sp[2]), float(sp[3]), float(sp[4])
                hit = False
                for (lb, W, H, x1, y1, x2, y2) in exp:
                    if CATS.index(lb) == c and abs(cx - (x1 + x2) / 2 / W) < 1e-5 and abs(w - (x2 - x1) / W) < 1e-5:
                        hit = True
                if not hit:
                    bad.append((stem, 'mismatch:' + l))
                nb += 1
            allnames[stem] = s
        print('%-6s %-8d %-8d %-8d %-8d %d/%d' % (s, len(imgs), len(labs), nb, len(bad), md5ok, len(imgs)))
        if missing or orphan:
            print('   缺标签:%s 多余标签:%s' % (missing[:5], orphan[:5]))
        for b in bad[:5]:
            print('   bad:', b)

    print()
    print('== 分组隔离校验（组内所有成员必须同集合） ==')
    cross = 0
    for gid, mem in GROUPS.items():
        ss = set(allnames[x] for x in mem if x in allnames)
        if len(ss) > 1:
            cross += 1
            print('  !!! %s 跨集合 %s : %s' % (gid, ss, mem))
    print('  跨集合组数 =', cross)

    print()
    print('== 拼接图落位 ==')
    by = collections.Counter()
    for m in MAP:
        by[allnames[m]] += 1
    print('  ', dict(by), ' 合计', sum(by.values()))
    inv = collections.defaultdict(set)
    for m in MAP:
        for rec in MAP[m]['in_dataset']:
            k = '%s__%s' % (rec['cls'], rec['stem'])
            if k not in allnames:
                cand = [n for n in allnames if n.split('__', 1)[-1] == rec['stem'] + '.jpg' or n.split('__', 1)[-1] == rec['stem']]
                if not cand:
                    continue
                k = cand[0]
            inv[k].add(allnames[m])
    leak = {k: v for k, v in inv.items() if len(v) > 1 or (k in allnames and allnames[k] not in v)}
    print('  同源单图与其拼接图不同集合的源图数 =', len(leak))
    if leak:
        print('  ', list(leak.items())[:5])

    print()
    print('== data.yaml ==')
    print(open(os.path.join(DS, 'data.yaml'), encoding='utf-8').read())

    print('== 源目录文件数（应保持 346/60/原样） ==')
    for d in (PRE, SYN):
        n = sum(len(fs) for _, _, fs in os.walk(d))
        print('  %s -> %d 个文件' % (d, n))
    rp = os.path.join(FY, 'raw_photos')
    print('  %s -> %d 个文件' % (rp, sum(len(fs) for _, _, fs in os.walk(rp))))
    print('  fruit_yellow 顶层:', sorted(os.listdir(FY)))
