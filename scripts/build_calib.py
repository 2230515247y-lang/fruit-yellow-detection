# -*- coding: utf-8 -*-
"""fruit_yellow int8 校准集构建（build_calib.py）

合并来源（核心算法逻辑未改动，保留原中文注释）：
  - build_calib_and_stage.py：抽取 int8 量化校准图集（X5 / Hailo 两套）+ 落盘双版本 ONNX + 生成 calib_manifest.json
仅把脚本内写死的绝对路径改为命令行参数，并把输出目录默认值由会话 workspace 改为 D:/software/yolo/output/fruit_yellow_v1_deploy。
注意：源 ONNX 与 export_monkey_patch_official.py 需位于 --aux 目录；缺文件时会打印 [SKIP] 并继续。

用法：
  python scripts/build_calib.py
  python scripts/build_calib.py --aux D:\software\yolo\aux --out D:\software\yolo\output\fruit_yellow_v1_deploy
"""
import argparse
import os
import sys

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow int8 量化校准集抽取与部署产物落盘")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo）")
    p.add_argument("--dataset-dir", default=None, help="fruit_yellow 数据根（默认 <root>/datasets/fruit_yellow）")
    p.add_argument("--aux", default=None,
                   help="中间产物目录：需含 std_export/ 与 x5_export/ 两版 ONNX（默认 <root>/aux）")
    p.add_argument("--out", default=None, help="部署产物输出目录（默认 <root>/output/fruit_yellow_v1_deploy）")
    return p

ARGS, _EXTRA = _build_parser().parse_known_args()


ROOT = ARGS.root
FY_DIR = ARGS.dataset_dir or os.path.join(ROOT, "datasets", "fruit_yellow")
DS_DIR = os.path.join(FY_DIR, "dataset")
AUX = ARGS.aux or os.path.join(ROOT, "aux")
if not ARGS.out:
    ARGS.out = os.path.join(ROOT, "output", "fruit_yellow_v1_deploy")

# 抽取 int8 量化校准图集（X5 / Hailo 两套）+ 落盘双版本 ONNX + 生成清单
import os, json, shutil, hashlib
import cv2
import numpy as np

TMP = AUX
OUT = ARGS.out
DS = DS_DIR
NAMES = ["apple_yellow", "lemon", "mango", "pear_yellow", "pomelo"]

os.makedirs(os.path.join(OUT, "onnx"), exist_ok=True)
CX5 = os.path.join(OUT, "calib_x5")
CHL = os.path.join(OUT, "calib_hailo")
os.makedirs(CX5, exist_ok=True)
os.makedirs(CHL, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def list_imgs(split):
    d = os.path.join(DS, "images", split)
    return sorted([os.path.join(d, f) for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMG_EXT])


def labels_of(src):
    """读取同名 txt 标签，返回 (类别id集合, 框数)"""
    split = "test" if "\\test\\" in src else "val"
    lbl = os.path.join(DS, "labels", split, os.path.splitext(os.path.basename(src))[0] + ".txt")
    ids, n = set(), 0
    if os.path.exists(lbl):
        with open(lbl, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 5:
                    ids.add(int(float(parts[0])))
                    n += 1
    return ids, n


def convert_to_jpg(src, dst_dir):
    stem = os.path.splitext(os.path.basename(src))[0]
    dst = os.path.join(dst_dir, stem + ".jpg")
    if os.path.splitext(src)[1].lower() in (".jpg", ".jpeg"):
        shutil.copy2(src, dst)          # jpg 直接复制，避免二次压缩
        return dst, "copied"
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("read fail: " + src)
    cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return dst, "converted_from_" + os.path.splitext(src)[1].lstrip(".").lower()


test_imgs = list_imgs("test")
val_imgs = list_imgs("val")

test_single = [p for p in test_imgs if "_synth_" not in os.path.basename(p)]
test_mosaic = [p for p in test_imgs if "_synth_" in os.path.basename(p)]
val_single = [p for p in val_imgs if "_synth_" not in os.path.basename(p)]
val_mosaic = [p for p in val_imgs if "_synth_" in os.path.basename(p)]

# --- X5 校准集：test 单图每类 6 张 + test 全部拼接图 8 张 ---
x5_sel = []
for i, cls in enumerate(NAMES):
    picked = [p for p in test_single if os.path.basename(p).startswith(cls + "__")]
    x5_sel += picked[:6]
x5_sel += test_mosaic

# --- Hailo 校准集：test 全部 + val 全部 ---
hl_sel = test_imgs + val_imgs

manifest = {"names": NAMES, "sets": {}}
for tag, sel, dst in (("calib_x5", x5_sel, CX5), ("calib_hailo", hl_sel, CHL)):
    entries = []
    for src in sel:
        jpg, mode = convert_to_jpg(src, dst)
        ids, nb = labels_of(src)
        entries.append({
            "calib_file": os.path.basename(jpg),
            "calib_path": jpg,
            "source_path": src,
            "split": "test" if "\\test\\" in src else "val",
            "kind": "mosaic(同框拼接)" if "_synth_" in os.path.basename(src) else "single",
            "mode": mode,
            "classes": sorted(NAMES[i] for i in ids),
            "boxes": nb,
        })
    cls_stat = {}
    for e in entries:
        for c in e["classes"]:
            cls_stat[c] = cls_stat.get(c, 0) + 1
    manifest["sets"][tag] = {
        "dir": dst,
        "count": len(entries),
        "single": sum(1 for e in entries if e["kind"] == "single"),
        "mosaic": sum(1 for e in entries if e["kind"] == "mosaic(同框拼接)"),
        "class_coverage_images": cls_stat,
        "mosaic_class_combos": sorted({" + ".join(e["classes"]) for e in entries if "mosaic" in e["kind"]}),
        "entries": entries,
    }

# --- 落盘两版 ONNX + 官方脚本 ---
for src, name in ((os.path.join(TMP, "x5_export", "fruit_yellow_v1_x5.onnx"), "fruit_yellow_v1_x5.onnx"),
                  (os.path.join(TMP, "std_export", "fruit_yellow_v1_standard.onnx"), "fruit_yellow_v1_standard.onnx")):
    if not os.path.exists(src):
        print("[SKIP] ONNX 不存在，跳过:", src)
        continue
    dst = os.path.join(OUT, "onnx", name)
    shutil.copy2(src, dst)
    manifest.setdefault("onnx", {})[name] = {"path": dst, "size_bytes": os.path.getsize(dst), "sha256": sha256(dst)}

if os.path.exists(os.path.join(TMP, "export_monkey_patch_official.py")):
    shutil.copy2(os.path.join(TMP, "export_monkey_patch_official.py"), os.path.join(OUT, "export_monkey_patch.py"))
    manifest["export_monkey_patch"] = {
        "path": os.path.join(OUT, "export_monkey_patch.py"),
        "sha256": sha256(os.path.join(OUT, "export_monkey_patch.py")),
        "source": "https://github.com/D-Robotics/rdk_model_zoo (branch rdk_x5, samples/vision/ultralytics_yolo/conversion)",
    }
else:
    print("[SKIP] export_monkey_patch_official.py 不存在:", os.path.join(TMP, "export_monkey_patch_official.py"))

with open(os.path.join(OUT, "calib_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

for tag in ("calib_x5", "calib_hailo"):
    s = manifest["sets"][tag]
    print(f"{tag}: {s['count']} 张 (单图 {s['single']} / 拼接 {s['mosaic']})  类别覆盖={s['class_coverage_images']}")
    print("   拼接同框类别组合:", s["mosaic_class_combos"])
print("ONNX:", json.dumps({k: {"size": v["size_bytes"], "sha256": v["sha256"][:16]} for k, v in manifest.get("onnx", {}).items()}, ensure_ascii=False))
print("OUT_DIR:", OUT)
