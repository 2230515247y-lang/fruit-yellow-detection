# -*- coding: utf-8 -*-
"""fruit_yellow 五类检测训练入口（合并 train_yolov8s.py + train_yolo11s.py）
- 权重: 默认 D:\software\yolo\weights\yolov8s.pt (COCO 预训练, nc=80 -> 自动适配 nc=5)
- 数据: 默认 D:\software\yolo\datasets\fruit_yellow\dataset\data.yaml (train325/val49/test48)
- 输出: 默认 D:\software\yolo\runs\<name>

合并说明：
- 两版脚本的增广超参一致（hsv_h=0.5 / hsv_s=0.9 / hsv_v=0.4 强抖动压制颜色捷径、erasing=0.3、mosaic=1.0），保留为默认值
- train_yolov8s.py 的 imgsz=640 与 train_yolo11s.py 的 imgsz=960 统一为 --imgsz（默认 640，跑 yolo11s 建议 960）
- 保留 v8 版的工作目录切换与 WEIGHTS_DIR 打印；保留 11s 版的“输出目录已存在即告警”（可用 --exist-ok 覆盖）

用法示例：
  python scripts/train.py
  python scripts/train.py --model D:\software\yolo\weights\yolo11s.pt --imgsz 960
  python scripts/train.py --model yolo11s.yaml --epochs 300 --batch 16 --name fruit_yellow_v2
"""
import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow 五类水果检测训练入口")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo），同时作为工作目录")
    p.add_argument("--model", default=os.path.join(DEFAULT_ROOT, "weights", "yolov8s.pt"),
                   help="预训练权重或模型 yaml（默认 <root>/weights/yolov8s.pt）")
    p.add_argument("--data", default=None, help="data.yaml 路径（默认 <root>/datasets/fruit_yellow/dataset/data.yaml）")
    p.add_argument("--imgsz", type=int, default=640, help="训练分辨率（默认 640；yolo11s 历史配置为 960）")
    p.add_argument("--epochs", type=int, default=150, help="训练轮数（默认 150）")
    p.add_argument("--batch", type=int, default=-1, help="批大小（默认 -1=AutoBatch，12GB 显存自适应）")
    p.add_argument("--device", default=0, help="训练设备（默认 0）")
    p.add_argument("--workers", type=int, default=4, help="DataLoader 进程数（默认 4）")
    p.add_argument("--seed", type=int, default=0, help="随机种子（默认 0）")
    p.add_argument("--project", default=None, help="输出根目录（默认 <root>/runs）")
    p.add_argument("--name", default="fruit_yellow_v1", help="实验名（默认 fruit_yellow_v1）")
    p.add_argument("--exist-ok", action="store_true", help="允许复用已存在的输出目录（默认不允许，避免误覆盖历史结果）")
    return p


def main():
    a, _extra = _build_parser().parse_known_args()
    root = a.root
    os.chdir(root)
    sys.path.insert(0, root)

    from ultralytics import YOLO
    from ultralytics.utils import WEIGHTS_DIR

    print(f"[env] WEIGHTS_DIR = {WEIGHTS_DIR}", flush=True)
    print(f"[env] cwd = {os.getcwd()}", flush=True)

    data = a.data or os.path.join(root, "datasets", "fruit_yellow", "dataset", "data.yaml")
    project = a.project or os.path.join(root, "runs")
    run_dir = os.path.join(project, a.name)

    print("[launcher] pid=%d model=%s data=%s" % (os.getpid(), a.model, data), flush=True)
    assert Path(a.model).is_file() or a.model.endswith(".yaml"), f"权重不存在: {a.model}"
    assert Path(data).is_file(), f"data.yaml 不存在: {data}"
    if os.path.exists(run_dir) and not a.exist_ok:
        print("[launcher] FATAL: 输出目录已存在 %s （如需复用请加 --exist-ok）" % run_dir, flush=True)
        sys.exit(2)

    t0 = time.time()
    model = YOLO(a.model)
    results = model.train(
        data=data,
        imgsz=a.imgsz,
        epochs=a.epochs,
        device=a.device,
        hsv_h=0.5, hsv_s=0.9, hsv_v=0.4,   # 强抖动：压制颜色捷径
        erasing=0.3,
        mosaic=1.0,
        pretrained=True,
        seed=a.seed,
        deterministic=True,
        workers=a.workers,
        batch=a.batch,       # AutoBatch，12GB 显存自适应
        cache=False,
        project=project,
        name=a.name,
        exist_ok=a.exist_ok,
        plots=True,
        verbose=True,
    )
    print("[done] finished in %.1f s" % (time.time() - t0), flush=True)
    print(f"[done] best = {results.save_dir}", flush=True)


if __name__ == "__main__":
    main()
