# -*- coding: utf-8 -*-
"""fruit_yellow ONNX 导出与校验（export_onnx.py）

合并来源（核心算法逻辑未改动，保留原中文注释）：
  - export_standard_onnx.py -> stage export：导出 ultralytics 原生标准 ONNX（供 Hailo DFC / Dataflow Compiler 使用）
  - verify_onnx.py          -> stage verify：对比 standard / x5 两版 ONNX 的输入输出张量结构，并做数值/功能等价性抽查
仅把脚本内写死的绝对路径改为命令行参数（默认指向 D:/software/yolo）。
注意：verify stage 的 x5 版 ONNX、monkey patch 脚本需位于 --aux 目录（x5_export/ 与 export_monkey_patch_official.py）；缺文件时会打印 [SKIP] 并继续。

用法：
  python scripts/export_onnx.py --stage export
  python scripts/export_onnx.py --stage verify --img D:\software\yolo\datasets\fruit_yellow\dataset\images\test\test_synth_001.jpg
"""
import argparse
import os
import sys

DEFAULT_ROOT = r"D:\software\yolo"


def _build_parser():
    p = argparse.ArgumentParser(description="fruit_yellow ONNX 导出与结构/数值校验")
    p.add_argument("--stage", default="export", choices=["export", "verify"],
                   help="export=导出 ultralytics 原生标准 ONNX(供 Hailo DFC 等使用)；verify=校验 ONNX 张量结构与数值/功能等价性")
    p.add_argument("--root", default=DEFAULT_ROOT, help="项目根目录（默认 D:/software/yolo）")
    p.add_argument("--dataset-dir", default=None, help="fruit_yellow 数据根（默认 <root>/datasets/fruit_yellow）")
    p.add_argument("--aux", default=None, help="中间产物目录（std_export 等，默认 <root>/aux）")
    p.add_argument("--pt", default=None, help="待导出/校验的 PyTorch 权重（默认 <root>/runs/fruit_yellow_v1/weights/best.pt）")
    p.add_argument("--imgsz", type=int, default=640, help="导出分辨率（默认 640）")
    p.add_argument("--opset", type=int, default=11, help="ONNX opset（默认 11）")
    p.add_argument("--img", default=None, help="verify stage 数值抽查所用图片（默认 <dataset-dir>/dataset/images/test/test_synth_001.jpg）")
    return p

ARGS, _EXTRA = _build_parser().parse_known_args()


ROOT = ARGS.root
FY_DIR = ARGS.dataset_dir or os.path.join(ROOT, "datasets", "fruit_yellow")
DS_DIR = os.path.join(FY_DIR, "dataset")
AUX = ARGS.aux or os.path.join(ROOT, "aux")
os.makedirs(AUX, exist_ok=True)
if not ARGS.pt:
    ARGS.pt = os.path.join(ROOT, "runs", "fruit_yellow_v1", "weights", "best.pt")
if not ARGS.img:
    ARGS.img = os.path.join(DS_DIR, "images", "test", "test_synth_001.jpg")

if ARGS.stage == "export":
    # 导出 ultralytics 原生标准 ONNX（供 Hailo DFC / Dataflow Compiler 使用）
    import os, shutil, time, json
    from ultralytics import YOLO

    PT = ARGS.pt
    TMP = AUX
    STD = os.path.join(TMP, "std_export")
    os.makedirs(STD, exist_ok=True)

    pt = os.path.join(STD, "fruit_yellow_v1_standard.pt")
    if not os.path.exists(pt):
        shutil.copy2(PT, pt)

    t0 = time.time()
    m = YOLO(pt)
    print("TASK:", m.task)
    print("NAMES:", m.names)
    print("IMGSZ_TRAIN_ARG:", getattr(m.model, "args", {}).get("imgsz") if hasattr(m.model, "args") else None)
    out = m.export(format="onnx", imgsz=ARGS.imgsz, opset=ARGS.opset, simplify=True,
                   dynamic=False, batch=1, half=False, nms=False, device="cpu")
    print("STD_ONNX:", out)
    print("ELAPSED: %.1fs" % (time.time() - t0))
    print("SIZE:", os.path.getsize(out))

if ARGS.stage == "verify":
    # 校验两版 ONNX 的输入输出张量结构，并做数值/功能等价性抽查
    import os, json, hashlib
    import numpy as np
    import cv2
    import onnx
    import onnxruntime as ort

    TMP = AUX
    STD = os.path.join(TMP, "std_export", "fruit_yellow_v1_standard.onnx")
    X5 = os.path.join(TMP, "x5_export", "fruit_yellow_v1_x5.onnx")
    PT_X5 = os.path.join(TMP, "x5_export", "fruit_yellow_v1_x5.pt")
    IMG = ARGS.img

    report = {}


    def sha256(p, n=1 << 20):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while True:
                b = f.read(n)
                if not b:
                    break
                h.update(b)
        return h.hexdigest().upper()


    def shape_of(vi):
        t = vi.type.tensor_type
        dims = []
        for d in t.shape.dim:
            if d.dim_value:
                dims.append(int(d.dim_value))
            elif d.dim_param:
                dims.append(d.dim_param)
            else:
                dims.append("?")
        return dims


    def letterbox(img, size=640, color=114):
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), color, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, (left, top)


    img_bgr = cv2.imread(IMG)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lb_rgb, ratio, pad = letterbox(img_rgb, 640)
    inp = (lb_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1,3,640,640
    print("INPUT_IMG:", IMG, img_bgr.shape, "->", inp.shape, "dtype", inp.dtype)

    for tag, path in (("standard", STD), ("x5", X5)):
        if not os.path.exists(path):
            print("[SKIP] ONNX 文件不存在，跳过:", path)
            continue
        info = {"path": path, "size_bytes": os.path.getsize(path), "sha256": sha256(path)}
        m = onnx.load(path)
        g = m.graph
        info["ir_version"] = m.ir_version
        info["producer"] = f"{m.producer_name} {m.producer_version}".strip()
        info["opset"] = [{"domain": o.domain or "ai.onnx", "version": o.version} for o in m.opset_import]
        info["inputs"] = [{"name": i.name, "dtype": onnx.TensorProto.DataType.Name(i.type.tensor_type.elem_type),
                           "shape": shape_of(i)} for i in g.input]
        info["outputs"] = [{"name": o.name, "dtype": onnx.TensorProto.DataType.Name(o.type.tensor_type.elem_type),
                            "shape": shape_of(o)} for o in g.output]
        ops = {}
        for n in g.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        info["node_count"] = len(g.node)
        info["has_nms"] = ops.get("NonMaxSuppression", 0)
        info["op_top"] = sorted(ops.items(), key=lambda kv: -kv[1])[:8]

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ort_in = sess.get_inputs()[0]
        info["ort_input"] = {"name": ort_in.name, "shape": ort_in.shape, "type": ort_in.type}
        outs = sess.run(None, {ort_in.name: inp})
        info["ort_outputs"] = [{"idx": i, "shape": list(o.shape), "dtype": str(o.dtype),
                               "min": float(np.min(o)), "max": float(np.max(o))} for i, o in enumerate(outs)]
        info["onnx_names_match_runtime"] = [o.name for o in sess.get_outputs()]
        report[tag] = info

        print(f"\n===== {tag} =====")
        print("ir", info["ir_version"], "| opset", info["opset"], "| producer", info["producer"])
        print("inputs :", json.dumps(info["inputs"], ensure_ascii=False))
        print("outputs:", json.dumps(info["outputs"], ensure_ascii=False))
        print("nodes", info["node_count"], "| NMS nodes:", info["has_nms"], "| top ops:", info["op_top"])
        print("runtime outputs:", json.dumps(info["ort_outputs"], ensure_ascii=False))

    # ---------- 功能/数值等价性抽查 ----------
    # 1) 标准 ONNX：与 pt 在同一张图上的检测框比对
    try:
        from ultralytics import YOLO
        pt = YOLO(ARGS.pt)
        onx = YOLO(STD)
        r_pt = pt.predict(IMG, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
        r_on = onx.predict(IMG, imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
        b_pt = r_pt.boxes
        b_on = r_on.boxes
        cmp = {
            "pt_boxes": int(len(b_pt)), "onnx_boxes": int(len(b_on)),
            "pt_classes": sorted(set(int(c) for c in b_pt.cls.tolist())),
            "onnx_classes": sorted(set(int(c) for c in b_on.cls.tolist())),
            "pt_conf_top": float(b_pt.conf.max()) if len(b_pt) else None,
            "onnx_conf_top": float(b_on.conf.max()) if len(b_on) else None,
        }
        if len(b_pt) == len(b_on) and len(b_pt) > 0:
            cmp["max_xyxy_absdiff"] = float(np.abs(b_pt.xyxy.numpy() - b_on.xyxy.numpy()).max())
            cmp["max_conf_absdiff"] = float(np.abs(b_pt.conf.numpy() - b_on.conf.numpy()).max())
        report["standard"]["functional_check"] = cmp
        print("\n[std functional]", json.dumps(cmp, ensure_ascii=False))
    except Exception as e:
        report["standard"]["functional_check"] = {"error": repr(e)}
        print("\n[std functional] FAILED:", repr(e))

    # 2) X5 ONNX：与打过 monkey patch 的 torch 前向输出逐张量比对
    try:
        import importlib.util, torch
        from ultralytics import YOLO
        spec = importlib.util.spec_from_file_location("emp", os.path.join(TMP, "export_monkey_patch_official.py"))
        emp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emp)
        y = YOLO(PT_X5)
        model = y.model.float().eval()
        emp.modelZooOptimizer(model.model)
        with torch.no_grad():
            t_out = model(torch.from_numpy(inp))
        if isinstance(t_out, (list, tuple)):
            t_list = [t.numpy() for t in t_out]
        else:
            t_list = [t_out.numpy()]
        sess = ort.InferenceSession(X5, providers=["CPUExecutionProvider"])
        o_list = sess.run(None, {sess.get_inputs()[0].name: inp})
        diffs = []
        for i, (t, o) in enumerate(zip(t_list, o_list)):
            t = np.asarray(t).reshape(o.shape)
            diffs.append({"idx": i, "shape": list(o.shape),
                          "max_absdiff": float(np.abs(t - o).max()),
                          "mean_absdiff": float(np.abs(t - o).mean())})
        report["x5"]["torch_equiv"] = {"torch_out_count": len(t_list), "onnx_out_count": len(o_list), "diffs": diffs}
        print("\n[x5 torch-equiv]", json.dumps(report["x5"]["torch_equiv"], ensure_ascii=False))
    except Exception as e:
        report["x5"]["torch_equiv"] = {"error": repr(e)}
        print("\n[x5 torch-equiv] FAILED:", repr(e))

    with open(os.path.join(TMP, "onnx_verify_raw.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nSAVED:", os.path.join(TMP, "onnx_verify_raw.json"))
