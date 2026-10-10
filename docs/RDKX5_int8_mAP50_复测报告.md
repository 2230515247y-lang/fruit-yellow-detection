---
AIGC:
    Label: "1"
    ContentProducer: 001191440300708461136T1XGW3
    ProduceID: 2e55e772cb5d0bb106e5d06c937dff97_6b78e11bc2c911f1a05452540064ee0f
    ReservedCode1: 7dvj/WwgcUiVn30De4iiOPvLTw8rNtjGocjRNUqFI+nDTdQgcUWy47hZyo9qvuOMumxQcgJCVfnI9+QB4O/Gv9/7XASkmwqWQHou1JkesKUM1/RkqDTnJCvxjtPNyLPn7kPtPBEid0jep8laaGNdoPEq81UON2tTbwemief0cdSHp2KWq92GwFQQZ1c=
    ContentPropagator: 001191440300708461136T1XGW3
    PropagateID: 2e55e772cb5d0bb106e5d06c937dff97_6b78e11bc2c911f1a05452540064ee0f
    ReservedCode2: 7dvj/WwgcUiVn30De4iiOPvLTw8rNtjGocjRNUqFI+nDTdQgcUWy47hZyo9qvuOMumxQcgJCVfnI9+QB4O/Gv9/7XASkmwqWQHou1JkesKUM1/RkqDTnJCvxjtPNyLPn7kPtPBEid0jep8laaGNdoPEq81UON2tTbwemief0cdSHp2KWq92GwFQQZ1c=
---

# RDK X5 板端 int8 量化 mAP50 精度复测报告

- 日期：2026-10-08
- 模型：fruit_yellow_v1（YOLOv8s，5 类：apple_yellow / lemon / mango / pear_yellow / pomelo）
- 板端模型：`~/yolo/demos/vision/ultralytics_yolo/model/fruit_yellow_v1_x5_bayese_640x640_nv12.bin`（11.8 MB，int8 / bayes-e / NV12）
- 数据：test split 48 张（44 jpg + 2 png + 2 webp），GT 共 243 个框
- 放行线：相对 FP32 基线掉点 ≤ 2 点

---

## 1. 板端推理结果

| 指标 | 数值 |
|---|---|
| 图像数 | 48 / 48（ok=48, failed=0） |
| 预测框总数 | 574 |
| 延迟 mean | 31.1 ms |
| 延迟 p50 | 29.9 ms |
| 延迟 p90 | 35.9 ms |
| 延迟 max | 43.2 ms |
| 吞吐（近似） | ≈ 32 FPS（含前后处理） |

推理参数：`conf=0.001`、`NMS IoU=0.7`、letterbox 640×640、max_det 未截断。
日志：板端 `~/yolo/eval/batch_log.txt`，预测 txt：板端 `~/yolo/eval/preds`，本机归档 `output\preds_board`。

> **修正记录（2026-10-08 复核）**：首轮仅推理 46 张，原因是 `batch_infer.py` 的扩展名白名单（`EXTS`）不含 `.webp`，2 张 webp 被静默跳过。
> 现已完成两项收尾：① 脚本改为「遍历目录全部文件 + `cv2.imread` 解码失败则跳过」，不再依赖扩展名白名单；② 清理板端 `~/yolo/eval/test` 的 2 张转码 PNG 副本，目录恢复为 48 个文件。
> 用修正后的脚本在 48 张原生图（含 webp 原件）上整组重跑复核：**images=48 / ok=48 / failed=0，boxes=574，mAP50=0.7930**，与上表完全一致——板端 OpenCV 本就可直接解码 webp，无需任何转码副本。

---

## 2. 精度结果

### 2.1 三种口径对照

| 口径 | 模型 / 运行位置 | 预处理 | 评估实现 | mAP50 |
|---|---|---|---|---|
| A 官方 val 口径 | best.pt / 本机 RTX 5070 FP32 | rect 矩形推理 | ultralytics 内置 Metric | **0.8200** |
| B 部署口径（同口径基线） | best.pt / 本机 RTX 5070 FP32 | letterbox 640 | pycocotools COCOeval | 0.7828 |
| C 部署口径（待验收） | int8 bin / RDK X5 BPU | letterbox 640 | pycocotools COCOeval | **0.7930** |

口径 A 用于复现历史基线 0.820（本次复核结果 0.8200，完全一致），但它使用 ultralytics val 默认的 **rect 矩形推理**，与板端实际部署的 **letterbox 640×640 正方形**预处理不同，因此不能直接与板端数值相减。

**唯一干净的对比是 B 与 C**：两者数据、预处理、置信度阈值（0.001）、NMS IoU（0.7）、导出格式、评估脚本全部一致，差异只来自「FP32 GPU 推理」与「int8 BPU 推理」。

### 2.2 同口径逐类对比（B vs C）

| 类别 | FP32 本机（B） | int8 板端（C） | Δ（C−B） |
|---|---|---|---|
| apple_yellow | 0.6667 | 0.6752 | +0.0085 |
| lemon | 0.7477 | 0.7451 | −0.0026 |
| mango | 0.8375 | 0.8806 | +0.0431 |
| pear_yellow | 0.9574 | 0.9556 | −0.0018 |
| pomelo | 0.7047 | 0.7086 | +0.0039 |
| **mAP50** | **0.7828** | **0.7930** | **+0.0102** |

### 2.3 预测规模变化

| 指标 | FP32 本机 | int8 板端 |
|---|---|---|
| 预测框总数 | 690 | 574（−16.8%） |
| apple_yellow | 126 | 103 |
| lemon | 51 | 48 |
| mango | 177 | 132 |
| pear_yellow | 99 | 86 |
| pomelo | 237 | 205 |

框数整体下降约 17%，说明量化后低分候选框的置信度被轻微压低；但按置信度排序后的 PR 曲线几乎不变，AP 未受损。

---

## 3. 结论

**通过放行。** 同口径下 int8 板端 mAP50 = 0.7930，FP32 本机 = 0.7828，掉点 = **−1.02 点（即无掉点）**，远优于「≤ 2 点」的放行线。逐类 AP50 波动在 ±0.05 以内，最大回退仅 lemon −0.0026、pear_yellow −0.0018。

需要说明：+1.02 点的「负掉点」不应理解为 int8 提升了精度，8 张合成图 + 40 张实拍图共 48 张的测试集规模偏小，1 个点左右的差异属于统计波动范围。可靠结论是：**未观察到 int8 量化的精度损失**。

若后续出现赛题精度争议，降压优先级建议：`--quantized int16`（部分层）→ `per_channel` → 用现场实拍图重做校准集，而非调整该模型的评估口径。

---

## 4. 备注与遗留项

1. ~~**建议修改 `batch_infer.py`**：将扩展名白名单改为「遍历目录全部文件 + `cv2.imread` 失败则跳过」，或直接把 `EXT` 扩到包含 `.webp`，避免今后评测集混入 webp 时被静默漏掉。~~ → **已完成（2026-10-08）**：脚本已改为遍历目录全部文件、解码失败仅计数跳过（日志打印 `SKIP`）；板端 `runtime/python/batch_infer.py` 已同步更新，本机副本见 `output\batch_infer.py`。
2. ~~**板端数据目录已污染**：`~/yolo/eval/test` 目前含 50 个文件（48 原图 + 2 张转码 PNG 副本 + 2 张原始 webp）。建议只保留 48 张标准图，删除 2 张 PNG 副本，避免下次推理重复计数。~~ → **已完成（2026-10-08）**：已删除 `apple_yellow__apple_yellow_057.png`、`lemon__lemon_069.png` 两张转码副本（原始 webp 保留），目录恢复为 48 个文件。
3. **两套口径建议固定其一**：报告归档与赛题说明统一使用「部署口径（letterbox 640）」，因为它与板端实际推理链路一致；`ultralytics val` 默认的 rect 口径仅用于训练期监控。
4. **测试集规模**：48 张、243 框，逐类 AP 的置信区间较宽（尤其 lemon 仅 51 框），如需更强结论应扩充 test split。
*（内容由AI生成，仅供参考）*
