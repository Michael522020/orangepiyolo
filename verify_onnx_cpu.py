# -*- coding: utf-8 -*-
"""
本地 PC：用 onnxruntime CPU 验证 ONNX，再决定能不能做 ATC。

这是防坑步骤，不是部署脚本。
Orange Pi 上没有 onnxruntime，也不允许安装。本脚本只在训练机的 yolo 环境运行。

它检查四件事：
    1. 输入节点名和静态 shape（ATC --input_shape 直接用这里打印的名字）
    2. 输出是不是 YOLOv8 标准检测头 [1, 4+nc, 8400]，而不是内置 NMS
    3. Letterbox + /255 + xywh→xyxy + 置信度过滤 + NMS 整条后处理
    4. 在原图上画出框，确认类别顺序没有反

执行顺序：
    python train_and_export.py
    python verify_onnx_cpu.py
    看到“判定: 通过”并且弹窗里的框正确之后，才能执行 atc_convert.sh
"""

import argparse
import os
import sys

import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 必须与 data.yaml 的 names 顺序一致。
CLASS_NAMES = [
    "five",
    "four",
    "one",
    "three",
    "two",
]

CONFIG = {
    "onnx_path": os.path.join("weights", "gesture_yolov8n.onnx"),
    "image_path": "test.jpg",
    "imgsz": 640,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "save_path": "verify_result.jpg",
}


def letterbox(image, new_shape, color=(114, 114, 114)):
    """
    与 ultralytics 推理 LetterBox 对齐：
    等比缩放，短边不足的部分用 114 灰边补齐，禁止直接 resize 拉伸。
    返回 float padding (dw, dh)，后处理要用这对浮点值把框映射回原图。
    """
    height, width = image.shape[:2]
    target_h, target_w = new_shape
    gain = min(float(target_h) / height, float(target_w) / width)
    new_w = int(round(width * gain))
    new_h = int(round(height * gain))
    dw = (target_w - new_w) / 2.0
    dh = (target_h - new_h) / 2.0

    if (width, height) != (new_w, new_h):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))
    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    if image.shape[0] != target_h or image.shape[1] != target_w:
        raise RuntimeError(
            "Letterbox 结果是 {}，不是 {}x{}。请不要改 padding 计算。".format(
                image.shape[:2], target_w, target_h
            )
        )
    return image, gain, (dw, dh)


def preprocess(frame, imgsz):
    """BGR 原图 -> NCHW float32 RGB / 255。这是无 AIPP 的 ONNX 输入。"""
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    canvas, gain, pad = letterbox(frame, (imgsz, imgsz))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    # 先转 float 再除 255。uint8 直接除法会变成 0。
    blob = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    blob /= 255.0
    blob = np.expand_dims(blob, axis=0)
    return blob, gain, pad


def xywh_to_xyxy(boxes):
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def scale_boxes(xyxy, gain, pad, orig_shape):
    """从 640 画布坐标还原到原图像素。pad 是 letterbox 的 (dw, dh)。"""
    xyxy = xyxy.copy()
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy[:, :4] /= gain
    height, width = orig_shape[:2]
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)
    return xyxy


def nms_numpy(xyxy, scores, iou_thres):
    """
    verify 脚本的 NMS 与板端 cv2.dnn.NMSBoxes 使用同一套坐标。
    这里仍调用 OpenCV NMS，保证和 gesture_infer.py 是同一个实现。
    类别无关的抑制会把叠在一起的两种手势吃掉一个，所以调用前已经按类别错开了 x。
    """
    boxes = xyxy.copy()
    wh = np.empty((boxes.shape[0], 4), dtype=np.float32)
    wh[:, 0] = boxes[:, 0]
    wh[:, 1] = boxes[:, 1]
    wh[:, 2] = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    wh[:, 3] = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    indices = cv2.dnn.NMSBoxes(wh.tolist(), scores.tolist(), 0.0, float(iou_thres))
    if indices is None or len(indices) == 0:
        return []
    return np.array(indices).reshape(-1).tolist()


def decode_yolov8(raw, gain, pad, orig_shape, conf_thres, iou_thres, class_names):
    """
    YOLOv8 检测头没有 objectness。
    每个锚点是 [cx, cy, w, h, cls0, cls1, ...]，置信度就是类别分数的最大值。
    不要照搬 YOLOv5 的 obj * cls。
    """
    pred = np.asarray(raw, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise RuntimeError("无法解析的输出维度: {}".format(pred.shape))

    # 导出默认是 (4+nc, 8400)。少数图是 (8400, 4+nc)。通道数远小于锚点数。
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)

    if pred.shape[1] < 5:
        raise RuntimeError("输出通道 {} 不像检测头。".format(pred.shape[1]))

    boxes = pred[:, :4]
    cls_scores = pred[:, 4:]
    if boxes.size and float(np.max(boxes[:, 2:4])) <= 2.0:
        print("警告：宽高最大值 <= 2，坐标像是 0~1 归一化。")
        print("本脚本按 640 像素坐标解码。若框缩在左上角，说明导出头不是当前 ultralytics 像素坐标。")

    class_ids = np.argmax(cls_scores, axis=1)
    confidences = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    keep = confidences >= conf_thres
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    if boxes.shape[0] == 0:
        return []

    xyxy = xywh_to_xyxy(boxes)
    # 类间 NMS：不同类别的框在 x 方向错开一个大常数，避免互相抑制。
    xyxy_nms = xyxy.copy()
    xyxy_nms[:, [0, 2]] += class_ids.astype(np.float32)[:, None] * 7680.0
    chosen = nms_numpy(xyxy_nms, confidences, iou_thres)

    xyxy = scale_boxes(xyxy, gain, pad, orig_shape)
    results = []
    for index in chosen:
        cid = int(class_ids[index])
        name = class_names[cid] if 0 <= cid < len(class_names) else "cls_{}".format(cid)
        x1, y1, x2, y2 = xyxy[index].tolist()
        results.append(
            {
                "box": (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))),
                "conf": float(confidences[index]),
                "cls": cid,
                "name": name,
            }
        )
    return results


def _static_dim(value):
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    return None


def analyze_output_shape(shape, declared_nc):
    """判断是不是可以送去 ATC 的 YOLOv8 原始检测头。"""
    print("ONNX 实际输出 shape: {}".format(tuple(shape)))
    if len(shape) != 3:
        print("判定: 失败。YOLOv8 静态导出应为 3 维，当前是 {}。".format(tuple(shape)))
        return False
    dims = [_static_dim(shape[0]), _static_dim(shape[1]), _static_dim(shape[2])]
    if any(item is None for item in dims) or dims[0] != 1:
        print("判定: 失败。输出仍含动态维或 batch 不是 1: {}。请重新导出 dynamic=False。".format(tuple(shape)))
        return False

    dim_a, dim_b = dims[1], dims[2]
    anchors = 8400
    if anchors not in (dim_a, dim_b):
        # 融合了 NMS 的模型常见输出 (1, 300, 6)，板端后处理对不上。
        print("判定: 失败。640 输入的标准锚点数是 80*80+40*40+20*20=8400。")
        print("当前形状不像原始检测头。请重新导出，并确认 nms=False、dynamic=False、imgsz=640。")
        return False

    channels = dim_a if dim_b == anchors else dim_b
    nc = channels - 4
    layout = "通道在前 (1, 4+nc, 8400)" if dim_a == channels else "锚点在前 (1, 8400, 4+nc)，后处理会转置"
    print("布局: {}".format(layout))
    print("推断类别数 nc = {}，data.yaml / CLASS_NAMES 声明 nc = {}".format(nc, declared_nc))
    if nc != declared_nc:
        print("判定: 失败。模型类别数和 CLASS_NAMES 不一致，先改类别列表再转换。")
        return False
    if dim_a != channels:
        print("判定: 有条件通过。后处理已兼容该布局，但标准 ultralytics 导出应是 (1, {}, 8400)。".format(channels))
        return True
    print("判定: 通过。输出与 YOLOv8 标准检测头一致。")
    return True


def draw_detections(image, detections):
    palette = (
        (56, 56, 255),
        (80, 176, 0),
        (0, 168, 255),
        (184, 61, 245),
        (255, 140, 0),
        (0, 200, 180),
        (200, 80, 80),
        (120, 120, 255),
    )
    canvas = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = palette[det["cls"] % len(palette)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = "{} {:.2f}".format(det["name"], det["conf"])
        cv2.putText(canvas, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="在 PC 上用 CPU 验证 YOLOv8 ONNX")
    parser.add_argument("--onnx", default=CONFIG["onnx_path"])
    parser.add_argument("--image", default=CONFIG["image_path"])
    parser.add_argument("--conf", type=float, default=CONFIG["conf_thres"])
    parser.add_argument("--iou", type=float, default=CONFIG["iou_thres"])
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError:
        print("本机验证需要 onnxruntime，当前 Python 里没有这个模块。")
        print("它只属于 PC 的 yolo 环境。Orange Pi 部署环境不支持、也不允许安装 onnxruntime。")
        sys.exit(1)

    if not os.path.isfile(args.onnx):
        print("找不到 ONNX: {}".format(os.path.abspath(args.onnx)))
        print("请先成功执行 train_and_export.py。")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print("找不到测试图片: {}".format(os.path.abspath(args.image)))
        print("请把一张含手势的图片放到该路径，或使用 --image 指定。")
        sys.exit(1)

    # 关掉图优化，看到的输出更接近 ATC 拿到的原始计算图。
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(args.onnx, sess_options=options, providers=["CPUExecutionProvider"])

    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    print("执行提供器: {}".format(session.get_providers()))
    print("输入 name={} shape={} type={}".format(model_input.name, model_input.shape, model_input.type))
    print('把这个名字填进 ATC: --input_shape="{}:1,3,640,640"'.format(model_input.name))
    input_shape = model_input.shape
    if list(input_shape) != [1, 3, CONFIG["imgsz"], CONFIG["imgsz"]]:
        print("判定: 失败。输入不是静态 [1, 3, 640, 640]，当前是 {}。".format(input_shape))
        print("请重新导出 dynamic=False imgsz=640。在此之前不要运行 ATC。")
        sys.exit(1)

    passed = analyze_output_shape(model_output.shape, len(CLASS_NAMES))
    if not passed:
        print("ONNX 结构未通过。禁止继续 ATC。")
        sys.exit(1)

    frame = cv2.imread(args.image)
    if frame is None:
        print("OpenCV 读不到图片: {}".format(args.image))
        sys.exit(1)

    blob, gain, pad = preprocess(frame, CONFIG["imgsz"])
    print(
        "预处理: shape={} dtype={} min={:.4f} max={:.4f} mean={:.4f}".format(
            blob.shape, blob.dtype, float(blob.min()), float(blob.max()), float(blob.mean())
        )
    )
    if blob.dtype != np.float32 or blob.max() > 1.0 + 1e-3:
        print("预处理错误：无 AIPP 的 ONNX 必须是 float32 且已除以 255。")
        sys.exit(1)

    outputs = session.run(None, {model_input.name: blob})
    detections = decode_yolov8(
        outputs[0],
        gain,
        pad,
        frame.shape,
        args.conf,
        args.iou,
        CLASS_NAMES,
    )
    print("检测数量: {}".format(len(detections)))
    for det in detections:
        print("  {} conf={:.3f} box={}".format(det["name"], det["conf"], det["box"]))

    painted = draw_detections(frame, detections)
    cv2.imwrite(CONFIG["save_path"], painted)
    print("结果已写入 {}".format(os.path.abspath(CONFIG["save_path"])))
    print("结构校验已通过。请再目视确认框和类别是否正确，然后才能 ATC。")

    try:
        cv2.imshow("onnx-cpu-verify", painted)
        print("按任意键关闭窗口。")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error as exc:
        print("当前环境无法弹窗，已保留图片文件。OpenCV 信息: {}".format(exc))


if __name__ == "__main__":
    main()
