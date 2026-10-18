# -*- coding: utf-8 -*-
"""
Orange Pi AI Pro 8T 端实时手势推理。

只使用开发板已有库：
    ais_bench（InferSession）、opencv-python、numpy，以及 Python 标准库。
本文件不导入 onnxruntime、tensorrt、acl 原生接口、PIL、ultralytics。

执行顺序：
    1. PC: train_and_export.py
    2. PC: verify_onnx_cpu.py 通过
    3. 板端: bash atc_convert.sh none|aipp_rgb|aipp_yuv
    4. 板端: 按转换模式修改下面的 model_path 和 input_mode
    5. 板端: python gesture_infer.py
       按 q 或 Esc 退出

Letterbox 与 verify_onnx_cpu.py 使用同一公式（等比缩放 + 114 灰边）。
两边的数值必须保持一致，否则 CPU 验证通过、NPU 上框会偏移。
"""

import os
import sys
import time

import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# CONFIG
# input_mode 必须和 ATC 命令成对出现：
#   float32        <-> atc_convert.sh none      -> gesture_yolov8n.om
#   aipp_rgb       <-> atc_convert.sh aipp_rgb  -> gesture_yolov8n_aipp_rgb.om
#   aipp_yuv420sp  <-> atc_convert.sh aipp_yuv  -> gesture_yolov8n_aipp_yuv.om
# CLASS_NAMES 的顺序必须与 data.yaml 完全一致。
# ---------------------------------------------------------------------------
CONFIG = {
    "model_path": "gesture_yolov8n.om",
    "device_id": 0,
    "input_mode": "float32",
    "imgsz": 640,
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "camera_id": 0,
    "camera_candidates": [0, 1, 2, 3],
    "cam_width": 640,
    "cam_height": 480,
    "class_names": [
        "five",
        "four",
        "one",
        "three",
        "two",
    ],
}

PALETTE = (
    (56, 56, 255),
    (80, 176, 0),
    (0, 168, 255),
    (184, 61, 245),
    (255, 140, 0),
    (0, 200, 180),
    (200, 80, 80),
    (120, 120, 255),
)


def letterbox(image, new_shape, color=(114, 114, 114)):
    """等比缩放后用 114 填充到 640x640。返回缩放比 gain 和浮点 padding (dw, dh)。"""
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
            "Letterbox 输出 {} 不是 {}x{}，预处理与验证脚本已不一致。".format(
                image.shape[:2], target_w, target_h
            )
        )
    return image, gain, (dw, dh)


def bgr_to_nv12(image):
    """
    BGR -> NV12（YUV420SP）。仅 aipp_yuv420sp 模式使用。
    OpenCV 的 I420 是 Y + U + V，NV12 要把 U/V 交错成 UV。
    这一步在 CPU 上完成，会吃掉 YUV AIPP 的大部分收益。
    USB 摄像头长期运行请改用 aipp_rgb。
    """
    height, width = image.shape[:2]
    if height % 2 or width % 2:
        raise RuntimeError("NV12 要求偶数分辨率，当前是 {}x{}。".format(width, height))
    i420 = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420)
    y_plane = i420[:height]
    u_plane = i420[height:height + height // 4].reshape((height // 2, width // 2))
    v_plane = i420[height + height // 4:].reshape((height // 2, width // 2))
    uv_plane = np.empty((height // 2, width), dtype=np.uint8)
    uv_plane[:, 0::2] = u_plane
    uv_plane[:, 1::2] = v_plane
    return np.vstack((y_plane, uv_plane))


def normalize_frame(frame):
    if frame is None:
        raise RuntimeError("空帧")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def build_input(frame, input_mode, imgsz):
    """
    返回 (blob, gain, pad)。
    float32:      [1,3,640,640] float32，RGB，已 /255。无 AIPP 时只能走这条。
    aipp_rgb:     [1,640,640,3] uint8，保持 OpenCV 的 BGR，交给 AIPP 交换 RB 并 /255。
    aipp_yuv420sp:[1,960,640] uint8 NV12。AIPP 负责 YUV 转 RGB 并 /255。
    """
    canvas, gain, pad = letterbox(frame, (imgsz, imgsz))
    if input_mode == "float32":
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
        blob /= 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, gain, pad
    if input_mode == "aipp_rgb":
        blob = np.expand_dims(canvas, axis=0)
        return np.ascontiguousarray(blob), gain, pad
    if input_mode == "aipp_yuv420sp":
        nv12 = bgr_to_nv12(canvas)
        blob = np.expand_dims(nv12, axis=0)
        return np.ascontiguousarray(blob), gain, pad
    raise ValueError("不支持的 input_mode: {}".format(input_mode))


def session_input_info(session):
    if not hasattr(session, "get_inputs"):
        return None
    try:
        infos = session.get_inputs()
    except Exception:
        return None
    if not infos:
        return None
    return infos[0]


def _declared_type(info):
    """
    把 OM 输入类型收成 float / uint8 / unknown。
    ais_bench 有的版本给字符串，有的给 ACL 枚举：
        0 float32，1 float16，4 uint8。
    """
    raw = getattr(info, "datatype", None)
    if raw is None:
        raw = getattr(info, "dtype", None)
    if raw is None:
        return "unknown"
    if isinstance(raw, (int, np.integer)):
        if int(raw) in (0, 1, 11):
            return "float"
        if int(raw) == 4:
            return "uint8"
        return "unknown"
    text = str(raw).lower()
    if "uint8" in text or "uint_8" in text:
        return "uint8"
    if "float" in text:
        return "float"
    return "unknown"


def _same_layout(src_shape, dst_shape):
    """只允许补 1 维或压平。NCHW 和 NHWC 元素数相同，但内存顺序不同，不能 reshape。"""
    src = tuple(int(dim) for dim in src_shape)
    dst = tuple(int(dim) for dim in dst_shape)
    if src == dst:
        return True
    src_tight = tuple(dim for dim in src if dim != 1)
    dst_tight = tuple(dim for dim in dst if dim != 1)
    if src_tight == dst_tight:
        return True
    if len(dst) == 1 and dst[0] == int(np.prod(src)):
        return True
    if len(dst) == 2 and dst[0] == 1 and dst[1] == int(np.prod(src)):
        return True
    return False


def align_to_model(blob, info, input_mode):
    """dtype 必须和有没有 AIPP 一致。布局只在不改变像素顺序时才按 OM shape 重整。"""
    if info is None:
        return np.ascontiguousarray(blob)

    declared = _declared_type(info)
    if input_mode == "float32" and declared == "uint8":
        raise RuntimeError("OM 输入是 uint8，说明转换时启用了 AIPP。请把 input_mode 改为 aipp_rgb 或 aipp_yuv420sp。")
    if input_mode != "float32" and declared == "float":
        raise RuntimeError("OM 输入仍是 float，说明这个 om 没有插入 AIPP。请把 input_mode 改回 float32。")
    if input_mode != "float32" and blob.dtype != np.uint8:
        raise RuntimeError("AIPP 模式只接受 uint8，当前是 {}。不要做 /255 或 float32 转换。".format(blob.dtype))

    shape = getattr(info, "shape", None)
    if not shape:
        return np.ascontiguousarray(blob)
    shape = tuple(int(dim) for dim in shape)
    if not shape or any(dim <= 0 for dim in shape):
        return np.ascontiguousarray(blob)
    if int(np.prod(shape)) != blob.size:
        raise RuntimeError(
            "输入元素数 {} 与 OM 形状 {} 不一致。input_mode={} 和 ATC 使用的 AIPP 配置不是同一套。".format(
                blob.size, shape, input_mode
            )
        )
    if not _same_layout(blob.shape, shape):
        print("警告：OM 声明形状 {}，脚本按 {} 送数。".format(shape, tuple(blob.shape)))
        print("二者只是维度顺序不同，保持当前内存布局，避免把 NHWC 错看成 NCHW。")
        return np.ascontiguousarray(blob)
    return np.ascontiguousarray(blob.reshape(shape))


def xywh_to_xyxy(boxes):
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return xyxy


def scale_boxes(xyxy, gain, pad, orig_shape):
    xyxy = xyxy.copy()
    xyxy[:, [0, 2]] -= pad[0]
    xyxy[:, [1, 3]] -= pad[1]
    xyxy[:, :4] /= gain
    height, width = orig_shape[:2]
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)
    return xyxy


def class_aware_nms(xyxy, scores, class_ids, iou_thres, conf_thres):
    """
    cv2.dnn.NMSBoxes 不区分类别。
    按类别把框平移到互不重叠的区域，效果与 YOLOv8 的 class-aware NMS 一致。
    传给 NMS 的矩形格式是 [x, y, w, h]。
    """
    wh = np.empty((xyxy.shape[0], 4), dtype=np.float32)
    wh[:, 0] = xyxy[:, 0] + class_ids.astype(np.float32) * 7680.0
    wh[:, 1] = xyxy[:, 1]
    wh[:, 2] = np.maximum(0.0, xyxy[:, 2] - xyxy[:, 0])
    wh[:, 3] = np.maximum(0.0, xyxy[:, 3] - xyxy[:, 1])
    valid = (wh[:, 2] >= 1.0) & (wh[:, 3] >= 1.0)
    if not np.any(valid):
        return []
    original_index = np.where(valid)[0]
    indices = cv2.dnn.NMSBoxes(
        wh[valid].tolist(),
        scores[valid].tolist(),
        float(conf_thres),
        float(iou_thres),
    )
    if indices is None or len(indices) == 0:
        return []
    picked = np.array(indices).reshape(-1)
    return original_index[picked].tolist()


def decode_yolov8(raw, gain, pad, orig_shape, conf_thres, iou_thres, class_names):
    """YOLOv8：无 objectness，置信度 = max(类别分数)。坐标是 640 画布上的像素 xywh。"""
    pred = np.asarray(raw, dtype=np.float32)
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        raise RuntimeError("NPU 输出维度异常: {}".format(getattr(raw, "shape", None)))
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)
    if pred.shape[1] < 5:
        raise RuntimeError("NPU 输出通道数为 {}，不是 YOLOv8 检测头。".format(pred.shape[1]))

    boxes = pred[:, :4]
    cls_scores = pred[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confidences = cls_scores[np.arange(cls_scores.shape[0]), class_ids]
    keep = confidences >= conf_thres
    boxes = boxes[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]
    if boxes.shape[0] == 0:
        return []

    xyxy = xywh_to_xyxy(boxes)
    chosen = class_aware_nms(xyxy, confidences, class_ids, iou_thres, conf_thres)
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


def run_infer(session, blob):
    """ais-bench 0.0.2 的 infer 默认就是静态 shape。旧签名没有 mode 时退回单参数调用。"""
    try:
        return session.infer([blob], mode="static")
    except TypeError:
        return session.infer([blob])


def first_array(outputs):
    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            raise RuntimeError("InferSession 返回了空输出。")
        return outputs[0]
    return outputs


def draw(frame, detections, fps, infer_ms):
    canvas = frame
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        color = PALETTE[det["cls"] % len(PALETTE)]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = "{} {:.2f}".format(det["name"], det["conf"])
        cv2.putText(
            canvas,
            label,
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "FPS {:.1f}  infer {:.1f} ms".format(fps, infer_ms),
        (8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def open_camera(preferred_id, candidates, width, height):
    """优先打开指定设备；失败后依次尝试候选号。每个号都先读一帧，避免“能打开但不能出图”。"""
    ordered = []
    for cam_id in [preferred_id] + list(candidates):
        if cam_id not in ordered:
            ordered.append(cam_id)

    errors = []
    for cam_id in ordered:
        capture = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(cam_id)
        if not capture.isOpened():
            errors.append("{} 打开失败".format(cam_id))
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = capture.read()
        if not ok or frame is None:
            errors.append("{} 打开后读不到帧".format(cam_id))
            capture.release()
            continue
        print("摄像头已打开: id={} 分辨率={}x{}".format(cam_id, frame.shape[1], frame.shape[0]))
        return capture
    print("摄像头打开失败，已尝试设备号: {}".format(ordered))
    for item in errors:
        print("  {}".format(item))
    print("请确认 USB 摄像头已插入。可用命令查看设备（只查看，不修改系统）: ls /dev/video*")
    return None


def load_session(model_path, device_id):
    absolute = os.path.abspath(model_path)
    if not os.path.isfile(absolute):
        print("模型加载失败：找不到 OM 文件 {}".format(absolute))
        print("请先在本机完成 ATC，并让 CONFIG['model_path'] 指向对应的 om。")
        return None
    try:
        from ais_bench.infer.interface import InferSession
    except ImportError:
        print("模型加载失败：当前解释器没有 ais_bench。")
        print("OM 推理只能在 Orange Pi 已安装的 ais-bench 0.0.2 上运行。")
        return None
    try:
        session = InferSession(device_id, absolute)
    except Exception as exc:
        print("模型加载失败：InferSession 无法加载 {}".format(absolute))
        print("请确认当前 shell 已经带好板上的 CANN 环境，且 om 的 soc_version 是 Ascend310B4。")
        print("input_mode 也必须与转换该 om 时是否插入 AIPP 一致。")
        print("原始错误: {}".format(exc))
        return None
    print("OM 已加载: {}".format(absolute))
    info = session_input_info(session)
    if info is not None:
        print(
            "OM 输入 name={} shape={} datatype={}".format(
                getattr(info, "name", "?"),
                getattr(info, "shape", "?"),
                getattr(info, "datatype", getattr(info, "dtype", "?")),
            )
        )
    return session


def main():
    mode = CONFIG["input_mode"]
    if mode not in ("float32", "aipp_rgb", "aipp_yuv420sp"):
        print("CONFIG['input_mode'] 只能是 float32、aipp_rgb 或 aipp_yuv420sp。")
        sys.exit(1)
    print("推理模式: {}".format(mode))
    print("类别: {}".format(", ".join(CONFIG["class_names"])))
    if mode != "float32":
        print("AIPP 已启用：送入 NPU 的数据保持 uint8，脚本不会做 /255，也不会转 float32。")

    session = load_session(CONFIG["model_path"], CONFIG["device_id"])
    if session is None:
        sys.exit(1)
    info = session_input_info(session)

    capture = open_camera(
        CONFIG["camera_id"],
        CONFIG["camera_candidates"],
        CONFIG["cam_width"],
        CONFIG["cam_height"],
    )
    if capture is None:
        session.free_resource()
        sys.exit(1)

    window = "gesture-npu"
    fps = 0.0
    warmed = False
    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok or frame is None:
                print("读帧失败，继续尝试。")
                time.sleep(0.05)
                continue
            frame = normalize_frame(frame)
            blob, gain, pad = build_input(frame, mode, CONFIG["imgsz"])
            blob = align_to_model(blob, info, mode)

            infer_start = time.perf_counter()
            outputs = run_infer(session, blob)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0
            detections = decode_yolov8(
                first_array(outputs),
                gain,
                pad,
                frame.shape,
                CONFIG["conf_thres"],
                CONFIG["iou_thres"],
                CONFIG["class_names"],
            )

            elapsed = time.perf_counter() - loop_start
            if elapsed > 0:
                instant = 1.0 / elapsed
                # 首帧包含 NPU 上下文开销，不计入平滑 FPS。
                if warmed:
                    fps = fps * 0.9 + instant * 0.1 if fps > 0 else instant
                else:
                    warmed = True
            draw(frame, detections, fps, infer_ms)
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if hasattr(session, "free_resource"):
            session.free_resource()


if __name__ == "__main__":
    main()
