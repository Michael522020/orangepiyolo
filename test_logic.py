# -*- coding: utf-8 -*-
"""
PC 上可运行的逻辑测试。不训练、不加载 OM、不打开摄像头。

覆盖：
    Letterbox 尺寸和灰边
    verify_onnx_cpu 与 gesture_infer 的预处理数值一致
    AIPP 的 uint8 / NV12 形状
    YOLOv8 解码、低分过滤、同类 NMS、异类保留
    输出 shape 门禁
    缺失 OM 时的失败路径

运行：
    python test_logic.py
"""

import io
import os
import sys
from contextlib import redirect_stdout

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gesture_infer as gi
import verify_onnx_cpu as vo


FAILURES = []


def check(name, condition, detail):
    if condition:
        print("PASS  {}".format(name))
        return
    FAILURES.append(name)
    print("FAIL  {}  {}".format(name, detail))


def make_output(channels, anchors, assignments):
    """assignments: (anchor, cx, cy, w, h, class_id, score)。"""
    pred = np.zeros((1, channels, anchors), dtype=np.float32)
    for anchor, cx, cy, width, height, class_id, score in assignments:
        pred[0, 0, anchor] = cx
        pred[0, 1, anchor] = cy
        pred[0, 2, anchor] = width
        pred[0, 3, anchor] = height
        pred[0, 4 + class_id, anchor] = score
    return pred


def test_letterbox_geometry():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[360, 640] = (1, 2, 3)
    canvas, gain, pad = vo.letterbox(image, (640, 640))
    check("letterbox 尺寸", canvas.shape == (640, 640, 3), canvas.shape)
    check("letterbox 缩放比", abs(gain - 0.5) < 1e-6, gain)
    check("letterbox 水平无灰边", abs(pad[0]) < 1e-6, pad)
    check("letterbox 垂直灰边", abs(pad[1] - 140.0) < 1e-6, pad)
    check("灰边颜色", int(canvas[0, 0, 0]) == 114 and int(canvas[-1, -1, 0]) == 114, canvas[0, 0].tolist())
    check("内容没有被拉伸到顶边", int(canvas[0, 320, 0]) == 114, int(canvas[0, 320, 0]))


def test_two_implementations_match():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    left, gain_l, pad_l = vo.letterbox(frame, (640, 640))
    right, gain_r, pad_r = gi.letterbox(frame, (640, 640))
    check("两边 Letterbox 像素一致", np.array_equal(left, right), "max diff {}".format(np.abs(left.astype(int) - right.astype(int)).max() if left.shape == right.shape else "shape"))
    check("两边 gain/pad 一致", gain_l == gain_r and pad_l == pad_r, (gain_l, gain_r, pad_l, pad_r))

    blob_v, _, _ = vo.preprocess(frame, 640)
    blob_g, _, _ = gi.build_input(frame, "float32", 640)
    diff = float(np.max(np.abs(blob_v - blob_g)))
    check("float32 预处理一致", blob_v.shape == blob_g.shape and diff == 0.0, "shape {} {} diff {}".format(blob_v.shape, blob_g.shape, diff))
    check("float32 值域", blob_v.dtype == np.float32 and 0.0 <= float(blob_v.min()) and float(blob_v.max()) <= 1.0, (blob_v.dtype, float(blob_v.min()), float(blob_v.max())))


def test_color_order_and_aipp_inputs():
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)  # B, G, R
    blob, _, _ = vo.preprocess(frame, 640)
    check("RGB 通道顺序", np.allclose(blob[0, 0], 30 / 255.0) and np.allclose(blob[0, 2], 10 / 255.0), (float(blob[0, 0, 0, 0]), float(blob[0, 2, 0, 0])))

    rgb_blob, _, _ = gi.build_input(frame, "aipp_rgb", 640)
    check("AIPP RGB 是 uint8 NHWC", rgb_blob.shape == (1, 640, 640, 3) and rgb_blob.dtype == np.uint8, (rgb_blob.shape, rgb_blob.dtype))
    check("AIPP RGB 保持 BGR 原值", rgb_blob[0, 0, 0].tolist() == [10, 20, 30], rgb_blob[0, 0, 0].tolist())
    check("AIPP RGB 没有被归一化", int(rgb_blob.max()) == 30, int(rgb_blob.max()))

    yuv_blob, _, _ = gi.build_input(frame, "aipp_yuv420sp", 640)
    check("AIPP NV12 形状", yuv_blob.shape == (1, 960, 640) and yuv_blob.dtype == np.uint8, (yuv_blob.shape, yuv_blob.dtype))
    check("NV12 字节数", yuv_blob.size == 640 * 640 * 3 // 2, yuv_blob.size)

    nv12 = yuv_blob[0]
    restored = cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)
    error = float(np.mean(np.abs(restored.astype(np.int16) - frame.astype(np.int16))))
    check("NV12 可还原回 BGR", error < 3.0, error)


def test_decode_and_nms():
    names = vo.CLASS_NAMES
    # 原图 640x480。letterbox 后上下各 80 像素灰边，gain = 1。
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, gain, pad = vo.letterbox(frame, (640, 640))
    pred = make_output(
        12,
        8400,
        [
            (0, 160, 180, 120, 120, 4, 0.92),  # ok，还原后 (100,40)-(220,160)
            (1, 160, 180, 120, 120, 4, 0.55),  # 同一个 ok，应被 NMS 去掉
            (2, 400, 400, 50, 40, 0, 0.80),    # 另一类，与高分框不重叠
            (3, 300, 300, 40, 40, 0, 0.10),    # 低于阈值
            (4, 160, 180, 120, 120, 1, 0.88),  # fist，与 ok 重叠但类别不同，要保留
        ],
    )
    found_v = vo.decode_yolov8(pred, gain, pad, frame.shape, 0.25, 0.45, names)
    found_g = gi.decode_yolov8(pred, gain, pad, frame.shape, 0.25, 0.45, names)
    names_v = sorted((item["name"], item["box"]) for item in found_v)
    names_g = sorted((item["name"], item["box"]) for item in found_g)
    check("两边解码结果一致", names_v == names_g, (names_v, names_g))
    by_name = {item["name"]: item for item in found_v}
    expected = {names[4], names[1], names[0]}
    check("保留三个不同类别", set(by_name) == expected, sorted(by_name))
    check("高分框回到原图", by_name.get(names[4], {}).get("box") == (100, 40, 220, 160), by_name.get(names[4]))
    check("同类重复框被抑制", sum(item["name"] == names[4] for item in found_v) == 1, found_v)
    check("低分框被丢弃", all(item["conf"] >= 0.25 for item in found_v), found_v)

    transposed = np.transpose(pred, (0, 2, 1))
    found_t = gi.decode_yolov8(transposed, gain, pad, frame.shape, 0.25, 0.45, names)
    check("锚点在前的输出也能解码", sorted(item["name"] for item in found_t) == sorted(expected), found_t)


def test_shape_gate_and_session_guard():
    with redirect_stdout(io.StringIO()):
        ok = vo.analyze_output_shape((1, 12, 8400), 8)
        transposed = vo.analyze_output_shape((1, 8400, 12), 8)
        end2end = vo.analyze_output_shape((1, 300, 6), 8)
        dynamic = vo.analyze_output_shape((1, "candidates", 8400), 8)
        wrong_nc = vo.analyze_output_shape((1, 10, 8400), 8)
    check("标准输出通过", ok is True, ok)
    check("转置输出可解析", transposed is True, transposed)
    check("内置 NMS 输出被拒绝", end2end is False, end2end)
    check("动态输出被拒绝", dynamic is False, dynamic)
    check("类别数不一致被拒绝", wrong_nc is False, wrong_nc)

    class Info(object):
        def __init__(self, shape, datatype):
            self.shape = shape
            self.datatype = datatype

    float_blob = np.zeros((1, 3, 8, 8), dtype=np.float32)
    uint_blob = np.zeros((1, 8, 8, 3), dtype=np.uint8)
    raised = False
    try:
        gi.align_to_model(float_blob, Info((1, 3, 8, 8), 4), "float32")
    except RuntimeError:
        raised = True
    check("float32 模式遇到 uint8 OM 会失败", raised, raised)

    raised = False
    try:
        gi.align_to_model(uint_blob, Info((1, 8, 8, 3), "float32"), "aipp_rgb")
    except RuntimeError:
        raised = True
    check("AIPP 模式遇到 float OM 会失败", raised, raised)

    fitted = gi.align_to_model(uint_blob, Info((8, 8, 3), "uint8"), "aipp_rgb")
    check("只差 batch 维时可以重整", fitted.shape == (8, 8, 3), fitted.shape)

    with redirect_stdout(io.StringIO()):
        kept = gi.align_to_model(uint_blob, Info((1, 3, 8, 8), "uint8"), "aipp_rgb")
    check("NHWC 不会被 reshape 成 NCHW", kept.shape == (1, 8, 8, 3), kept.shape)

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        missing = gi.load_session("missing_model.om", 0)
    check("缺失 OM 返回失败", missing is None, missing)
    check("缺失 OM 有明确提示", "找不到 OM 文件" in buffer.getvalue(), buffer.getvalue())


def test_labels_match_yaml():
    names = []
    with open(os.path.join(os.path.dirname(__file__), "data.yaml"), "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if ":" in text and text.split(":", 1)[0].strip().isdigit():
                names.append(text.split(":", 1)[1].strip())
    check("类别列表一致", names == vo.CLASS_NAMES == gi.CONFIG["class_names"], (names, vo.CLASS_NAMES, gi.CONFIG["class_names"]))


def main():
    test_letterbox_geometry()
    test_two_implementations_match()
    test_color_order_and_aipp_inputs()
    test_decode_and_nms()
    test_shape_gate_and_session_guard()
    test_labels_match_yaml()
    print("-" * 40)
    if FAILURES:
        print("失败 {} 项: {}".format(len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
