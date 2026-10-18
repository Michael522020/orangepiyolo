# -*- coding: utf-8 -*-
"""
本地 PC：YOLOv8n 迁移学习 + ONNX 导出。

运行环境（仅训练机）：
    conda activate yolo
    该环境需要已安装 ultralytics、torch、onnx、onnxsim/onnxslim。
    本脚本禁止拿到 Orange Pi 上运行。部署板依赖只读，不能在这里 pip/conda 安装任何东西。

执行顺序：
    1. 按 data.yaml 放好数据集
    2. 在本目录执行: python train_and_export.py
    3. 确认生成 weights/gesture_yolov8n.onnx
    4. 接着运行 verify_onnx_cpu.py，通过后再做 ATC
"""

import os
import shutil
import sys

# 无论从哪个工作目录启动，都切到脚本所在目录，保证 data.yaml 相对路径稳定。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# CONFIG：训练和导出的全部硬编码参数
# ---------------------------------------------------------------------------
CONFIG = {
    "model": "yolov8n.pt",
    "data": "data.yaml",
    # 这台训练机没有 NVIDIA 显卡，按 CPU 能跑完的规模设置。
    "epochs": 40,
    "imgsz": 640,
    "batch": 8,
    "workers": 0,
    "patience": 30,
    "project": "runs/gesture",
    "name": "yolov8n_hand5",
    "export_dir": "weights",
    "export_name": "gesture_yolov8n.onnx",
}


def _load_ultralytics():
    """训练机依赖。缺失时只提示，不在脚本里安装。"""
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print("当前不是可用的 PC 训练环境，缺少 ultralytics 或 torch。")
        print("请在 conda 环境 yolo 中运行本脚本。不要在 Orange Pi 上安装或运行。")
        print("原始错误: {}".format(exc))
        sys.exit(1)
    return torch, YOLO


def _check_data_yaml(path):
    if not os.path.isfile(path):
        print("找不到数据集配置: {}".format(os.path.abspath(path)))
        sys.exit(1)
    try:
        import yaml
    except ImportError:
        print("训练环境缺少 PyYAML，无法读取 data.yaml。")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    names = data.get("names")
    nc = int(data.get("nc", -1))
    if isinstance(names, dict):
        ordered = [names[k] for k in sorted(names, key=lambda x: int(x))]
    elif isinstance(names, list):
        ordered = list(names)
    else:
        print("data.yaml 的 names 格式无法识别。")
        sys.exit(1)
    if nc != len(ordered):
        print("data.yaml 中 nc={} 与 names 数量 {} 不一致。".format(nc, len(ordered)))
        sys.exit(1)
    if not (6 <= nc <= 10):
        print("警告：当前类别数是 {}，嵌入式端建议控制在 6 到 10 类。".format(nc))
    print("数据集类别 ({}): {}".format(nc, ", ".join(ordered)))
    return ordered


def _resolve_device(torch):
    if torch.cuda.is_available():
        print("检测到 CUDA，使用 GPU 0 训练。")
        return 0
    print("未检测到 CUDA，使用 CPU 训练。CPU 会很慢，只建议用来打通流程。")
    return "cpu"


def _inspect_onnx(onnx_path):
    """用训练环境自带的 onnx 包核对静态形状，避免把动态图送进 ATC。"""
    try:
        import onnx
    except ImportError:
        print("训练环境没有 onnx 包，跳过导出后检查。")
        return
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("-" * 60)
    print("ONNX 检查通过: {}".format(os.path.abspath(onnx_path)))
    for item in model.graph.input:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
        print("  输入节点: name={} shape={}".format(item.name, dims))
        if any(not isinstance(d, int) or d <= 0 for d in dims):
            print("  错误：输入仍是动态维度。ATC 需要静态 shape，请重新导出 dynamic=False。")
            sys.exit(1)
        print('  ATC 参数应写: --input_shape="{}:1,3,640,640"'.format(item.name))
    for item in model.graph.output:
        dims = []
        for dim in item.type.tensor_type.shape.dim:
            dims.append(dim.dim_value if dim.dim_value else dim.dim_param)
        print("  输出节点: name={} shape={}".format(item.name, dims))
    print("-" * 60)
    print("下一步：python verify_onnx_cpu.py")
    print("verify 通过之前，不要执行 atc。")


def main():
    torch, YOLO = _load_ultralytics()
    class_names = _check_data_yaml(CONFIG["data"])
    device = _resolve_device(torch)

    if not os.path.isfile(CONFIG["model"]):
        print("当前目录没有 {}，ultralytics 将尝试从官方地址下载预训练权重。".format(CONFIG["model"]))

    model = YOLO(CONFIG["model"])
    print("开始迁移学习。类别顺序必须与部署脚本 CLASS_NAMES 一致：")
    print(class_names)

    model.train(
        data=CONFIG["data"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=device,
        workers=CONFIG["workers"],
        patience=CONFIG["patience"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        # 关闭 mosaic 的最后几轮由 ultralytics 默认 close_mosaic 处理。
        # 输入尺寸锁死 640，和后续 ONNX / OM 静态 shape 对齐。
        rect=False,
        cache=False,
    )

    # ultralytics 8.4 会把保存目录放到 runs/detect 下面，不能写死相对路径。
    trainer = getattr(model, "trainer", None)
    best_pt = str(getattr(trainer, "best", "") or "")
    if not best_pt or not os.path.isfile(best_pt):
        print("训练结束但找不到 best.pt: {}".format(best_pt or CONFIG["project"]))
        sys.exit(1)

    print("加载最优权值并导出 ONNX: {}".format(best_pt))
    exporter = YOLO(best_pt)
    # 导出参数是 ATC 的硬约束，不要改：
    #   opset=11      Ascend ATC 对 YOLOv8 最稳的算子集
    #   dynamic=False 禁止动态 batch / 动态分辨率，310B 走静态 OM
    #   simplify=True 去掉 ATC 不支持的冗余节点
    #   nms=False     NMS 留在 Python 里做。若导出进网络，输出不再是 (1,4+nc,8400)
    try:
        exported = exporter.export(
            format="onnx",
            imgsz=640,
            opset=11,
            simplify=True,
            dynamic=False,
            nms=False,
        )
    except TypeError:
        # 极老的 ultralytics 没有 nms 参数，但默认就是不融合 NMS。
        exported = exporter.export(
            format="onnx",
            imgsz=640,
            opset=11,
            simplify=True,
            dynamic=False,
        )
    except Exception as exc:
        print("ONNX 导出失败。simplify=True 需要训练环境里已有 onnxsim 或 onnxslim。")
        print("不要到 Orange Pi 上安装这些包。原始错误: {}".format(exc))
        sys.exit(1)

    os.makedirs(CONFIG["export_dir"], exist_ok=True)
    target = os.path.join(CONFIG["export_dir"], CONFIG["export_name"])
    shutil.copy2(str(exported), target)
    print("已复制到固定路径: {}".format(os.path.abspath(target)))
    _inspect_onnx(target)


if __name__ == "__main__":
    main()
