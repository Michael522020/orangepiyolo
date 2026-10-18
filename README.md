# 五类手势检测（YOLOv8n → Orange Pi AI Pro 8T）

PC 上训练 YOLOv8n，导出静态 ONNX，在 Orange Pi AI Pro 8T（昇腾 310B4）上用 ATC 转成 OM，再用 USB 摄像头推理。

类别顺序：`five`、`four`、`one`、`three`、`two`。

- 过程、指标和板端结果见 [报告.md](报告.md)
- 路径、配额和不能改的约定见 [记忆.md](记忆.md)
- 传到 Orange Pi 的步骤见 [mobaxtem_orange_pi.md](mobaxtem_orange_pi.md)

数据集不在本仓库里。克隆 [Hand-Gesture-YOLOv8](https://github.com/dwivedi1997/Hand-Gesture-YOLOv8) 到 `datasets/hand-gesture-yolov8` 后再训练。

```bash
conda activate yolo
python train_and_export.py
python verify_onnx_cpu.py --image <一张验证图片>
```

开发板上在 `~/gesture` 执行 `bash atc_convert.sh none weights/gesture_yolov8n.onnx images`，然后 `python3 gesture_infer.py`。板上不要安装新的包。
