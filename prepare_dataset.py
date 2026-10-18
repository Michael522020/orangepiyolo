# -*- coding: utf-8 -*-
"""
把 Sign Language Digits 的分类图片收成 YOLOv8 检测集。

原始数据只有类别文件夹，没有框。手在浅色背景上，用 Otsu 前景得到框。
输出：
    datasets/gesture/images/train|val
    datasets/gesture/labels/train|val
    datasets/gesture_preview.jpg

运行：
    E:\\yolo\\python.exe prepare_dataset.py
"""

import os
import random
import shutil

import cv2
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

SOURCE = os.path.join("datasets", "Sign-Language-Digits-Dataset", "Dataset")
TARGET = os.path.join("datasets", "gesture")
CLASS_NAMES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
VAL_RATIO = 0.2
SEED = 17


def foreground_box(image):
    """返回归一化 xywh。前景太小或找不到时退回接近整图的框。"""
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.5, 0.5, 1.0, 1.0
    contour = max(contours, key=cv2.contourArea)
    x, y, box_w, box_h = cv2.boundingRect(contour)
    if box_w * box_h < 0.04 * width * height:
        return 0.5, 0.5, 1.0, 1.0
    pad_x = int(round(0.06 * width))
    pad_y = int(round(0.06 * height))
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + box_w + pad_x)
    y2 = min(height, y + box_h + pad_y)
    return ((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height, (x2 - x1) / width, (y2 - y1) / height


def reset_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)


def main():
    if not os.path.isdir(SOURCE):
        raise SystemExit("找不到原始数据: {}".format(os.path.abspath(SOURCE)))
    random.seed(SEED)
    for split in ("train", "val"):
        reset_dir(os.path.join(TARGET, "images", split))
        reset_dir(os.path.join(TARGET, "labels", split))

    counts = {"train": 0, "val": 0}
    areas = []
    preview = None
    for class_id, name in enumerate(CLASS_NAMES):
        folder = os.path.join(SOURCE, str(class_id))
        files = [item for item in os.listdir(folder) if item.lower().endswith((".jpg", ".jpeg", ".png"))]
        files.sort()
        random.shuffle(files)
        val_count = max(1, int(round(len(files) * VAL_RATIO)))
        for index, filename in enumerate(files):
            split = "val" if index < val_count else "train"
            image = cv2.imread(os.path.join(folder, filename))
            if image is None:
                continue
            cx, cy, bw, bh = foreground_box(image)
            areas.append(bw * bh)
            stem = "{}_{:04d}".format(name, index)
            cv2.imwrite(os.path.join(TARGET, "images", split, stem + ".jpg"), image)
            with open(os.path.join(TARGET, "labels", split, stem + ".txt"), "w", encoding="utf-8") as handle:
                handle.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(class_id, cx, cy, bw, bh))
            counts[split] += 1
            if preview is None and split == "val":
                preview = image.copy()
                height, width = image.shape[:2]
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(preview, name, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    if preview is not None:
        cv2.imwrite("datasets/gesture_preview.jpg", preview)
    areas = np.array(areas, dtype=np.float32)
    print("train={} val={}".format(counts["train"], counts["val"]))
    print("box area median={:.3f} min={:.3f} max={:.3f}".format(float(np.median(areas)), float(areas.min()), float(areas.max())))


if __name__ == "__main__":
    main()
