#!/bin/bash
# Orange Pi AI Pro（Ascend310B4）上把已验证的 ONNX 转成 OM。
# 在开发板已有的 CANN 环境里执行。不要 apt/pip 安装转换工具。
# 如果 shell 里找不到 atc，只 source 板上已经存在的环境脚本，例如：
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
# 用法（三选一，不要对同一个输出名重复覆盖后还用错预处理）：
#   bash atc_convert.sh none
#   bash atc_convert.sh aipp_rgb
#   bash atc_convert.sh aipp_yuv
#
# 第 2 个参数是 ONNX 路径，第 3 个参数是输入节点名。
# 节点名以 verify_onnx_cpu.py 打印的为准，ultralytics 默认一般是 images。
#   bash atc_convert.sh aipp_rgb weights/gesture_yolov8n.onnx images

set -euo pipefail

MODE="${1:-none}"
ONNX_MODEL="${2:-weights/gesture_yolov8n.onnx}"
INPUT_NAME="${3:-images}"
SOC_VERSION="Ascend310B4"

if [[ ! -f "${ONNX_MODEL}" ]]; then
    echo "找不到 ONNX: ${ONNX_MODEL}"
    echo "请先在 PC 上完成 train_and_export.py 和 verify_onnx_cpu.py，再把 onnx 拷到开发板。"
    exit 1
fi

if ! command -v atc >/dev/null 2>&1; then
    echo "当前 shell 找不到 atc。请先 source 板上已有的 CANN set_env.sh，不要重新安装 CANN。"
    exit 1
fi

COMMON_ARGS=(
    --model="${ONNX_MODEL}"
    --framework=5
    --input_format=NCHW
    --input_shape="${INPUT_NAME}:1,3,640,640"
    --soc_version="${SOC_VERSION}"
    --log=error
)

case "${MODE}" in
    none)
        # 不插入 AIPP。推理时输入必须是 float32 NCHW RGB，且已经 /255。
        # gesture_infer.py: input_mode = "float32"，model_path 指向这个 om。
        atc "${COMMON_ARGS[@]}" --output="gesture_yolov8n"
        echo "生成 gesture_yolov8n.om"
        echo "板端 CONFIG: input_mode=float32  model_path=gesture_yolov8n.om"
        ;;
    aipp_rgb)
        # USB 摄像头推荐这条。AIPP 吃 BGR uint8，并在 NPU 上交换 RB、除以 255。
        if [[ ! -f "aipp_rgb.cfg" ]]; then
            echo "找不到 aipp_rgb.cfg"
            exit 1
        fi
        atc "${COMMON_ARGS[@]}" \
            --insert_op_conf="aipp_rgb.cfg" \
            --output="gesture_yolov8n_aipp_rgb"
        echo "生成 gesture_yolov8n_aipp_rgb.om"
        echo "板端 CONFIG: input_mode=aipp_rgb  model_path=gesture_yolov8n_aipp_rgb.om"
        echo "此时预处理必须喂 uint8，不能喂 float32。"
        ;;
    aipp_yuv)
        # 仅当输入本身就是 NV12 时使用。配置见 aipp.cfg。
        if [[ ! -f "aipp.cfg" ]]; then
            echo "找不到 aipp.cfg"
            exit 1
        fi
        atc "${COMMON_ARGS[@]}" \
            --insert_op_conf="aipp.cfg" \
            --output="gesture_yolov8n_aipp_yuv"
        echo "生成 gesture_yolov8n_aipp_yuv.om"
        echo "板端 CONFIG: input_mode=aipp_yuv420sp  model_path=gesture_yolov8n_aipp_yuv.om"
        echo "此时预处理必须喂 NV12 uint8，不能喂 float32。"
        ;;
    *)
        echo "未知模式: ${MODE}"
        echo "可选: none | aipp_rgb | aipp_yuv"
        exit 1
        ;;
esac
