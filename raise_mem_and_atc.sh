#!/bin/bash
# 放开当前终端所在内存配额，再用现有内存把 ONNX 转成 OM。
# 在 Orange Pi 上执行：bash raise_mem_and_atc.sh
set -euo pipefail

cd "$(dirname "$0")"

P="$(awk -F: '$2 ~ /memory/ {print $3; exit}' /proc/self/cgroup)"
if [[ -z "${P}" || ! -d /sys/fs/cgroup/memory ]]; then
    echo "没有找到 cgroup v1 的 memory 控制器。"
    echo "----- /proc/self/cgroup -----"
    cat /proc/self/cgroup
    echo "----- /sys/fs/cgroup -----"
    ls /sys/fs/cgroup
    exit 1
fi

echo "memory cgroup: ${P}"
node="${P}"
list=""
while true; do
    list="${node} ${list}"
    [[ "${node}" == "/" ]] && break
    node="$(dirname "${node}")"
done

for node in ${list}; do
    for name in memory.memsw.limit_in_bytes memory.limit_in_bytes; do
        f="/sys/fs/cgroup/memory${node}/${name}"
        if [[ -f "${f}" ]]; then
            echo "before ${f} = $(cat "${f}")"
            echo -1 | sudo tee "${f}" >/dev/null
            echo "after  ${f} = $(cat "${f}")"
        fi
    done
done

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TE_PARALLEL_COMPILER=1
bash atc_convert.sh none weights/gesture_yolov8n.onnx images
