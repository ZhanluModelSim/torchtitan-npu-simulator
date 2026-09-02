#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if [ -n "${ASCEND_SET_ENV_PATH:-}" ]; then
    source "${ASCEND_SET_ENV_PATH}"
elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /home/developer/Ascend/ascend-toolkit/set_env.sh ]; then
    source /home/developer/Ascend/ascend-toolkit/set_env.sh
fi

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-7200}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"
export STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}"
export MULTI_STREAM_MEMORY_RESERVE="${MULTI_STREAM_MEMORY_RESERVE:-1}"

Network_Interface=${Network_Interface:-$(ip -o -4 addr show scope global | awk '$2 != "docker0" {print $2; exit}')}
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${Network_Interface}}"
export HCCL_SOCKET_IFNAME="${HCCL_SOCKET_IFNAME:-${Network_Interface}}"
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-30000}"

export LOG_RANK="${LOG_RANK:-0}"  # rank to show log
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export TORCHTITAN_NPU_PATTERN_IMPORTS="${PATTERN_IMPORTS:-${TORCHTITAN_NPU_PATTERN_IMPORTS:-}}"

LOCAL_HOST=${LOCAL_HOST:-$(ip addr show "${Network_Interface}" | grep "inet " | awk '{print $2}' | cut -d'/' -f1 | head -n1)}
LOCAL_HOST=${LOCAL_HOST:-$(hostname -I | awk '{print $1}')}
echo "$LOCAL_HOST"
if [[ -n "${NODE_IPS:-}" ]]; then
    read -r -a IPs <<< "${NODE_IPS//,/ }"
else
    IPs=("${LOCAL_HOST}")
fi
NGPU=${NGPU:-8}
NPUS_PER_NODE=${NGPU}
MASTER_ADDR=${MASTER_ADDR:-${IPs[0]}}
MASTER_PORT=${MASTER_PORT:-6300}
NNODES=${NNODES:-${#IPs[@]}}
NODE_RANK=${NODE_RANK:-}
for i in "${!IPs[@]}"; do
    if [[ "$LOCAL_HOST" == "${IPs[$i]}" ]]; then
        echo "Node Rank : ${i}"
        NODE_RANK=$i
        break
    fi
done
if [[ $NODE_RANK == "" ]]; then
    echo "[Error] Variable \"NODE_RANK\" must be configured"
    exit 1
fi

MODULE=${MODULE:-"torchtitan.models.deepseek_v3"}
CONFIG=${CONFIG:-"deepseek_v3_debugmodel"}
TRAIN_FILE=${TRAIN_FILE:-torchtitan_npu.train}
time=$(date +%Y%m%d%H%M)
logfile=${LOG_PREFIX:-${CONFIG}}_${time}_node${NODE_RANK}_${LOCAL_HOST//./_}.log
mkdir -p logs

ARGS=()
if [ -n "${COMPILE_BACKEND:-}" ]; then
    ARGS+=(
        --compile.enable
        --compile.components model
        --compile.backend "${COMPILE_BACKEND}"
    )
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --nproc_per_node=${NPUS_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --local-ranks-filter ${LOG_RANK} \
    --role rank \
    --tee 3 \
    -m "${TRAIN_FILE}" \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    "${ARGS[@]}" "$@" 2>&1 | tee -a logs/${logfile}
