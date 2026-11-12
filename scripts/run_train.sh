#!/usr/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Single-node TorchTitan launcher. Training settings belong in the selected Python
# recipe; extra native launch arguments are passed through unchanged.
#
#   MODULE=my_recipes CONFIG=my_run ./scripts/run_train.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

export NGPU="${NGPU:-1}"
export MODULE="${MODULE:-torchtitan_npu_recipes.models.deepseek_v4}"
export CONFIG="${CONFIG:-deepseek_v4_debugmodel}"
TRAIN_FILE="${TRAIN_FILE:-torchtitan_npu.train}"
COMM_BACKEND=${COMM_BACKEND:-}
if [[ -n "${COMM_BACKEND}" && "${COMM_BACKEND}" != fake ]]; then
    echo "COMM_BACKEND must be empty or fake; received ${COMM_BACKEND}." >&2
    exit 1
fi

if [ -n "${ASCEND_SET_ENV_PATH:-}" ]; then
    source "${ASCEND_SET_ENV_PATH}"
elif [ -f /usr/local/Ascend/cann/set_env.sh ]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /home/developer/Ascend/ascend-toolkit/set_env.sh ]; then
    source /home/developer/Ascend/ascend-toolkit/set_env.sh
fi

# Set the Inductor backend to AscendC.
export TORCHINDUCTOR_NPU_BACKEND="${TORCHINDUCTOR_NPU_BACKEND:-ascendc}"

export LOG_RANK=${LOG_RANK:-0}
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-}

time=$(date +%Y%m%d%H%M)
logfile=${LOG_PREFIX:-${CONFIG}}_${time}.log
mkdir -p logs

if [[ "${COMM_BACKEND}" == fake ]]; then
    LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" \
        --module "${MODULE}" --config "${CONFIG}" \
        --comm-backend fake "$@"
else
    PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}" \
    CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}" \
    CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}" \
    TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}" \
    HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}" \
    STREAMS_PER_DEVICE="${STREAMS_PER_DEVICE:-32}" \
    MULTI_STREAM_MEMORY_RESERVE="${MULTI_STREAM_MEMORY_RESERVE:-1}" \
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
    torchrun --nproc_per_node="${NGPU}" --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
    -m "${TRAIN_FILE}" --module "${MODULE}" --config "${CONFIG}" "$@" 2>&1 | tee -a "logs/${logfile}"
fi
