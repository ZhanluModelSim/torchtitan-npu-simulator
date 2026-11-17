#!/usr/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Generic single-node TorchTitan launcher. Model, config, dataset, and
# override arguments belong in an example under examples/; extra command-line
# arguments are passed through unchanged.
#
#   ./scripts/run_train.sh --compile.enable --compile.components model --compile.backend inductor

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

# Set the Inductor backend to AscendC.
export TORCHINDUCTOR_NPU_BACKEND="${TORCHINDUCTOR_NPU_BACKEND:-ascendc}"

NGPU=${NGPU:-1}
export LOG_RANK=${LOG_RANK:-0}
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
MODULE=${MODULE:-"torchtitan.models.deepseek_v3"}
CONFIG=${CONFIG:-"deepseek_v3_debugmodel"}
TRAIN_FILE=${TRAIN_FILE:-torchtitan_npu.train}
COMM_MODE=${COMM_MODE:-}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-}

time=$(date +%Y%m%d%H%M)
logfile=${LOG_PREFIX:-${CONFIG}}_${time}.log
mkdir -p logs

if [[ -n "${COMM_MODE}" ]]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" \
        --module "${MODULE}" --config "${CONFIG}" \
        --comm.mode="${COMM_MODE}" "$@"
else
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}" \
    TASK_QUEUE_ENABLE=2 \
    HCCL_CONNECT_TIMEOUT=3600 \
    STREAMS_PER_DEVICE=32 \
    MULTI_STREAM_MEMORY_RESERVE=1 \
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
    torchrun --nproc_per_node="${NGPU}" --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
    -m "${TRAIN_FILE}" --module "${MODULE}" --config "${CONFIG}" "$@" 2>&1 | tee -a logs/${logfile}
fi
