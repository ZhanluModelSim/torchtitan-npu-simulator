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
# Optional graph-pattern compile controls:
#
#   PATTERN_IMPORTS=<pattern-import-path> \
#   COMPILE_BACKEND=inductor ./scripts/run_train.sh
#
# Profiling is off by default. Enable it explicitly with ENABLE_PROFILING=1
# (optionally combined with the PROFILE_START/PROFILE_END window below):
#
#   ENABLE_PROFILING=1 PROFILE_START=5 PROFILE_END=6 ./scripts/run_train.sh

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

NGPU=${NGPU:-1}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan.models.deepseek_v3"}
CONFIG=${CONFIG:-"deepseek_v3_debugmodel"}
TRAIN_FILE=${TRAIN_FILE:-torchtitan_npu.train}
COMM_MODE=${COMM_MODE:-}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-}
export TORCHTITAN_NPU_PATTERN_IMPORTS="${PATTERN_IMPORTS:-${TORCHTITAN_NPU_PATTERN_IMPORTS:-}}"

ARGS=()

ENABLE_PROFILING=${ENABLE_PROFILING:-0}
if [ "${ENABLE_PROFILING}" = "1" ]; then
    ARGS+=(--profiler.enable-profiling)
fi

if [ -n "${PROFILE_START:-}" ] || [ -n "${PROFILE_END:-}" ]; then
    if [[ ! "${PROFILE_START:-}" =~ ^[1-9][0-9]*$ ]] || [[ ! "${PROFILE_END:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "PROFILE_START and PROFILE_END must be positive integers" >&2
        exit 2
    fi
    PROFILE_WARMUP=${PROFILE_WARMUP:-3}
    if [[ ! "${PROFILE_WARMUP}" =~ ^[0-9]+$ ]]; then
        echo "PROFILE_WARMUP must be a non-negative integer" >&2
        exit 2
    fi
    if [ "${PROFILE_END}" -le "${PROFILE_START}" ]; then
        echo "PROFILE_END must be greater than PROFILE_START" >&2
        exit 2
    fi

    PROFILE_SKIP_FIRST=$(( PROFILE_START > PROFILE_WARMUP ? PROFILE_START - PROFILE_WARMUP - 1 : 0 ))
    PROFILE_WARMUP_STEPS=$(( PROFILE_START - 1 - PROFILE_SKIP_FIRST ))
    PROFILE_ACTIVE=$(( PROFILE_END - PROFILE_START ))
    PROFILE_FREQ=$(( PROFILE_WARMUP_STEPS + PROFILE_ACTIVE ))
    ARGS+=(
        --profiler.profile-freq "${PROFILE_FREQ}"
        --profiler.profiler-warmup "${PROFILE_WARMUP_STEPS}"
        --profiler.profiler-active "${PROFILE_ACTIVE}"
        --profiler.profiler-repeat 1
        --profiler.profiler-skip-first "${PROFILE_SKIP_FIRST}"
    )
fi

if [ -n "${COMPILE_BACKEND:-}" ]; then
    ARGS+=(
        --compile.enable
        --compile.components model
        --compile.backend "${COMPILE_BACKEND}"
    )
fi
ARGS+=("$@")

if [[ -n "${COMM_MODE}" ]]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" \
        --module "${MODULE}" --config "${CONFIG}" \
        --comm.mode="${COMM_MODE}" "${ARGS[@]}"
else
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    CPU_AFFINITY_CONF=1 \
    TASK_QUEUE_ENABLE=2 \
    HCCL_CONNECT_TIMEOUT=3600 \
    STREAMS_PER_DEVICE=32 \
    MULTI_STREAM_MEMORY_RESERVE=1 \
    TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
    torchrun --nproc_per_node="${NGPU}" --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 \
    -m "${TRAIN_FILE}" --module "${MODULE}" --config "${CONFIG}" "${ARGS[@]}"
fi
