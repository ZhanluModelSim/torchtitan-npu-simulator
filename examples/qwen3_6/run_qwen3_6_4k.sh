#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    source /home/developer/Ascend/ascend-toolkit/set_env.sh
fi

: "${HF_ASSETS_PATH:?Set HF_ASSETS_PATH to the Qwen3.5/Qwen3.6 Hugging Face model directory}"
: "${DATA_FILES:?Set DATA_FILES to the CANNBOT 4K SFT JSONL file}"

NGPU=${NGPU:-16}
SEQ_LEN=${SEQ_LEN:-4096}
TRAIN_FILE=${TRAIN_FILE:-torchtitan_npu.train}
((SEQ_LEN >= 64 && SEQ_LEN <= 4096 && SEQ_LEN % 64 == 0)) || {
    echo "SEQ_LEN must be a multiple of 64 in [64, 4096], got ${SEQ_LEN}" >&2
    exit 1
}
export LOG_RANK=${LOG_RANK:-0}
export DATA_FILES

ARGS=(
    --module torchtitan_npu.models.qwen3_5.config_registry
    --config qwen35_27b_4k_sft
    --hf-assets-path "${HF_ASSETS_PATH}"
    --override.imports torchtitan_npu.override.qwen3_5.gated_delta.npu_gated_delta
    --checkpoint.enable
    --checkpoint.initial-load-path "${CHECKPOINT_INITIAL_LOAD_PATH:-${HF_ASSETS_PATH}}"
    --checkpoint.initial-load-in-hf
    --checkpoint.initial-load-model-only
    --dump-folder "${DUMP_FOLDER:-./outputs/qwen3_6_4k_sft}"
    --training.seq-len "${SEQ_LEN}"
    "$@"
)

PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True} \
    CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1} \
    CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1} \
    TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2} \
    HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-3600} \
    STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32} \
    MULTI_STREAM_MEMORY_RESERVE=${MULTI_STREAM_MEMORY_RESERVE:-1} \
    torchrun --nproc-per-node "${NGPU}" --rdzv-backend c10d --rdzv-endpoint localhost:0 \
    --local-ranks-filter "${LOG_RANK}" --role rank --tee 3 -m "${TRAIN_FILE}" \
    "${ARGS[@]}"
