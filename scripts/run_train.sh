# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# [TODO]
# current branch is not stable , please clone torchtitan and checkout ac13e536.

NGPU=${NGPU:-"16"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_npu.models.deepseek_v3"}
CONFIG=${CONFIG:-"deepseek_v3_671b_debug"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan_npu.entry"}
COMM_MODE=${COMM_MODE:-""}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}


if [ -n "$COMM_MODE" ]; then
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" \
        --module "${MODULE}" --config "${CONFIG}" \
        --comm.mode=${COMM_MODE} --training.steps=1 "$@"
else
    PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    CPU_AFFINITY_CONF=1 \
    TASK_QUEUE_ENABLE=2 \
    HCCL_CONNECT_TIMEOUT=3600 \
    STREAMS_PER_DEVICE=32 \
    MULTI_STREAM_MEMORY_RESERVE=1 \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@"
fi
