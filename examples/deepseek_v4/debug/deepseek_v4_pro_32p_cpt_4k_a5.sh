#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on each participating node; NODE_IPS lists all node addresses.
# Append CLI arguments to override the defaults below:
#   NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13 \
#     ./examples/deepseek_v4/debug/deepseek_v4_pro_32p_cpt_4k_a5.sh \
#     --checkpoint.initial-load-path /path/to/model_ckpt --training.steps 5
# USE_GOLDEN=1 selects Golden. For deterministic execution, append
# --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17330}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-2147480}"
export HCCL_EVENT_TIMEOUT="${HCCL_EVENT_TIMEOUT:-2147480}"
# Disable the HCCL watchdog for long checkpoint loading on the A5 setup.
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
# Use the command line `npu-smi info -t topo` to query the CPU Affinity of the NPU cards for configuration.
export CPU_AFFINITY_CONF=1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-8}"
NNODES=$(awk -F, '{print NF}' <<< "${NODE_IPS}")
WORLD_SIZE=$((NGPU * NNODES))

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
CONFIG="${CONFIG:-deepseek_v4_pro_61layers_32experts}"

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-tests/assets/c4_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV4_tokenizer}" # your tokenizer path
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-/path/to/save_ckpt}" # your model save/load ckpt path
CKPT_INIT_LOAD_PATH="${CKPT_INIT_LOAD_PATH:-/path/to/init_load_ckpt}" # your model initial load ckpt path

# Parallelism
TP="${TP:-1}"
PP="${PP:-1}"
EP="${EP:-32}"
CP="${CP:-1}"
DP_SHARD="${DP_SHARD:-32}"
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
SEQ_LEN="${SEQ_LEN:-4096}"
MBS=1
GBS="${GBS:-256}"
STEPS="${STEPS:-100}"

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.no-moe-force-load-balance
    --debug.print-config
"

# A5 defaults to block-FP8 quantized training. Append
# --extension.quantization.no-enable-quantized-training for BF16 training.
QUANTIZATION_ARGS="
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
"

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Dataloader
DATALOADER_ARGS="
    --dataloader.dataset ${DATASET}
    --dataloader.dataset-path ${DATASET_PATH}
"

# Parallelism
PARALLELISM_ARGS="
    --parallelism.spmd-backend ${SPMD_BACKEND}
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
"

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

# Checkpoint
# `checkpoint.folder` is the output/resume root (`CKPT_SAVE_LOAD_PATH`). If it
# already contains a valid step-* checkpoint, upstream TorchTitan resumes from
# it and ignores `initial-load-path`; use a new/empty folder when cold-starting
# from `CKPT_INIT_LOAD_PATH`.
CHECKPOINT_ARGS="
    --checkpoint.enable
    --checkpoint.load-only
    --checkpoint.folder ${CKPT_SAVE_LOAD_PATH}
    --checkpoint.initial-load-path ${CKPT_INIT_LOAD_PATH}
    --checkpoint.initial-load-in-hf
"

# Profiler
PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.profile-freq 1
    --profiler.profiler-warmup 0
    --profiler.profiler-active 1
    --profiler.profiler-repeat 1
    --profiler.profiler-skip-first 4
"
PROFILER_OVERRIDES=(
    'torchtitan_npu.override.common.profiler.cann={
        "profile_ranks": [0],
        "profile_with_memory": false,
        "profile_with_stack": false,
        "enable_online_parse": false
    }'
)

# Communication
COMM_ARGS="
    --comm.init-timeout-seconds 7200
    --comm.train-timeout-seconds 600
"

# Optimizer & LR scheduler
OPTIMIZER_ARGS="
    --optimizer.implementation fused
    --optimizer.param-groups.0.optimizer-name AdamW
    --optimizer.param-groups.0.optimizer-kwargs.lr 1.0e-5
    --optimizer.param-groups.0.optimizer-kwargs.betas 0.9 0.95
    --optimizer.param-groups.0.optimizer-kwargs.eps 1.0e-6
    --optimizer.param-groups.0.optimizer-kwargs.weight-decay 1.0e-1
    --lr-scheduler.warmup-steps 25
    --lr-scheduler.decay-type cosine
    --lr-scheduler.decay-ratio 1.0
    --lr-scheduler.min-lr-factor 1.0e-2
"
OPTIMIZER_OVERRIDES="
    torchtitan_npu.override.common.optimizer.virtual
    torchtitan_npu.override.common.optimizer.checkpoint_virtual
"

if [[ "${USE_GOLDEN}" == "1" ]]; then
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rope.workaround
        torchtitan_npu.override.deepseek_v4.sparse_attn.golden
    )
else
    NPU_OPS_OVERRIDES=(
        # Attention / DSA
        torchtitan_npu.override.common.rms_norm.asc
        torchtitan_npu.override.common.rope.asc_complex
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc_metadata
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc
        # MHC
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post
        # MoE token dispatcher
        torchtitan_npu.override.common.token_dispatcher.asc
    )
fi

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NODE_IPS="${NODE_IPS}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train_multinodes.sh \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $QUANTIZATION_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" "${PROFILER_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "$@"
