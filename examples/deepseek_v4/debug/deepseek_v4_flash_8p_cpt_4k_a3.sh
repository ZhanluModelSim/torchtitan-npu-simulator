#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a3.sh --training.steps 5
# USE_GOLDEN=1 selects Golden. For deterministic execution, append
# --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail

# Enable model compilation by default; callers can override the backend.
export COMPILE_BACKEND="${COMPILE_BACKEND:-aot_eager}"

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
CONFIG="${CONFIG:-deepseek_v4_flash_43layers_16experts}"

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-tests/assets/c4_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV4_tokenizer}" # your tokenizer path
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-/path/to/save_ckpt}" # your model save/load ckpt path

# Parallelism
TP=1
PP=1
EP=8
CP=1
DP_SHARD=8
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
SEQ_LEN=4096
MBS=1
GBS=64
STEPS=100

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.no-moe-force-load-balance
    --debug.print-config
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
# `checkpoint.folder` is the save/load root (`CKPT_SAVE_LOAD_PATH`). If it
# already contains a valid step-* checkpoint, upstream TorchTitan resumes from
# it; use a new/empty folder when starting a fresh run.
CHECKPOINT_ARGS="
    --checkpoint.no-enable
    --checkpoint.load-only
    --checkpoint.folder ${CKPT_SAVE_LOAD_PATH}
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
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train.sh \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" "${PROFILER_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "$@"
