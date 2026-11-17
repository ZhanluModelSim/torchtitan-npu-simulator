#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a3.sh --training.steps 5
# USE_GOLDEN=1 selects Golden. For deterministic execution, append
# --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail
export TORCHTITAN_ENGRAM_TABLE_ROWS="${TORCHTITAN_ENGRAM_TABLE_ROWS:-100000}"

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}"
CONFIG="${CONFIG:-deepseek_v4_1_flash_40layers_16experts_multimodal}"

# Dataloader & Checkpoint
DATASET="${DATASET:-cc12m-test}"
DATASET_PATH="${DATASET_PATH:-}" # optional; unset keeps the upstream dataset source
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV41_tokenizer}" # your tokenizer path
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
# Retain the validated eager / Muon / FullAC resource crop.
SEQ_LEN=4096
MBS=1
GBS=8
STEPS=40

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.print-config
    --debug.moe-force-load-balance
"

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Dataloader
DATALOADER_ARGS="
    --dataloader.dataset ${DATASET}
"
if [[ -n "${DATASET_PATH}" ]]; then
    DATALOADER_ARGS+="
    --dataloader.dataset-path ${DATASET_PATH}
"
fi

# Parallelism
PARALLELISM_ARGS="
    --parallelism.spmd-backend ${SPMD_BACKEND}
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
    --parallelism.fsdp-reshard-after-forward always
    --parallelism.context-parallel-load-balancer None
    --parallelism.enable-sequence-parallel
"

# Compile
COMPILE_ARGS="
    --compile.enable
    --compile.components model
    --compile.backend inductor
"

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
    --metrics.log-freq 1
"

# LR scheduler
LR_SCHEDULER_ARGS="
    --lr-scheduler.warmup-steps 2
    --lr-scheduler.total-steps ${STEPS}
    --lr-scheduler.decay-ratio 1.0
    --lr-scheduler.decay-type cosine
    --lr-scheduler.min-lr-factor 0.01
"

# Checkpoint
# `checkpoint.folder` is the save/load root (`CKPT_SAVE_LOAD_PATH`). If it
# already contains a valid step-* checkpoint, upstream TorchTitan resumes from
# it; use a new/empty folder when starting a fresh run.
CHECKPOINT_ARGS="
    --checkpoint.no-enable
    --checkpoint.load-only
    --checkpoint.interval 10
    --checkpoint.folder ${CKPT_SAVE_LOAD_PATH}
"

# Profiler
PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.profile-freq 10
    --profiler.profiler-active 10
    --profiler.profiler-warmup 0
"

# Communication
COMM_ARGS="
    --comm.init-timeout-seconds 7200
    --comm.train-timeout-seconds 600
"

# Optimizer
# AdamW hyperparameters come from the recipe, not from CLI flags.
OPTIMIZER_ARGS="
    --optimizer.name Muon
    --optimizer.lr 1.0e-5
    --optimizer.beta1 0.9
    --optimizer.beta2 0.95
    --optimizer.eps 1.0e-8
    --optimizer.weight-decay 1.0e-1
    --optimizer.muon_momentum 0.95
    --optimizer.muon_enable_nesterov
    --optimizer.muon_ns_steps 10
    --optimizer.muon_adjust_lr_fn match_rms_adamw
"
OPTIMIZER_OVERRIDES="${OPTIMIZER_OVERRIDES-torchtitan_npu.override.common.optimizer.swap_optimizer}"

# The text-rope entry is a hardware choice: A3 defaults to asc_complex; the A5
# wrapper supplies asc_partial through the same CLI_OVERRIDES channel. Set the
# default in both branches so `set -u` never sees it unbound.
if [[ "${USE_GOLDEN}" == "1" ]]; then
    NPU_OPS_OVERRIDES=(torchtitan_npu.override.common.rope.workaround)
    DEFAULT_TEXT_ROPE_OVERRIDE=""
else
    DEFAULT_TEXT_ROPE_OVERRIDE="torchtitan_npu.override.common.rope.asc_complex"
    MHC_POST_OVERRIDE="${MHC_POST_OVERRIDE:-torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post}"
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rms_norm.asc
        torchtitan_npu.override.common.rope.asc_half_rotation
        torchtitan_npu.override.common.token_dispatcher.asc
        ${MHC_POST_OVERRIDE}
    )
fi

# Wrapper defaults extend override.imports after the base targets.
CLI_OVERRIDES="${CLI_OVERRIDES:-${DEFAULT_TEXT_ROPE_OVERRIDE}}"

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train.sh \
    $COMPILE_ARGS \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $LR_SCHEDULER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    $CLI_OVERRIDES \
    "$@"
