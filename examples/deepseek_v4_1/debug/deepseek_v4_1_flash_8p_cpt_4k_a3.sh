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
CONFIG="${CONFIG:-deepseek_v4_1_flash_40layers_16experts_multimodal_compiled}"

# Dataloader
DATASET="${DATASET:-cc12m-test}"
DATASET_PATH="${DATASET_PATH:-}" # optional; unset keeps the upstream dataset source
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV41_tokenizer}" # your tokenizer path

# Parallelism
TP=1
PP=1
EP=8
CP=1
DP_SHARD=8
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))

# Training
# Retain the validated eager / Muon / FullAC resource crop.
SEQ_LEN=4096
NUM_TOKENS_PER_MICROBATCH_PER_DP_RANK=4096
NUM_TOKENS_PER_TRAIN_STEP=32768
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

# Native Grain dataset configuration is selected when the recipe is constructed.
export DATASET DATASET_PATH
export PACKING_BUFFER_SIZE="${PACKING_BUFFER_SIZE:-0}"

# Parallelism
PARALLELISM_ARGS="
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
    --parallelism.fsdp-reshard-after-forward always
    --parallelism.enable-sequence-parallel
"

# Compilation policy is declared by the selected Python recipe.

# Training
TRAINING_ARGS="
    --training.num-tokens-per-microbatch-per-dp-rank ${NUM_TOKENS_PER_MICROBATCH_PER_DP_RANK}
    --training.num-tokens-per-train-step ${NUM_TOKENS_PER_TRAIN_STEP}
    --training.max-context-length ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
    --metrics.log-freq 1
"

# LR scheduler
LR_SCHEDULER_ARGS="
    --optim.lr-scheduler.warmup-steps 2
    --optim.lr-scheduler.total-steps ${STEPS}
    --optim.lr-scheduler.decay-ratio 1.0
    --optim.lr-scheduler.decay-type cosine
    --optim.lr-scheduler.min-lr-factor 0.01
"

# Checkpoint policy is declared by the selected Python recipe.
CHECKPOINT_ARGS=""

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
    --optim.optimizer.name Muon
    --optim.optimizer.lr 1.0e-5
    --optim.optimizer.beta1 0.9
    --optim.optimizer.beta2 0.95
    --optim.optimizer.eps 1.0e-8
    --optim.optimizer.weight-decay 1.0e-1
    --optim.optimizer.muon_momentum 0.95
    --optim.optimizer.muon_enable_nesterov
    --optim.optimizer.muon_ns_steps 10
    --optim.optimizer.muon_adjust_lr_fn match_rms_adamw
"
OPTIMIZER_OVERRIDES="${OPTIMIZER_OVERRIDES-torchtitan_npu.override.common.optimizer.swap_optimizer}"

# The text-rope entry is a hardware choice: A3 defaults to asc_complex; the A5
# wrapper supplies asc_partial through the same CLI_OVERRIDES channel. The fused
# sparse attention and its selector also ride this channel, so a wrapper that
# already names them does not add a second copy. Set the defaults in both branches
# so `set -u` never sees them unbound.
if [[ "${USE_GOLDEN}" == "1" ]]; then
    NPU_OPS_OVERRIDES=(torchtitan_npu.override.common.rope.workaround)
    DEFAULT_TEXT_ROPE_OVERRIDE=""
    DEFAULT_FUSED_OVERRIDES=""
else
    DEFAULT_TEXT_ROPE_OVERRIDE="torchtitan_npu.override.common.rope.asc_complex"
    DEFAULT_FUSED_OVERRIDES="torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc"
    MHC_POST_OVERRIDE="${MHC_POST_OVERRIDE:-torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post}"
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rms_norm.asc
        torchtitan_npu.override.common.rope.asc_half_rotation
        torchtitan_npu.override.common.token_dispatcher.asc
        ${MHC_POST_OVERRIDE}
    )
fi

# Wrapper defaults extend override.imports after the base targets.
CLI_OVERRIDES="${CLI_OVERRIDES:-${DEFAULT_TEXT_ROPE_OVERRIDE} ${DEFAULT_FUSED_OVERRIDES}}"

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train.sh \
    $HF_ASSETS_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $LR_SCHEDULER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    $CLI_OVERRIDES ${DATALOADER_OVERRIDES:-} \
    "$@"
