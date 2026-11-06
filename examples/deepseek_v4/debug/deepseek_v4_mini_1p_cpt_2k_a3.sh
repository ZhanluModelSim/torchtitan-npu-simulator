#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4/debug/deepseek_v4_mini_1p_cpt_2k_a3.sh --training.steps 5
# USE_GOLDEN=1 selects reference attention, the RoPE workaround and AdamW.
# For deterministic execution, append
# --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail

NGPU="${NGPU:-1}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}"
CONFIG="${CONFIG:-deepseek_v4_debugmodel}"

# Dataloader
# Dataset selection and source paths belong to the selected MODULE/CONFIG recipe.
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV4_tokenizer}" # your tokenizer path

# Parallelism
TP=1
PP=1
EP=1
CP="${CP:-1}"
DP_SHARD=1
DP_REPLICATE=1

# Training
SEQ_LEN=2048
NUM_TOKENS_PER_MICROBATCH_PER_DP_RANK=2048
NUM_TOKENS_PER_TRAIN_STEP=-1
STEPS=100

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.print-config
"

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Parallelism
PARALLELISM_ARGS="
    --parallelism.data-parallel-shard-degree ${DP_SHARD}
    --parallelism.data-parallel-replicate-degree ${DP_REPLICATE}
    --parallelism.expert-parallel-degree ${EP}
    --parallelism.tensor-parallel-degree ${TP}
    --parallelism.context-parallel-degree ${CP}
    --parallelism.pipeline-parallel-degree ${PP}
"

# Training
TRAINING_ARGS="
    --training.num-tokens-per-microbatch-per-dp-rank ${NUM_TOKENS_PER_MICROBATCH_PER_DP_RANK}
    --training.num-tokens-per-train-step ${NUM_TOKENS_PER_TRAIN_STEP}
    --training.max-context-length ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

# Checkpoint policy is declared by the selected Python recipe.
CHECKPOINT_ARGS=""

# Profiler
PROFILER_ARGS="
    --profiler.no-enable-profiling
    --profiler.save-traces-folder profiling_path
    --profiler.extension.no-enable-online-parse
    --profiler.extension.profiler-start 6
    --profiler.extension.profiler-end 7
    --profiler.extension.profile-ranks 0
"

# Communication
COMM_ARGS="
    --comm.init-timeout-seconds 7200
    --comm.train-timeout-seconds 600
"

# Optimizer
OPTIMIZER_ARGS="
    --optim.optimizer.name Muon
    --optim.optimizer.lr 1.0e-5
    --optim.optimizer.beta1 0.9
    --optim.optimizer.beta2 0.95
    --optim.optimizer.eps 1.0e-8
    --optim.optimizer.weight_decay 1.0e-1
    --optim.optimizer.muon_momentum 0.95
    --optim.optimizer.muon_enable_nesterov
    --optim.optimizer.muon_ns_steps 10
    --optim.optimizer.muon_adjust_lr_fn match_rms_adamw
"
OPTIMIZER_OVERRIDES="
    torchtitan_npu.override.common.optimizer.swap_optimizer
"

if [[ "${USE_GOLDEN}" == "1" ]]; then
    OPTIMIZER_ARGS="--optim.optimizer.name AdamW --optim.optimizer.lr 1.0e-5"
    OPTIMIZER_OVERRIDES=""
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rope.workaround
    )
else
    NPU_OPS_OVERRIDES=(
        # Attention / DSA
        torchtitan_npu.override.common.rms_norm.asc
        torchtitan_npu.override.common.rope.asc_complex

        torchtitan_npu.override.deepseek_v4.lightning_indexer.asc

        'torchtitan_npu.override.deepseek_v4.sparse_attn.asc={"num_heads":16,"index_num_heads":8}'
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
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "$@"
