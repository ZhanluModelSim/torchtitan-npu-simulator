#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on each participating node; NODE_IPS lists all node addresses.
# Append CLI arguments to override the defaults below:
#   NODE_IPS=192.168.1.10,192.168.1.11,192.168.1.12,192.168.1.13,192.168.1.14,192.168.1.15,192.168.1.16,192.168.1.17 \
#     ./examples/deepseek_v4_1/deepseek_v4_1_flash_cpt_4k_a3.sh \
#     --checkpoint.initial-load-path /path/to/model_ckpt --training.steps 5
# USE_GOLDEN=1 selects Golden. For deterministic execution, append
# --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-16}"
NNODES=$(awk -F, '{print NF}' <<< "${NODE_IPS}")
WORLD_SIZE=$((NGPU * NNODES))

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}"
CONFIG="${CONFIG:-deepseek_v4_1_flash}"

# Dataloader & Checkpoint
DATASET="${DATASET:-cc12m-test}"
DATASET_PATH="${DATASET_PATH:-}" # optional; unset keeps the upstream dataset source
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV41_tokenizer}" # your tokenizer path
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-/path/to/save_ckpt}" # your model save/load ckpt path
CKPT_INIT_LOAD_PATH="${CKPT_INIT_LOAD_PATH:-/path/to/init_load_ckpt}" # your model initial load ckpt path

# Parallelism
TP=1
PP=1
EP="${EP:-128}"
CP=1
DP_SHARD="${DP_SHARD:-128}"
if (( WORLD_SIZE <= 0 || DP_SHARD <= 0 || EP <= 0 || WORLD_SIZE % DP_SHARD != 0 || WORLD_SIZE % EP != 0 || 384 % EP != 0 )); then
    echo "Invalid topology: WORLD_SIZE must divide into DP_SHARD/EP groups; EP must divide 384 experts." >&2
    exit 1
fi
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
# Full Flash uses eager execution and the recipe-owned FullAC policy.
SEQ_LEN=4096
MBS="${MBS:-1}"
GBS="${GBS:-1024}"
STEPS="${STEPS:-100}"

# Debug
USE_GOLDEN="${USE_GOLDEN:-0}"
DEBUG_ARGS="
    --debug.moe-force-load-balance
    --debug.print-config
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
    DEFAULT_CLI_OVERRIDES=""
    NPU_OPS_OVERRIDES=(torchtitan_npu.override.common.rope.workaround)
else
    DEFAULT_CLI_OVERRIDES="torchtitan_npu.override.common.rope.asc_complex"
    MHC_POST_OVERRIDE="${MHC_POST_OVERRIDE:-torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post}"
    NPU_OPS_OVERRIDES=(
        torchtitan_npu.override.common.rms_norm.asc
        # Vision RoPE (V4.1 keeps the text rope in CLI_OVERRIDES)
        torchtitan_npu.override.common.rope.asc_half_rotation
        # MHC
        ${MHC_POST_OVERRIDE}
        # MoE token dispatcher
        torchtitan_npu.override.common.token_dispatcher.asc
    )
fi

# Wrapper defaults extend override.imports after the base targets.
CLI_OVERRIDES="${CLI_OVERRIDES:-${DEFAULT_CLI_OVERRIDES}}"

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NODE_IPS="${NODE_IPS}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train_multinodes.sh \
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
