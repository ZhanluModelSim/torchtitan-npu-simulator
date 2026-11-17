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
# For deterministic execution, append --debug.seed 42 --debug.deterministic to the command line.

set -euo pipefail

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-8}"
NNODES=$(awk -F, '{print NF}' <<< "${NODE_IPS}")
WORLD_SIZE=$((NGPU * NNODES))

# Model
MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}"
CONFIG="${CONFIG:-deepseek_v4_1_flash_text}"

# Dataloader & Checkpoint
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-}" # optional; unset keeps the upstream dataset source
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/path/to/DeepSeekV41_tokenizer}" # your tokenizer path
CKPT_SAVE_LOAD_PATH="${CKPT_SAVE_LOAD_PATH:-/path/to/save_ckpt}" # your model save/load ckpt path
CKPT_INIT_LOAD_PATH="${CKPT_INIT_LOAD_PATH:-/path/to/init_load_ckpt}" # your model initial load ckpt path

# Parallelism
TP=1
PP=1
EP="${EP:-128}"
CP="${CP:-128}"
DP_SHARD="${DP_SHARD:-1}"
if (( WORLD_SIZE <= 0 || DP_SHARD <= 0 || EP <= 0 || WORLD_SIZE % DP_SHARD != 0 || WORLD_SIZE % EP != 0 || 384 % EP != 0 )); then
    echo "Invalid topology: WORLD_SIZE must divide into DP_SHARD/EP groups; EP must divide 384 experts." >&2
    exit 1
fi
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training
# Full Flash uses eager execution and the recipe-owned FullAC policy.
SEQ_LEN="${SEQ_LEN:-1048576}"
MBS="${MBS:-1}"
GBS="${GBS:-1}"
STEPS="${STEPS:-10}"

# Debug
DEBUG_ARGS="
    --debug.moe-force-load-balance
    --debug.print-config
    --no_engram_enabled
"

# HF assets
HF_ASSETS_ARGS="
    --hf-assets-path ${HF_ASSETS_PATH}
"

# Dataloader
DATALOADER_ARGS="
    --dataloader.dataset ${DATASET}
    --dataloader.per_doc_alignment 256
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
    --compile.no-enable
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
    --checkpoint.no-enable
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

NPU_OPS_OVERRIDES=(
    torchtitan_npu.override.common.rms_norm.asc \
    torchtitan_npu.override.common.rope.asc_partial \
    torchtitan_npu.override.common.token_dispatcher.asc \
    torchtitan_npu.override.common.swiglu_group.asc \
    torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
    torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn \
    torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post \
    torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc \
    torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc
)

# Wrapper defaults extend override.imports after the base targets.
CLI_OVERRIDES="${CLI_OVERRIDES:-}"

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
