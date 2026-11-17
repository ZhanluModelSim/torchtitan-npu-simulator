#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run this script on a single node.
# Append CLI arguments to override the defaults below:
#   ./examples/qwen3_5/qwen3_5_debugmodel_moe_8p_cpt_512.sh
# ENABLE_NPU_MOE_DISPATCHER=1 opts in the ASC MoE token dispatcher.
# Set ASCEND_SET_ENV_PATH to select a specific CANN toolkit.

set -euo pipefail

NGPU="${NGPU:-8}"
WORLD_SIZE="${NGPU}"

# Model
MODULE="${MODULE:-torchtitan_npu.models.qwen3_5}"
CONFIG="${CONFIG:-qwen35_debugmodel_moe}"

# Dataloader: the multimodal test assets live in the TorchTitan checkout;
# TORCHTITAN_DIR (exported by .ci/smoke_test.sh) or a sibling checkout
# provides the tokenizer and cc12m-test data.
TORCHTITAN_REPO="${TORCHTITAN_REPO:-${TORCHTITAN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/../torchtitan}}"
DATASET="${DATASET:-cc12m-test}"
DATASET_PATH="${DATASET_PATH:-${TORCHTITAN_REPO}/tests/assets/cc12m_test}" # your data path
HF_ASSETS_PATH="${HF_ASSETS_PATH:-${TORCHTITAN_REPO}/tests/assets/tokenizer}" # your tokenizer path

# Parallelism
TP=1
PP=1
EP=8
CP=1
DP_SHARD=8
DP_REPLICATE=$((WORLD_SIZE / (DP_SHARD * CP * TP * PP)))
SPMD_BACKEND="spmd_types"

# Training (smoke-sized defaults)
SEQ_LEN=512
MBS=2
GBS=-1
STEPS=10

# NPU overrides: the Triton GDN kernel is always imported; the MoE token
# dispatcher is a performance option enabled via ENABLE_NPU_MOE_DISPATCHER=1.
# An explicitly set (possibly empty) OVERRIDE_IMPORTS wins verbatim.
if [[ ${OVERRIDE_IMPORTS+x} ]]; then
    :
else
    OVERRIDE_IMPORTS="torchtitan_npu.override.qwen3_5.gated_delta.npu"
    if [[ "${ENABLE_NPU_MOE_DISPATCHER:-0}" == "1" ]]; then
        OVERRIDE_IMPORTS="${OVERRIDE_IMPORTS},torchtitan_npu.override.common.token_dispatcher.asc"
    fi
fi

DEBUG_ARGS="
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

# Compile
COMPILE_ARGS="
    --compile.no_enable
    --compile.components model
    --compile.backend aot_eager
"

# Training
TRAINING_ARGS="
    --training.local-batch-size ${MBS}
    --training.global-batch-size ${GBS}
    --training.seq-len ${SEQ_LEN}
    --training.steps ${STEPS}
    --training.disable-cuda-graphs
"

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
    --override.imports "${OVERRIDE_IMPORTS}" \
    "$@"
