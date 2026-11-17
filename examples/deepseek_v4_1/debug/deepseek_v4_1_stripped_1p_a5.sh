#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Single-card DeepSeek-V4.1 with the fused ds41 LightningIndexer.
#
# The flavor behind this wrapper (`deepseek_v4_1_stripped`) is the flash model with six
# layers instead of forty; ``dim``, the attention heads and both released geometries are
# untouched, because the MLA attention and the ds41 indexer are both fixed by their kernels.
# See ``_stripped`` in ``torchtitan_npu/models/deepseek_v4_1/__init__.py``.
#
# Six layers is the smallest count that still reaches all three branches of the fused
# selector: layer 2 is Full Mode at ratio 2 with the pool off, layer 3 is the Full Mode pool
# source at ratio 1, and layer 4 is the Reindex Mode searcher that reads what layer 3 built.
# The candidate pool is a ratio-1 mechanism in the released model too -- its source sits at
# the first ratio-1 layer -- so this shape does not invent a combination the model lacks.
#
# Run from the repository root:
#   NGPU=1 bash examples/deepseek_v4_1/debug/deepseek_v4_1_stripped_1p_a5.sh --training.steps 5
#
# Append CLI arguments to override the defaults below. `QUANTIZATION_ARGS` swaps the whole
# quantized stack in -- QLI/QSLI for the selector and the quantized sparse attention for the
# teacher provider -- and needs `custom_ops`; without those flags the recipe runs both BF16 ports.

set -euo pipefail

export NGPU="${NGPU:-1}"
export MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4_1}"
export CONFIG="${CONFIG:-deepseek_v4_1_stripped}"
export PYTHONUNBUFFERED=1
export TORCHTITAN_ENGRAM_TABLE_ROWS="${TORCHTITAN_ENGRAM_TABLE_ROWS:-100000}"

# The fused operators this flavor exists to exercise.  `sparse_attn.asc` is not optional
# alongside `lightning_indexer.asc`: the indexer is trained purely through the teacher that
# SMLA's backward emits, and the model refuses a half-fused stack rather than training an
# indexer that never receives a gradient.
#
LIGHTNING_INDEXER_OVERRIDE="torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc"

# The quantized selector launches the quantized sparse attention's schedules, so the two switches
# go together: without the sparse-attention quantization the entry carries the unquantized kernel's
# metadata, which is a different schedule.  Needs `custom_ops` (kv_compress_epilog_v2).
QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.li-quantization mxfp4
    --extension.quantization.enable-sparse-attention-quantization
)

CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rms_norm.asc \
                               torchtitan_npu.override.common.rope.asc_complex \
                               torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post \
                               ${LIGHTNING_INDEXER_OVERRIDE} \
                               torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc}"

exec bash scripts/run_train.sh \
    --hf-assets-path "${HF_ASSETS_PATH:-tests/assets/deepseek_v3}" \
    --dataloader.dataset "${DATASET:-c4_test}" \
    --dataloader.dataset-path "${DATASET_PATH:-tests/assets/c4_test}" \
    --training.local-batch-size "${MBS:-1}" \
    --training.global-batch-size "${GBS:-1}" \
    --training.seq-len "${SEQ_LEN:-2048}" \
    --training.steps "${STEPS:-5}" \
    --parallelism.data-parallel-shard-degree 1 \
    --parallelism.data-parallel-replicate-degree 1 \
    --parallelism.expert-parallel-degree 1 \
    --parallelism.tensor-parallel-degree 1 \
    --parallelism.pipeline-parallel-degree 1 \
    --parallelism.context-parallel-degree 1 \
    --compile.no-enable \
    --checkpoint.no-enable \
    --metrics.log-freq 1 \
    --override.imports ${CLI_OVERRIDES} \
    "${QUANTIZATION_ARGS[@]}" \
    "$@"
