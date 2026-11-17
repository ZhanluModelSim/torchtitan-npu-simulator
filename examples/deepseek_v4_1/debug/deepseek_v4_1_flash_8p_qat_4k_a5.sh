#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4.1 Flash CPT launcher. Keep the shared
# model, training, and parallelism defaults in the A3 script; this entrypoint
# only supplies A5-specific runtime and fused-operator settings.
# This entry enables routed-expert MXFP4 fake quant, V4.1 QLI/QSLI, and
# sparse-attention/KV-source Compressor quantization.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_qat_4k_a5.sh --training.steps 5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# The quantized sparse attention (torchao_npu's QuantV41SparseAttention) is its own backend:
# its converter-installed config builds the attention entry, so this entry launches no per-layer
# metadata operator.  It replaces the node `sparse_attn.asc` targets, so that override (kept
# below for the unquantized variants of this list) is a no-op here.
# The override list is the non-tilelang fused branch of
# deepseek_v4_1_flash_8p_cpt_4k_a5.sh -- keep the two lists in sync.
export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                       torchtitan_npu.override.common.swiglu_group.asc \
                       torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
                       torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc \
                       torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc}"

QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.enable-mxfp4-qat
    --extension.quantization.li-quantization mxfp4
    --extension.quantization.enable-sparse-attention-quantization
)

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_8p_cpt_4k_a5.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    "$@" \
