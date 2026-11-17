#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4.1 Flash CPT launcher. Keep the shared
# model, training, and parallelism defaults in the A3 script; this entrypoint
# only supplies A5-specific runtime and fused-operator settings.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh --training.steps 5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Use the command line `npu-smi info -t topo` to query the CPU affinity of the
# NPU cards for configuration.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

# A5-only fused ops; USE_GOLDEN=1 keeps the pure reference list.
# USE_TILELANG=1 swaps the mHC pre/post entries to the TileLang overrides
# (swiglu/sparse-attn keep the asc fused path; the PR730/731 tilelang swiglu
# and topk adapters are single-rank-validated and reject the EP>1 DTensor
# weights this 8p recipe produces).
if [[ "${USE_GOLDEN:-0}" != "1" ]]; then
    if [[ "${USE_TILELANG:-0}" == "1" ]]; then
        export MHC_POST_OVERRIDE=torchtitan_npu.override.deepseek_v4_1.tilelang.tilelang_hc_post
        export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                              torchtitan_npu.override.deepseek_v4_1.tilelang.tilelang_hc_pre \
                              torchtitan_npu.override.common.swiglu_group.asc \
                              torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
                              torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc \
                              torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc}"
    else
        export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                                               torchtitan_npu.override.common.swiglu_group.asc \
                                               torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
                                               torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc \
                                               torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc \
                                               torchtitan_npu.override.deepseek_v4_1.mhc.asc_sinkhorn}"
    fi
fi

QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
)

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_8p_cpt_4k_a3.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    "$@" \
