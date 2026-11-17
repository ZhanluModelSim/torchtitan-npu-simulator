#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4 Flash CPT launcher. Keep the shared
# model, training, and parallelism defaults in the A3 script; this entrypoint
# only supplies A5-specific runtime, quantization, and recompute settings.
# Append CLI arguments to override the defaults below:
#   ./examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh --training.steps 5

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Use the command line `npu-smi info -t topo` to query the CPU affinity of the
# NPU cards for configuration.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

# A5-only fused ops; USE_GOLDEN=1 keeps the pure reference list.
if [[ "${USE_GOLDEN:-0}" != "1" ]]; then
    export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                                           torchtitan_npu.override.deepseek_v4.compressor.asc \
                                           torchtitan_npu.override.deepseek_v4.compressor.asc_metadata \
                                           torchtitan_npu.override.common.swiglu_group.asc \
                                           torchtitan_npu.override.common.swiglu_group.asc_shared_experts}"
fi

# A5 defaults to block-FP8 quantized training. Append
# --extension.quantization.no-enable-quantized-training for BF16 training.
#
# fsdp-prequantize leaves fsdp-prequantize-fqns unset, so the recipe default
# whitelist (all Block FP8 projections) applies. The runtime guard requires
# each whitelisted weight's dp-sharded dim0 to stay 64-aligned: at this
# script's DP_SHARD=8 every default entry is aligned (wkv dim0=512 -> 64
# rows/rank). At degrees where an entry breaks alignment (e.g. DP_SHARD=16:
# wkv shards to 32 rows) training fails fast at startup with a fix hint;
# narrow the whitelist by appending the default patterns minus the offending
# one, e.g. excluding wkv (space-separated, matching the 4K entry below):
#   --extension.quantization.fsdp-prequantize-fqns .attention.wq_a .attention.wq_b .attention.wo_a .attention.wo_b .attention.indexer.wq_b .moe.shared_experts.w1 .moe.shared_experts.w2 .moe.shared_experts.w3 .moe.routed_experts.inner_experts
# A user whitelist fails fast on any pattern absent from the model, so drop
# the .moe.shared_experts.* entries on a model without shared experts.
QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.enable-fsdp-prequantize
    --extension.quantization.li-quantization fp8
    --extension.quantization.kv-norm-quantization.format mxfp8
    --extension.quantization.kv-norm-quantization.fqns .attention.kv_norm,.attention.compressor.norm
    --extension.quantization.kv-norm-quantization.block-size 64
)

# Disable swap optimizer for better performance
export OPTIMIZER_OVERRIDES=""

# Tyro treats activation-checkpoint:selective as a subcommand, so it must be
# the final token after all regular options and override targets.
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_8p_cpt_4k_a3.sh" \
    --training.local-batch-size 2 \
    "${QUANTIZATION_ARGS[@]}" \
    --debug.moe-force-load-balance \
    "$@" \
    activation-checkpoint:selective
