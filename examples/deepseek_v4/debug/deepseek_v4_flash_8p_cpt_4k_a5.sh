#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4 Flash CPT launcher. Keep the shared
# model and parallelism defaults in the A3 script; this entrypoint supplies
# A5-specific runtime, quantization, batch, and recompute settings.
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
                                           torchtitan_npu.override.common.swiglu_group.asc \
                                           torchtitan_npu.override.common.swiglu_group.asc_shared_experts}"
fi

# Disable swap optimizer for better performance
export OPTIMIZER_OVERRIDES=""

# Activation Checkpoint strategy
# Allowed values: selective | full | memory-budget | none (memory-budget requires compile).
AC_MODE="${AC_MODE:-selective}"

# A5 defaults to block-FP8 quantized training. Append
# --extension.quantization.no-enable-quantized-training for BF16 training.
#
# fsdp-prequantize keeps the explicit whitelist below; weights are excluded
# for one of two reasons:
#   1) the dp-sharded dim0 breaks the 64-alignment the path requires (the
#      runtime guard fails fast at startup);
#   2) the FSDP mesh is size 1, so no all-gather runs: pre-quantizing saves no
#      communication and only keeps an extra quantized copy of the weight
#      resident. Here efsdp = dp_shard * cp * tp / ep = 1 (DP_SHARD=8, EP=8),
#      which excludes the routed experts; add
#      .moe.routed_experts.inner_experts back when an override makes efsdp > 1.
#
# At DP_SHARD=8 every listed entry is aligned (wkv dim0=512 -> 64 rows/rank);
# at DP_SHARD=16 wkv shards to 32 rows, so drop the offending pattern from the
# comma-joined argument (one argument, no spaces), e.g. excluding wkv:
#   --extension.quantization.fsdp-prequantize-fqns \
#   .attention.wq_a,.attention.wq_b,.attention.wo_a,.attention.wo_b,\
#   .attention.indexer.wq_b,.moe.shared_experts.w1,.moe.shared_experts.w2,\
#   .moe.shared_experts.w3,.moe.routed_experts.inner_experts
# A user whitelist fails fast on any pattern absent from the model, so drop
# the .moe.shared_experts.* entries on a model without shared experts.
PREQUANTIZE_FQNS=(
    .attention.wq_a
    .attention.wq_b
    .attention.wkv
    .attention.wo_a
    .attention.wo_b
    .attention.indexer.wq_b
    .moe.shared_experts.w1
    .moe.shared_experts.w2
    .moe.shared_experts.w3
)
# The CLI takes the whitelist as one comma-joined argument.
PREQUANTIZE_FQNS_JOINED="$(IFS=,; printf '%s' "${PREQUANTIZE_FQNS[*]}")"

QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.enable-fsdp-prequantize
    --extension.quantization.fsdp-prequantize-fqns "${PREQUANTIZE_FQNS_JOINED}"
    --extension.quantization.li-quantization fp8
    --extension.quantization.kv-norm-quantization.format mxfp8
    --extension.quantization.kv-norm-quantization.fqns .attention.kv_norm,.attention.compressor.norm
    --extension.quantization.kv-norm-quantization.block-size 64
)


# To check performance under load-balanced MoE routing, uncomment the following block.
#
#
# # Drop mm and bmm from the saved op list using update_save_ops override to avoid OOM
# SELECTIVE_AC_SAVE_OP_EDITS=(
#     -torch.ops.aten.bmm.default
#     -torch.ops.aten.mm.default
# )
# if [[ "${AC_MODE}" == "selective" ]]; then
#     SELECTIVE_AC_OVERRIDE="torchtitan_npu.override.common.activation_checkpoint.update_save_ops"
#     if [[ ${#SELECTIVE_AC_SAVE_OP_EDITS[@]} -gt 0 ]]; then
#         SELECTIVE_AC_SAVE_OP_EDITS_JOINED="$(IFS=,; printf '%s' "${SELECTIVE_AC_SAVE_OP_EDITS[*]}")"
#         SELECTIVE_AC_OVERRIDE+="={\"edits\":\"${SELECTIVE_AC_SAVE_OP_EDITS_JOINED}\"}"
#     fi
#     CLI_OVERRIDES="${CLI_OVERRIDES:-} ${SELECTIVE_AC_OVERRIDE}"
#     export CLI_OVERRIDES
# fi
#
# export CLI_OVERRIDES="${CLI_OVERRIDES:-} --debug.moe-force-load-balance --training.local-batch-size 2"


# Tyro treats activation-checkpoint:<mode> (AC_MODE) as a subcommand: it must
# follow all regular options and override targets.
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_8p_cpt_4k_a3.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    --hf-assets-path "/path/to/tokenizer/folder" \
    "$@" \
    activation-checkpoint:"${AC_MODE}"
