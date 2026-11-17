#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# A5 wrapper around the common DeepSeek-V4 Flash CPT launcher. Keep model,
# training, and parallelism differences as CLI overrides so user-supplied
# arguments remain the final source of truth.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17330}"
export ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-2147480}"
export HCCL_EVENT_TIMEOUT="${HCCL_EVENT_TIMEOUT:-2147480}"
# Disable the HCCL watchdog for long checkpoint loading on the A5 setup.
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
# Override this for the host topology reported by `npu-smi info -t topo`.
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

# A5-only fused ops; USE_GOLDEN=1 keeps the pure reference list.
if [[ "${USE_GOLDEN:-0}" != "1" ]]; then
    export CLI_OVERRIDES="${CLI_OVERRIDES:-torchtitan_npu.override.common.rope.asc_partial \
                                           torchtitan_npu.override.deepseek_v4.compressor.asc \
                                           torchtitan_npu.override.deepseek_v4.compressor.asc_metadata \
                                           torchtitan_npu.override.common.swiglu_group.asc \
                                           torchtitan_npu.override.common.swiglu_group.asc_shared_experts}"
fi

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-8}"
export NODE_IPS NGPU

# A5 defaults to block-FP8 quantized training. A trailing
# --extension.quantization.no-enable-quantized-training selects BF16 instead.
#
# The runtime guard requires each pre-quantized weight's dp-sharded dim0 to
# stay 64-aligned, and the base script runs DP_SHARD=128 (dim0 must be a
# multiple of 64*128=8192). For the default deepseek_v4_flash config that
# breaks the evenly-sharded dense projections -- wq_a (1024 -> 8 rows/rank),
# wkv (512 -> 4), wo_b (4096 -> 32) and shared_experts w1/w3 (2048 -> 16) and
# w2 (4096 -> 32) -- which fail fast at startup with a fix hint instead of
# silently falling back to BF16 communication. fsdp-prequantize-fqns below
# therefore narrows the whitelist to the projections that stay aligned: the
# per-head/per-group BlockShard projections (wq_b, wo_a, indexer.wq_b) and
# the expert-axis-sharded routed experts. At lower DP_SHARD degrees (e.g. 8,
# where every default entry is aligned) drop that line to restore the recipe
# default whitelist (all Block FP8 projections).
QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.enable-fsdp-prequantize
    --extension.quantization.fsdp-prequantize-fqns .attention.wq_b .attention.wo_a .attention.indexer.wq_b .moe.routed_experts.inner_experts
    --extension.quantization.li-quantization fp8
    --extension.quantization.kv-norm-quantization.format mxfp8
    --extension.quantization.kv-norm-quantization.fqns .attention.kv_norm,.attention.compressor.norm
    --extension.quantization.kv-norm-quantization.block-size 64
)

# Disable swap optimizer for better performance
export OPTIMIZER_OVERRIDES=""

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a3.sh" \
    --training.local-batch-size 2 \
    "${QUANTIZATION_ARGS[@]}" \
    "$@" \
    activation-checkpoint:selective
