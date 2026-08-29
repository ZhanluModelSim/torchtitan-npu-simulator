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

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-8}"
export NODE_IPS NGPU

# A5 defaults to block-FP8 quantized training. A trailing
# --extension.quantization.no-enable-quantized-training selects BF16 instead.
QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
)

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a3.sh" "${QUANTIZATION_ARGS[@]}" "$@"
