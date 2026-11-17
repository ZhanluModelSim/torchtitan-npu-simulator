#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# QAT wrapper around the DeepSeek-V4 Flash CPT A5 launcher. Inherit the A5
# environment and training defaults, and override only quantization arguments.


set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-8}"
export NODE_IPS NGPU

#Introduction of relevant parameters, please refer to docs/user-guides/quickstart.md.
QUANTIZATION_ARGS=(
    --extension.quantization.enable-quantized-training
    --extension.quantization.recipe all_block_fp8
    --extension.quantization.enable-mxfp4-qat
    --extension.quantization.li-quantization mxfp4
)

# Disable swap optimizer for better performance
export OPTIMIZER_OVERRIDES=""

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a5.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    "$@"
