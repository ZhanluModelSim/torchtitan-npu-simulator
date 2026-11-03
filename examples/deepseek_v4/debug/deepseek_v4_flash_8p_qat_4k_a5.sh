#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MXFP4 QAT entrypoint: reuse the CPT A5 launcher and append the QAT options
# below.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# QAT-specific quantization configurations
# The arguments will be appended to or override the options listed in
# the QUANTIZATION_ARGS of the pretraining script.
QUANTIZATION_ARGS=(
    --extension.quantization.enable-mxfp4-qat
    --extension.quantization.li-quantization mxfp4
)

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_8p_cpt_4k_a5.sh" \
    "${QUANTIZATION_ARGS[@]}" \
    "$@"
