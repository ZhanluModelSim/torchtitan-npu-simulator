#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# LoRA wrapper around the shared V4.1 CPT launcher; user arguments take precedence.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG="${CONFIG:-deepseek_v4_1_flash_lora}"
# Match the documented reference-operator LoRA path by default.
export USE_GOLDEN="${USE_GOLDEN:-1}"
export OPTIMIZER_OVERRIDES="${OPTIMIZER_OVERRIDES-}"
if [[ "${CONFIG}" == "deepseek_v4_1_flash_text_lora" ]]; then
    export DATASET="${DATASET:-c4_test}"
fi

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_cpt_4k_a3.sh" \
    --compile.no-enable \
    --checkpoint.no-load-only \
    --checkpoint.save-training-state \
    "$@"
