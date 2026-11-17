#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# LoRA wrapper around the common DeepSeek-V4 Flash CPT launcher. Keep model,
# training, and parallelism differences as CLI overrides so user-supplied
# arguments remain the final source of truth.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG="${CONFIG:-deepseek_v4_flash_lora}"

# LoRA uses periodic native adapter checkpoints and final PEFT export.
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a3.sh" \
    --checkpoint.no-load-only \
    --checkpoint.save-training-state \
    "$@"
