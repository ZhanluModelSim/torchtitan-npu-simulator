#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Vision-language SFT wrapper around the 8p A5 CPT launcher in this directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-/path/to/train.jsonl}"

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_8p_cpt_4k_a5.sh" \
    torchtitan_npu.override.deepseek_v4_1.vision_language_dataloader.sft \
    --dataloader.dataset-path "${DATASET_PATH}" \
    "$@"
