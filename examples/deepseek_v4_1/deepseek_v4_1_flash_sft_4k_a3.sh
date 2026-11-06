#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Vision-language SFT wrapper around the A3 CPT launcher in this directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export DSV41_SFT_DATASET="${DSV41_SFT_DATASET:-${DATASET_PATH:-/path/to/train.jsonl}}"
export CONFIG="${CONFIG:-deepseek_v4_1_flash_sft_compiled}"

exec bash "${SCRIPT_DIR}/deepseek_v4_1_flash_cpt_4k_a3.sh" "$@"
