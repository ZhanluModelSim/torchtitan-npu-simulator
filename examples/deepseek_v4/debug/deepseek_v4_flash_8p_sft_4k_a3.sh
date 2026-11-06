#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# SFT wrapper around the 8p CPT launcher in this directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATASET_PATH="${DATASET_PATH:-/path/to/train.jsonl}"

CONFIG="${CONFIG:-deepseek_v4_flash_43layers_16experts_sft_compiled}" \
DSV4_SFT_DATASET="${DATASET_PATH}" \
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_8p_cpt_4k_a3.sh" \
    --checkpointer.no-load-only \
    "$@"
