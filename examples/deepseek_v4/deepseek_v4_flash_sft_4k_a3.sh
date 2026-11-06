#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Reuse the CPT launcher with an SFT JSONL or JSON file.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CPT_SCRIPT="${CPT_SCRIPT:-${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a3.sh}"
DATASET_PATH="${DATASET_PATH:-/path/to/train.jsonl}"

CONFIG="${CONFIG:-deepseek_v4_flash_sft_compiled}" \
DSV4_SFT_DATASET="${DATASET_PATH}" \
exec bash "${CPT_SCRIPT}" \
    --checkpointer.no-load-only \
    "$@"
