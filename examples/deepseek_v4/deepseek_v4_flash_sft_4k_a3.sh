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
SFT_OVERRIDE="${SFT_OVERRIDE:-torchtitan_npu.override.deepseek_v4.chat_dataloader.sft}"
DATASET_PATH="${DATASET_PATH:-/path/to/train.jsonl}"

exec bash "${CPT_SCRIPT}" \
    "${SFT_OVERRIDE}" \
    --dataloader.dataset json \
    --dataloader.dataset-path "${DATASET_PATH}" \
    --checkpoint.enable \
    --checkpoint.no-load-only \
    "$@"
