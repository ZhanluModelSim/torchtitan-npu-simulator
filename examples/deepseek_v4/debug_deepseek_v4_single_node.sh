#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

DATASET="${DATASET:-c4_test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/deepseek_v4_tokenizer_only}"

EXTRA_ARGS=(
  --hf-assets-path "${HF_ASSETS_PATH}"
  --dataloader.dataset "${DATASET}"
  # --profiling.enable-profiling
  --profiling.no-enable-online-parse
  --profiling.profile-ranks 0
  --profiling.profile-step-start 6
  --profiling.profile-step-end 7
  --profiling.profile-record-shapes
  --profiling.profile-with-memory
  # --training.steps 20 # debug
  # --lr-scheduler.warmup-steps 4 # debug
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-debug_deepseek_v4_flash_single_node}" \
NGPU="${NGPU:-8}" \
bash scripts/run_train.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
