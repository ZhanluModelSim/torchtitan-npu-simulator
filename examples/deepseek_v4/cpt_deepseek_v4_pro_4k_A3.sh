#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

DATASET="${DATASET:-c4_test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/DeepSeek-V4-Pro-bf16}"
CHECKPOINT_INITIAL_LOAD_PATH="${CHECKPOINT_INITIAL_LOAD_PATH:-/data/models/DeepSeek-V4-Pro-bf16}"

EXTRA_ARGS=(
  --hf-assets-path "${HF_ASSETS_PATH}"
  --dataloader.dataset "${DATASET}"
  # --checkpoint.no-enable # debug
  --checkpoint.initial-load-path "${CHECKPOINT_INITIAL_LOAD_PATH}"
  # --profiling.enable-profiling
  --profiling.no-enable-online-parse
  --profiling.profile-ranks 0
  --profiling.profile-step-start 6
  --profiling.profile-step-end 7
  --profiling.profile-record-shapes
  --profiling.profile-with-memory
  --profiling.no-enable-memory-snapshot
  --profiling.save-memory-snapshot-folder memory_snapshot
  # --training.steps 20 # debug
  # --lr-scheduler.warmup-steps 4 # debug
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-deepseek_v4_pro_4k_384die}" \
NGPU="${NGPU:-16}" \
bash scripts/run_train_multinodes.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
