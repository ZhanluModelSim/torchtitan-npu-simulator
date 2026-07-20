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
  --hf_assets_path "${HF_ASSETS_PATH}"
  --dataloader.dataset "${DATASET}"
  # --checkpoint.no_enable # debug
  --checkpoint.initial_load_path "${CHECKPOINT_INITIAL_LOAD_PATH}"
  # --profiling.enable_profiling
  --profiling.no_enable_online_parse
  --profiling.profile_ranks 0
  --profiling.profile_step_start 6
  --profiling.profile_step_end 7
  --profiling.profile_record_shapes
  --profiling.profile_with_memory
  --profiling.no_enable_memory_snapshot
  --profiling.save_memory_snapshot_folder memory_snapshot
  # --training.steps 20 # debug
  # --lr_scheduler.warmup_steps 4 # debug
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-deepseek_v4_pro_4k_384npus}" \
NGPU="${NGPU:-16}" \
bash scripts/run_train_multinodes.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
