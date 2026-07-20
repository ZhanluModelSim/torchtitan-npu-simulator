#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

DATASET="${DATASET:-c4_test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/deepseek_v4_tokenizer_only}"

EXTRA_ARGS=(
  # Trainer overrides
  --hf_assets_path "${HF_ASSETS_PATH}"
  # --profiling.enable_profiling
  --profiling.no_enable_online_parse
  --profiling.profile_ranks 0
  --profiling.profile_step_start 6
  --profiling.profile_step_end 7
  --profiling.profile_record_shapes
  --profiling.profile_with_memory
  # --training.steps 20 # debug
  # --lr_scheduler.warmup_steps 4 # debug
  # User top-level Trainer overrides
  "$@"

  ################## Top-level CLI overrides end; following subcommands cannot be overridden. ##################
  # DataLoader subcommand
  dataloader:config
  --dataloader.dataset "${DATASET}"
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-debug_deepseek_v4_flash_single_node}" \
NGPU="${NGPU:-8}" \
bash scripts/run_train.sh \
  "${EXTRA_ARGS[@]}"
