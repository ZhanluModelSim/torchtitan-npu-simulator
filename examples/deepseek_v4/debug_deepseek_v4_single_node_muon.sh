#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

DATASET="${DATASET:-c4_test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/deepseek_v4_tokenizer_only}"

EXTRA_ARGS=(
  --hf_assets_path "${HF_ASSETS_PATH}"
  --dataloader.dataset "${DATASET}"
  # --profiling.enable_profiling
  --profiling.no_enable_online_parse
  --profiling.profile_ranks 0
  --profiling.profile_step_start 6
  --profiling.profile_step_end 7
  --profiling.profile_record_shapes
  --profiling.profile_with_memory
  --optimizer.name Muon
  --optimizer.lr 2.2e-4
  --optimizer.weight_decay 0.1
  --optimizer.muon_momentum 0.95
  --optimizer.muon_enable_nesterov
  --optimizer.muon_ns_steps 10
  --optimizer.muon_adjust_lr_fn match_rms_adamw
  --optimizer.muon_hybrid_ns
  --optimizer.swap_merge_buckets 4
  # --training.steps 20 # debug
  # --lr_scheduler.warmup_steps 4 # debug
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-debug_deepseek_v4_flash_single_node}" \
NGPU="${NGPU:-8}" \
bash scripts/run_train.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
