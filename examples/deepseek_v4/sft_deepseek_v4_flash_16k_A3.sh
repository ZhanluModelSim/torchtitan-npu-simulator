#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/DeepSeek-V4-Flash-bf16}"
DATASET_PATH="${DATASET_PATH:-/data/dataset/tau-historical-sft/processed_clean}"
CHECKPOINT_INITIAL_LOAD_PATH="${CHECKPOINT_INITIAL_LOAD_PATH:-/data/models/DeepSeek-V4-Flash-bf16}"
ENCODING_MODULE_PATH="${ENCODING_MODULE_PATH:-${HF_ASSETS_PATH}/encoding/encoding_dsv4.py}"

EXTRA_ARGS=(
  --hf_assets_path "${HF_ASSETS_PATH}"
  --dataloader.dataset_path "${DATASET_PATH}"
  # --checkpoint.no_enable # debug
  --checkpoint.initial_load_path "${CHECKPOINT_INITIAL_LOAD_PATH}"
  # --profiling.enable_profiling
  --profiling.no_enable_online_parse
  --profiling.profile_ranks 0
  --profiling.profile_step_start 6
  --profiling.profile_step_end 7
  --profiling.profile_record_shapes
  --profiling.profile_with_memory
  dataloader.chat_encoder:dsv4_encoder_config
  --dataloader.chat_encoder.encoding_module_path "${ENCODING_MODULE_PATH}"
  # --training.steps 20 # debug
  # --lr_scheduler.warmup_steps 4 # debug
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-sft_deepseek_v4_flash_16k_128die_tau}" \
NGPU="${NGPU:-16}" \
bash scripts/run_train_multinodes.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
