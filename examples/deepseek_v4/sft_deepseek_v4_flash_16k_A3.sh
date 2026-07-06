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
  --hf-assets-path "${HF_ASSETS_PATH}"
  --dataloader.dataset-path "${DATASET_PATH}"
  # --checkpoint.no-enable # debug
  --checkpoint.initial-load-path "${CHECKPOINT_INITIAL_LOAD_PATH}"
  # --profiling.enable-profiling
  --profiling.no-enable-online-parse
  --profiling.profile-ranks 0
  --profiling.profile-step-start 6
  --profiling.profile-step-end 7
  --profiling.profile-record-shapes
  --profiling.profile-with-memory
  dataloader.chat-encoder:dsv4-encoder-config
  --dataloader.chat-encoder.encoding-module-path "${ENCODING_MODULE_PATH}"
  # --training.steps 20 # debug
  # --lr-scheduler.warmup-steps 4 # debug
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-sft_deepseek_v4_flash_16k_128die_tau}" \
NGPU="${NGPU:-16}" \
bash scripts/run_train_multinodes.sh \
  "${EXTRA_ARGS[@]}" \
  "$@"
