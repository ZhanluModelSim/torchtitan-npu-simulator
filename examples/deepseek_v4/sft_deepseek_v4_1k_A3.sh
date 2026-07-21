#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/deepseek-v4-mini-1B-init}"
CHECKPOINT_INITIAL_LOAD_PATH="${CHECKPOINT_INITIAL_LOAD_PATH:-/data/models/deepseek-v4-mini-1B-init}"
DATA_FILES="${DATA_FILES:-yelp_review_full/train-00000-of-00001.parquet}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
CHECKPOINT_FOLDER="${CHECKPOINT_FOLDER:-checkpoint}"

if [[ ! -f "${HF_ASSETS_PATH}/tokenizer.json" ]]; then
  echo "Missing tokenizer.json under HF_ASSETS_PATH: ${HF_ASSETS_PATH}" >&2
  exit 1
fi

EXTRA_ARGS=(
  --hf_assets_path "${HF_ASSETS_PATH}"
  --checkpoint.initial_load_path "${CHECKPOINT_INITIAL_LOAD_PATH}"
  --checkpoint.enable
  --checkpoint.initial_load_in_hf
  --checkpoint.initial_load_model_only
  --checkpoint.no_load_only
  --checkpoint.folder "${CHECKPOINT_FOLDER}"
  --checkpoint.interval 500
  --checkpoint.last_save_model_only
  --checkpoint.export_dtype bfloat16
  --training.local_batch_size 8
  --training.global_batch_size 8
  --training.seq_len 1152
  --training.steps 1000
  --lr_scheduler.warmup_steps 60
  # User top-level Trainer overrides
  "$@"

  ################## Top-level CLI overrides end; following subcommands cannot be overridden. ##################
  # DataLoader subcommand
  dataloader:chat_data_loader_config
  --dataloader.dataset_path parquet
  --dataloader.chat_processor torchtitan_npu.hf_datasets.chat_processors.process_yelp_sample
  --dataloader.data_files "${DATA_FILES}"
  --dataloader.dataset_split "${DATASET_SPLIT}"
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-debug_deepseek_v4_single_node_1b}" \
NGPU="${NGPU:-1}" \
bash scripts/run_train_multinodes.sh \
  "${EXTRA_ARGS[@]}"
