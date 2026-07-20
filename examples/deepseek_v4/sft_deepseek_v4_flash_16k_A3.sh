#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# Multi-node settings are passed through to scripts/run_train_multinodes.sh.
# Example for multi-node TAU SFT. Run the same command on every node:
#   HF_ASSETS_PATH=/data/models/DeepSeek-V4-Flash-bf16 \
#   CHECKPOINT_INITIAL_LOAD_PATH=/data/models/DeepSeek-V4-Flash-bf16 \
#   DATASET_PATH=/data/dataset/tau_historical_sft \
#   DATA_FILES=train-00000-of-00001.parquet \
#   DATASET_CONFIG_NAME=default \
#   CHAT_PROCESSOR=torchtitan_npu.hf_datasets.chat_processors.process_tau_sample \
#   NODE_IPS="10.90.22.43,10.90.22.44" Network_Interface=enp189s0f0 \
#   NGPU=16 \
#   bash examples/deepseek_v4/sft_deepseek_v4_flash_16k_A3.sh

HF_ASSETS_PATH="${HF_ASSETS_PATH:-/data/models/DeepSeek-V4-Flash-bf16}"
# Override dataloader and encoder defaults with environment variables.
DATASET_PATH="${DATASET_PATH:-./tests/assets/tau_historical_sft}"
DATA_FILES="${DATA_FILES:-}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-default}"
CHAT_PROCESSOR="${CHAT_PROCESSOR:-torchtitan_npu.hf_datasets.chat_processors.process_tau_sample}"
CHECKPOINT_INITIAL_LOAD_PATH="${CHECKPOINT_INITIAL_LOAD_PATH:-/data/models/DeepSeek-V4-Flash-bf16}"
ENCODING_MODULE_PATH="${ENCODING_MODULE_PATH:-${HF_ASSETS_PATH}/encoding/encoding_dsv4.py}"

EXTRA_ARGS=(
  # Trainer overrides
  --hf_assets_path "${HF_ASSETS_PATH}"
  --parallelism.data_parallel_shard_degree -1
  --parallelism.expert_parallel_degree 64
  --parallelism.context_parallel_degree 4
  --training.global_batch_size -1
  --training.seq_len 16384
  # --checkpoint.no_enable # debug
  --checkpoint.initial_load_path "${CHECKPOINT_INITIAL_LOAD_PATH}"
  --checkpoint.no_load_only
  --checkpoint.interval 500
  --checkpoint.export_dtype bfloat16
  --training.steps 100
  --lr_scheduler.warmup_steps 20
  # User top-level Trainer overrides
  "$@"

  ################## Top-level CLI overrides end; following subcommands cannot be overridden. ##################
  # DataLoader subcommand
  dataloader:chat_data_loader_config
  --dataloader.dataset_path "${DATASET_PATH}"
  --dataloader.chat_processor "${CHAT_PROCESSOR}"
  --dataloader.data_files "${DATA_FILES}"
  --dataloader.dataset_config_name "${DATASET_CONFIG_NAME}"
  # ChatEncoder subcommand
  dataloader.chat_encoder:dsv4_encoder_config
  --dataloader.chat_encoder.encoding_module_path "${ENCODING_MODULE_PATH}"
)

MODULE="${MODULE:-torchtitan_npu.models.deepseek_v4}" \
CONFIG="${CONFIG:-deepseek_v4_flash_4k_128npus}" \
NGPU="${NGPU:-16}" \
bash scripts/run_train_multinodes.sh \
  "${EXTRA_ARGS[@]}"
