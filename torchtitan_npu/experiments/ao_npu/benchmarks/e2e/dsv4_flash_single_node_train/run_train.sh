#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# E2E perf benchmark for DSV4 Flash single-node QAT.
# Recipe-controlled via env vars (tyro cannot route converter-specific flags):
#   RECIPE                    - all_mxfp8, mix, all_block_fp8 (default mix)
#   ENABLE_QUANTIZED_TRAINING - true/false (default true)
#   ENABLE_MXFP4_QAT          - true/false (default true)
#   DST_TYPE_MAX              - MXFP4 QAT weight fake-quantize target dtype max (default 0.0 = auto)
# Other overrides via $@:
#   BASE_MODEL  - flash or 1b (default flash)
#   NGPU        - gpus per node (flash=8, 1b=4)
#   ENABLE_DETERMINISM - true adds --debug.seed 42 --debug.deterministic

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

BASE_MODEL="${BASE_MODEL:-flash}"
RECIPE="${RECIPE:-mix}"
ENABLE_QUANTIZED_TRAINING="${ENABLE_QUANTIZED_TRAINING:-true}"
ENABLE_MXFP4_QAT="${ENABLE_MXFP4_QAT:-true}"
DST_TYPE_MAX="${DST_TYPE_MAX:-0.0}"
DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-}"
ENABLE_DETERMINISM="${ENABLE_DETERMINISM:-false}"
DATA_FILES="${DATA_FILES:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_CONFIG_NAME="${DATASET_CONFIG_NAME:-default}"
CHAT_PROCESSOR="${CHAT_PROCESSOR:-torchtitan_npu.hf_datasets.chat_processors.process_yelp_sample}"

case "${BASE_MODEL}" in
  flash)
    NGPU="${NGPU:-8}"
    DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/tests/assets/${DATASET}}" ;;
  1b)
    NGPU="${NGPU:-4}"
    DATA_FILES="${DATA_FILES:-/path/to/your/yelp_review_full/train-00000-of-00001.parquet}" ;;
  *)
    echo "ERROR: BASE_MODEL must be one of (flash, 1b), got '${BASE_MODEL}'"; exit 1 ;;
esac

OUTPUT_FOLDER="${OUTPUT_FOLDER:-${SCRIPT_DIR}/outputs/${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_FOLDER}/terminal_${TIMESTAMP}.log}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/DeepSeek-V4-Flash}"

EXTRA_ARGS=(
  --hf_assets_path "${HF_ASSETS_PATH}"
  --dump_folder "${OUTPUT_FOLDER}"
  --checkpoint.no_enable
  --compile.enable
  --training.local_batch_size 1
  --training.global_batch_size 64
  --training.steps 10
  --lr_scheduler.warmup_steps 5
  --optimizer.lr 0
  --profiling.enable_profiling
  --profiling.no_enable_online_parse
  --profiling.profile_ranks 0
  --profiling.profile_step_start 8
  --profiling.profile_step_end 9
  --profiling.profile_record_shapes
  --profiling.no_profile_with_memory
  --profiling.no_profile_with_stack
  # User overrides
  "$@"

  ################## Top-level CLI overrides end; following subcommands cannot be overridden. ##################
)

if [ "${ENABLE_DETERMINISM}" = "true" ]; then
  EXTRA_ARGS+=(
    --debug.seed 42
    --debug.deterministic
    --debug.no_deterministic_warn_only
  )
fi

if [ "${BASE_MODEL}" = "1b" ]; then
  EXTRA_ARGS+=(
    dataloader:chat_data_loader_config
    --dataloader.dataset_path parquet
    --dataloader.chat_processor "${CHAT_PROCESSOR}"
    --dataloader.data_files "${DATA_FILES}"
    --dataloader.dataset_split "${DATASET_SPLIT}"
    --dataloader.dataset_config_name "${DATASET_CONFIG_NAME}"
  )
else
  EXTRA_ARGS+=(
    dataloader:config
    --dataloader.dataset "${DATASET}"
    --dataloader.dataset_path "${DATASET_PATH}"
  )
fi

echo "=== DSV4 Flash QAT Benchmark ==="
echo "BASE_MODEL = ${BASE_MODEL}  NGPU = ${NGPU}  RECIPE = ${RECIPE}"
echo "OUTPUT     = ${OUTPUT_FOLDER}"
echo "================================"

cd "${SCRIPT_DIR}"
mkdir -p "${OUTPUT_FOLDER}"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
NGPU="${NGPU}" \
MODULE="${MODULE:-config_registry}" \
CONFIG="debug_deepseek_v4_flash_single_node_train" \
BASE_MODEL="${BASE_MODEL}" \
RECIPE="${RECIPE}" \
ENABLE_QUANTIZED_TRAINING="${ENABLE_QUANTIZED_TRAINING}" \
ENABLE_MXFP4_QAT="${ENABLE_MXFP4_QAT}" \
DST_TYPE_MAX="${DST_TYPE_MAX}" \
HCCL_NPU_SOCKET_PORT_RANGE="16667-16677" \
bash "${REPO_ROOT}/scripts/run_train.sh" \
  "${EXTRA_ARGS[@]}" \
2>&1 | tee "${LOG_FILE}"
