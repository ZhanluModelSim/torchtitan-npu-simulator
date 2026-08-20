#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

# E2E perf benchmark for DSV4 Flash single-node HiF8 QAT. See
# ./config_registry.py (debug_deepseek_v4_flash_single_node_hif8_qat) for the
# quantization rules.
#
# Override priority (low to high): env vars below < user CLI args.
#   NGPU                - forwarded to scripts/run_train.sh → torchrun --nproc_per_node (default 8)
#   DATASET             - torchtitan dataset registry key (default c4_test)
#   DATASET_PATH        - absolute path to the dataset folder (default ${REPO_ROOT}/tests/assets/c4_test)
#   HF_ASSETS_PATH      - tokenizer folder (default /DeepSeek-V4-Flash)
#   OUTPUT_FOLDER       - root for outputs (default ${SCRIPT_DIR}/outputs/<timestamp>)
#   LOG_FILE            - tee'd log path (default ${OUTPUT_FOLDER}/terminal_<timestamp>.log)
#   WEIGHTS_PATH        - checkpoint folder to load; empty = random init (default empty)
#   ENABLE_UPDATE       - "true" keeps the config-registry default lr; anything else sets lr=0 (default false)
#   ENABLE_DETERMINISM  - "true" enables --debug.{seed 42,deterministic,no_deterministic_warn_only} (default false)

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

DATASET="${DATASET:-c4_test}"
DATASET_PATH="${DATASET_PATH:-${REPO_ROOT}/tests/assets/c4_test}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-/DeepSeek-V4-Flash}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-${SCRIPT_DIR}/outputs/${TIMESTAMP}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_FOLDER}/terminal_${TIMESTAMP}.log}"
WEIGHTS_PATH="${WEIGHTS_PATH:-}"
ENABLE_UPDATE="${ENABLE_UPDATE:-false}"
ENABLE_DETERMINISM="${ENABLE_DETERMINISM:-false}"

EXTRA_ARGS=(
  # Paths
  --hf_assets_path "${HF_ASSETS_PATH}"
  --dump_folder "${OUTPUT_FOLDER}"
  # Checkpoint / compile
  --checkpoint.no_enable
  --compile.enable
  # Training
  --training.local_batch_size 1
  --training.global_batch_size 64
  --training.steps 10
  --lr_scheduler.warmup_steps 5
  # Profiling
  --profiling.enable_profiling
  --profiling.no_enable_online_parse
  --profiling.profile_ranks 0
  --profiling.profile_step_start 8
  --profiling.profile_step_end 9
  --profiling.profile_record_shapes
  --profiling.no_profile_with_memory
  --profiling.no_profile_with_stack
)

if [ -n "${WEIGHTS_PATH}" ]; then
  EXTRA_ARGS+=(
    --checkpoint.enable
    --checkpoint.initial_load_path "${WEIGHTS_PATH}"
    --checkpoint.load_only
  )
fi

if [ "${ENABLE_UPDATE}" != "true" ]; then
  EXTRA_ARGS+=(--optimizer.lr 0)
fi

if [ "${ENABLE_DETERMINISM}" = "true" ]; then
  EXTRA_ARGS+=(
    --debug.seed 42
    --debug.deterministic
    --debug.no_deterministic_warn_only
  )
fi

# User overrides must precede the dataloader:config subcommand — tyro applies
# flags to the directly preceding subcommand.
EXTRA_ARGS+=("$@")

EXTRA_ARGS+=(
  dataloader:config
  --dataloader.dataset "${DATASET}"
  --dataloader.dataset_path "${DATASET_PATH}"
)

INVOKE_DIR="$(pwd)"
cd "${SCRIPT_DIR}"
mkdir -p "${OUTPUT_FOLDER}"

# Resolve the final compile state for the echo block (last occurrence wins).
COMPILE_RESOLVED=true
for arg in "$@"; do
  case "$arg" in
    --compile.enable)         COMPILE_RESOLVED=true  ;;
    --compile.no_enable)     COMPILE_RESOLVED=false ;;
    --compile.no-enable)     COMPILE_RESOLVED=false ;;
  esac
done

{
  echo "==== Benchmark env vars ===="
  echo "Script Invoked From    = ${INVOKE_DIR}"
  echo "Training Launched From = $(pwd)"
  echo "Training Launched At   = ${TIMESTAMP}"
  echo "NGPU                   = ${NGPU:-8}"
  echo "MODULE                 = ${MODULE:-config_registry}"
  echo "CONFIG                 = ${CONFIG:-debug_deepseek_v4_flash_single_node_hif8_qat}"
  echo "DATASET                = ${DATASET}"
  echo "DATASET_PATH           = ${DATASET_PATH}"
  echo "HF_ASSETS_PATH         = ${HF_ASSETS_PATH}"
  echo "OUTPUT_FOLDER          = ${OUTPUT_FOLDER}"
  echo "LOG_FILE               = ${LOG_FILE}"
  echo "WEIGHTS_PATH           = ${WEIGHTS_PATH:-<empty (random init)>}"
  echo "ENABLE_UPDATE          = ${ENABLE_UPDATE}"
  echo "ENABLE_DETERMINISM     = ${ENABLE_DETERMINISM}"
  echo "REPO_ROOT              = ${REPO_ROOT}"
  echo "SCRIPT_DIR             = ${SCRIPT_DIR}"
  echo "PYTHONPATH             = ${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
  echo "ASCEND_LAUNCH_BLOCKING = ${ASCEND_LAUNCH_BLOCKING:-<unset (async launches)>}"
  echo "Compile Enabled        = ${COMPILE_RESOLVED}"
  echo "============================"

  # PYTHONPATH prepends REPO_ROOT so workers import this repo's torchtitan_npu.
  PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  NGPU="${NGPU:-8}" \
  MODULE="${MODULE:-config_registry}" \
  CONFIG="${CONFIG:-debug_deepseek_v4_flash_single_node_hif8_qat}" \
  bash "${REPO_ROOT}/scripts/run_train.sh" \
    "${EXTRA_ARGS[@]}"
} 2>&1 | tee "${LOG_FILE}"
