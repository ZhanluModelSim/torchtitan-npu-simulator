#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

(
  set -euo pipefail
  cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
  if [[ "$#" != 2 || "$1" != --output-dir ]]; then
    echo "Usage: bash $0 --output-dir OUTPUT" >&2
    exit 2
  fi
  mkdir -p "$2"
  RUN_DIR=$(cd "$2" && pwd)
  export ENGRAM_HF_RUN_DIR="$RUN_DIR"
  export ENGRAM_HF_JOINT_SHARDING="${ENGRAM_HF_JOINT_SHARDING:-0}"
  export ENGRAM_HF_BACKEND="${ENGRAM_HF_BACKEND:-host}"
  export ENGRAM_HF_EXPORT_DTYPE="${ENGRAM_HF_EXPORT_DTYPE:-bfloat16}"
  export HF_ASSETS_PATH="${HF_ASSETS_PATH:-tests/assets/deepseek_v3}"
  export NGPU=4 LOG_RANK=0,1,2,3
  export MODULE=tests.integration_tests.engram_hf CONFIG=engram_hf
  PYTHON_BIN="${PYTHON_BIN:-python3}"

  for mode in source native hf; do
    export ENGRAM_HF_MODE="$mode"
    bash scripts/run_train.sh --output-dir "$RUN_DIR/$mode" 2>&1 | tee "$RUN_DIR/$mode.log"
  done
  "$PYTHON_BIN" -m tests.integration_tests.engram_hf --compare "$RUN_DIR" | tee "$RUN_DIR/summary.log"
  echo "Completed: $RUN_DIR"
)
