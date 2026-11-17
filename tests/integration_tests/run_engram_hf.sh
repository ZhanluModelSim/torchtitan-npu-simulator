#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

(
  set -euo pipefail
  cd "$(dirname -- "${BASH_SOURCE[0]}")/../.."
  if [[ "$#" != 2 || "$1" != --dump_folder ]]; then
    echo "Usage: bash $0 --dump_folder OUTPUT" >&2
    exit 2
  fi
  mkdir -p "$2"
  RUN_DIR=$(cd "$2" && pwd)
  export ENGRAM_HF_RUN_DIR="$RUN_DIR"
  ep=4
  case "${ENGRAM_HF_JOINT_SHARDING:-0}" in
    0) joint=false ;;
    1) joint=true; ep=2 ;;
    *) echo "ENGRAM_HF_JOINT_SHARDING must be 0 or 1" >&2; exit 2 ;;
  esac
  export CLI_OVERRIDES="torchtitan_npu.override.deepseek_v4_1.engram.host_offload_mxfp8={\"num_max_tokens_per_rank\":3072,\"shard_over_efsdp\":$joint}"
  export NGPU=4 LOG_RANK=0,1,2,3
  export MODULE=torchtitan_npu.models.deepseek_v4_1
  export CONFIG=deepseek_v4_1_debugmodel_text
  export DATASET=c4_test DATASET_PATH="$PWD/tests/assets/c4_test"
  : "${HF_ASSETS_PATH:?Set HF_ASSETS_PATH to the V4.1 tokenizer assets}"
  export USE_GOLDEN=1 TRAIN_FILE=tests.integration_tests.engram_hf OPTIMIZER_OVERRIDES=''
  for mode in source native hf; do
    steps=3
    args=(--checkpoint.load-only --checkpoint.initial-load-model-only)
    case "$mode" in
      source)
        steps=2
        args=(--checkpoint.no-load-only --checkpoint.last-save-model-only
          --checkpoint.last-save-in-hf --checkpoint.export-dtype float32) ;;
      native) args+=(--checkpoint.initial-load-path "$RUN_DIR/source/checkpoint/step-2-native") ;;
      hf) args+=(--checkpoint.initial-load-path "$RUN_DIR/source/checkpoint/step-2" --checkpoint.initial-load-in-hf) ;;
    esac
    export CKPT_SAVE_LOAD_PATH="$RUN_DIR/$mode/checkpoint"
    bash examples/deepseek_v4_1/debug/deepseek_v4_1_flash_8p_cpt_4k_a5.sh \
      --extension.quantization.no-enable-quantized-training \
      --compile.no-enable \
      --optimizer.name AdamW --training.no-enable-cpu-offload \
      --parallelism.expert-parallel-degree "$ep" \
      --parallelism.data-parallel-shard-degree 4 \
      --parallelism.data-parallel-replicate-degree 1 \
      --training.global-batch-size 4 --training.seq-len 512 --training.steps "$steps" \
      --training.max-norm 1.0 --lr-scheduler.total-steps 20 \
      --debug.seed 42 --debug.deterministic \
      --dump-folder "$RUN_DIR/$mode" --metrics.enable-tensorboard \
      --checkpoint.enable --checkpoint.interval 100 \
      "${args[@]}" 2>&1 | tee "$RUN_DIR/$mode.log"
  done
  python -m tests.integration_tests.engram_hf --compare "$RUN_DIR" | tee "$RUN_DIR/summary.log"
  echo "Completed: $RUN_DIR"
)
