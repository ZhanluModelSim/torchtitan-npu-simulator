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
  : "${HF_ASSETS_PATH:?Set HF_ASSETS_PATH to the V4.1 tokenizer assets}"
  export MODULE=tests.integration_tests.engram_cp CONFIG=engram_cp
  export TORCHTITAN_ENGRAM_TABLE_ROWS=100000
  export LOG_RANK=0
  # Use the BF16 indexer for this opt-in test; quantized indexer selection
  # is configured separately through TorchAO.
  export CLI_OVERRIDES='torchtitan_npu.override.common.rms_norm.asc torchtitan_npu.override.common.rope.asc_complex torchtitan_npu.override.deepseek_v4_1.mhc.asc_hc_post torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc'
  case "${ENGRAM_CP_BACKEND:-torch}" in
    torch) ;;
    fp32) CLI_OVERRIDES+=' torchtitan_npu.override.deepseek_v4_1.engram.host_offload={"num_max_tokens_per_rank":12288}' ;;
    mxfp8) CLI_OVERRIDES+=' torchtitan_npu.override.deepseek_v4_1.engram.host_offload_mxfp8={"num_max_tokens_per_rank":12288}' ;;
    *) echo 'ENGRAM_CP_BACKEND must be torch, fp32 or mxfp8' >&2; exit 2 ;;
  esac
  # Keep DP, EP, samples and global batch fixed. CP alone changes the world
  # size from two to four, so the paired runs see the same packed documents.
  for cp in 1 2; do
    export NGPU=$((2 * cp))
    bash examples/deepseek_v4_1/debug/deepseek_v4_1_stripped_1p_a5.sh \
      --extension.quantization.no-enable-quantized-training \
      --extension.quantization.li-quantization None \
      --parallelism.spmd-backend spmd_types \
      --parallelism.data-parallel-shard-degree 2 \
      --parallelism.expert-parallel-degree 2 \
      --parallelism.context-parallel-degree "$cp" \
      --parallelism.context-parallel-load-balancer None \
      --dataloader.per-doc-alignment 4 \
      --training.global-batch-size 2 --training.seq-len 512 --training.steps 3 \
      --training.no-enable-cpu-offload --training.disable-cuda-graphs --optimizer.name AdamW \
      --debug.seed 42 --debug.deterministic \
      --dump-folder "$RUN_DIR/cp$cp" --metrics.enable-tensorboard \
      --metrics.save-tb-folder tb 2>&1 | tee "$RUN_DIR/cp$cp.log"
  done
  python -m tests.integration_tests.engram_cp "$RUN_DIR" | tee "$RUN_DIR/summary.log"
)
