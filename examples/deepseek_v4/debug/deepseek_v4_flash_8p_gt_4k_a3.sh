#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# GraphTrainer 变体:复用 deepseek_v4_flash_8p_cpt_4k_a3.sh 的参数定义
# (sed 截去其执行段),差异经 CLI 后值覆盖;sed 锚点 `MODULE="${MODULE}" \`
# 依赖参考脚本执行段起始行格式,参考脚本改动时需同步。

set -euo pipefail

# graph_trainer 工厂:编译配置 aot_fx_trace + 禁用 cudagraph_pass 内置
export CONFIG="graph_trainer_deepseek_v4_flash_43layers_16experts"

REF="$(dirname "$0")/deepseek_v4_flash_8p_cpt_4k_a3.sh"
eval "$(sed '/^MODULE="\${MODULE}" \\$/,$d' "${REF}")"

MODULE="${MODULE}" \
CONFIG="${CONFIG}" \
NGPU="${NGPU}" \
LOG_PREFIX="${LOG_PREFIX:-${CONFIG}}" \
bash scripts/run_train.sh \
    $HF_ASSETS_ARGS \
    $DATALOADER_ARGS \
    $PARALLELISM_ARGS \
    $TRAINING_ARGS \
    $DEBUG_ARGS \
    $OPTIMIZER_ARGS \
    $PROFILER_ARGS \
    $COMM_ARGS \
    $CHECKPOINT_ARGS \
    --parallelism.spmd-backend default \
    --training.global-batch-size 8 \
    --override.imports "${NPU_OPS_OVERRIDES[@]}" $OPTIMIZER_OVERRIDES \
    "$@"
