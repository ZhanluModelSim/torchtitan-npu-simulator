#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run A3 GraphTrainer with sequence-based MoE EP overlap and NPU auto scheduling.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG="${CONFIG:-graph_trainer_deepseek_v4_flash_43layers_16experts}"
export TORCHINDUCTOR_USE_TORCH_PROFILER_BENCHMARKER=1
export TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER=0
export TORCH_NPU_USE_COMPATIBLE_IMPL=1

EP_OVERLAP_ARGS=(
    --compile.ep_overlap.enabled
    --compile.ep_overlap.chunk_dim seq
    --compile.ep_overlap.strategy graph
    --compile.ep_overlap.module_fqn "layers.*.moe"
    --compile.pass_pipeline mutation-functionalization+npu_auto_overlap
)

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_8p_cpt_4k_a3.sh" \
    --parallelism.spmd-backend partial_dtensor \
    "${EP_OVERLAP_ARGS[@]}" \
    "$@"
