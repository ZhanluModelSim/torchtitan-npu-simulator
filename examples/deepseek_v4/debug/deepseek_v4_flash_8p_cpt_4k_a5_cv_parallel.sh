#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Add batch C/V scheduling to the A5 training recipe.
# Append CLI arguments to override training defaults, for example:
#   bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5_cv_parallel.sh --training.steps 10
set -euo pipefail

export MODULE=torchtitan_npu.models.deepseek_v4
export CONFIG="${CONFIG:-graph_trainer_deepseek_v4_flash_43layers_16experts}"
export NGPU="${NGPU:-8}"

bash examples/deepseek_v4/debug/deepseek_v4_flash_8p_cpt_4k_a5.sh \
    --parallelism.spmd-backend partial_dtensor \
    --parallelism.data-parallel-shard-degree "${NGPU}" \
    --parallelism.data-parallel-replicate-degree 1 \
    --parallelism.expert-parallel-degree "${NGPU}" \
    --training.local-batch-size 2 \
    --compile.mode aot_fx_trace \
    --compile.enable-passes \
    --compile.backend aot_eager \
    --compile.inductor-compilation regional \
    --compile.pass-pipeline cv_parallel \
    --compile.memory-policy full \
    --compile.ep-overlap.enabled \
    --compile.ep-overlap.chunk-dim batch \
    --compile.ep-overlap.strategy graph \
    --compile.ep-overlap.module-fqn "layers.*" \
    --compile.no-enable-autoparallel \
    --compile.no-enable-fsdp-ag-rs-overlap \
    --compile.no-enable-fsdp-dense-region-overlap \
    --compile.no-numerics-changing-optim \
    --override.imports \
        torchtitan_npu.override.common.rms_norm.asc \
        torchtitan_npu.override.common.rope.asc_partial \
        torchtitan_npu.override.common.swiglu_group.asc \
        torchtitan_npu.override.common.swiglu_group.asc_shared_experts \
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata \
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li \
        torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_metadata.asc_metadata \
        torchtitan_npu.override.deepseek_v4.sparse_attn.asc \
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_pre \
        torchtitan_npu.override.deepseek_v4.mhc.asc_hc_post \
        torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_token_dispatcher.asc_dispatcher \
    "$@"
