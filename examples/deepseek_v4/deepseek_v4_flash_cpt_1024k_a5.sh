#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# 1M sequence specialization. Reuse the A5 4K wrapper and override only the
# Trainer configuration that differs from the 4K setup. User CLI arguments are
# appended last and therefore can override these defaults.

set -euo pipefail

# Quantization defaults are inherited from the A5 4K wrapper.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-deepseek_v4_flash_cp_cpt}" \
DSV4_PER_DOC_ALIGNMENT="${DSV4_PER_DOC_ALIGNMENT:-16384}" \
exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a5.sh" \
    --training.max-context-length 1048576 \
    --training.num-tokens-per-microbatch-per-dp-rank 1048576 \
    --training.num-tokens-per-train-step 8388608 \
    --parallelism.context-parallel-degree 128 \
    --parallelism.data-parallel-shard-degree 1 \
    --parallelism.data-parallel-replicate-degree 1 \
    "$@"
