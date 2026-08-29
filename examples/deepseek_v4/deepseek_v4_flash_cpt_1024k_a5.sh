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

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_cpt_4k_a5.sh" \
    --parallelism.context-parallel-degree 128 \
    --parallelism.data-parallel-shard-degree 1 \
    --parallelism.data-parallel-replicate-degree 1 \
    --training.seq-len 1048576 \
    --training.global-batch-size 8 \
    "$@"
