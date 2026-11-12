#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${NODE_IPS:-}" ]]; then
    IFS=',' read -r -a node_ips <<< "${NODE_IPS}"
    default_nodes="${#node_ips[@]}"
else
    default_nodes=8
fi

MODULE="${MODULE:-torchtitan_npu_recipes.examples.deepseek_v4_1}" \
CONFIG="${CONFIG:-deepseek_v4_1_flash_sft_4k_a3}" \
NGPU="${NGPU:-16}" \
NNODES="${NNODES:-${default_nodes}}" \
exec bash "${SCRIPT_DIR}/../../scripts/run_train_multinodes.sh" "$@"
