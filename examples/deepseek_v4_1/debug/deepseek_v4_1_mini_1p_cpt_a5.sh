#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODULE="${MODULE:-torchtitan_npu_recipes.examples.deepseek_v4_1}" \
CONFIG="${CONFIG:-deepseek_v4_1_mini_1p_cpt_a5}" \
NGPU="${NGPU:-1}" \
NNODES="${NNODES:-1}" \
exec bash "${SCRIPT_DIR}/../../../scripts/run_train.sh" "$@"
