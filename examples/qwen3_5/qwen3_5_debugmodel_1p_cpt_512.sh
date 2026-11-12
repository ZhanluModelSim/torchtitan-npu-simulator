#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MODULE="${MODULE:-torchtitan_npu_recipes.examples.qwen3_5}" \
CONFIG="${CONFIG:-qwen3_5_debugmodel_1p_cpt_512}" \
NGPU="${NGPU:-1}" \
exec bash "${SCRIPT_DIR}/../../scripts/run_train.sh" "$@"
