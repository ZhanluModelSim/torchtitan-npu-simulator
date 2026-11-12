#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

MODULE="${MODULE:-torchtitan_npu_recipes.examples.deepseek_v4}" \
CONFIG="${CONFIG:-deepseek_v4_flash_cpt_4k_a3}" \
NGPU="${NGPU:-16}" \
NNODES="${NNODES:-$(awk -F, '{print NF ? NF : 8}' <<< "${NODE_IPS:-}")}" \
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../../scripts/run_train_multinodes.sh" "$@"
