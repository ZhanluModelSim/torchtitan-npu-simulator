#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

MODULE="${MODULE:-torchtitan_npu_recipes.examples.deepseek_v4}" \
CONFIG="${CONFIG:-deepseek_v4_flash_8p_cpt_4k_a3_eager}" \
NGPU="${NGPU:-8}" \
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../../../scripts/run_train.sh" "$@"
