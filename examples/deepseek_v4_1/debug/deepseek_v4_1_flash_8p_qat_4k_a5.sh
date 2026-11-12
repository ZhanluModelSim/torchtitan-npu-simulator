#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}"

MODULE="${MODULE:-torchtitan_npu_recipes.examples.deepseek_v4_1}" \
CONFIG="${CONFIG:-deepseek_v4_1_flash_8p_qat_4k_a5}" \
NGPU="${NGPU:-8}" \
NNODES="${NNODES:-1}" \
exec bash "${SCRIPT_DIR}/../../../scripts/run_train.sh" "$@"
