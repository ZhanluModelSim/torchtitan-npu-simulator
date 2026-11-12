#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

MODULE="${MODULE:-torchtitan_npu_recipes.examples.deepseek_v4}" \
CONFIG="${CONFIG:-deepseek_v4_flash_qat_4k_a5}" \
NGPU="${NGPU:-8}" \
NNODES="${NNODES:-$(awk -F, '{print NF ? NF : 16}' <<< "${NODE_IPS:-}")}" \
CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1,npu0:288-311,npu1:312-335,npu2:336-359,npu3:360-383,npu4:96-119,npu5:120-143,npu6:144-167,npu7:168-191}" \
HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}" \
HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-17330}" \
ACL_DEVICE_SYNC_TIMEOUT="${ACL_DEVICE_SYNC_TIMEOUT:-2147480}" \
HCCL_EVENT_TIMEOUT="${HCCL_EVENT_TIMEOUT:-2147480}" \
HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}" \
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../../scripts/run_train_multinodes.sh" "$@"
