#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Eight-node SFT wrapper around the common DeepSeek-V4 Flash SFT launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

NODE_IPS="${NODE_IPS:-xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, \
                      xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx, xx.xx.xx.xx}"
NGPU="${NGPU:-16}"
export NODE_IPS NGPU

exec bash "${SCRIPT_DIR}/deepseek_v4_flash_sft_4k_a3.sh" "$@"
