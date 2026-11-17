#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

# Vendor environment setup may read unset optional variables.
if [[ -n "${CANN_ENV_FILE:-}" ]]; then
    source "$CANN_ENV_FILE"
fi
exec python -m torchtitan_simulator "$@"
