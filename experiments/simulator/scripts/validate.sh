#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python -m pytest tests/ -q "$@"

# Static analyzers do not resolve all editable-install import hooks.
# Supply source roots when the training packages were installed editable.
search_paths=()
if [[ -n "${TORCHTITAN_SOURCE:-}" ]]; then
    search_paths+=(--search-path "$TORCHTITAN_SOURCE")
fi
if [[ -n "${TORCHTITAN_NPU_SOURCE:-}" ]]; then
    search_paths+=(--search-path "$TORCHTITAN_NPU_SOURCE")
    if [[ -d "$TORCHTITAN_NPU_SOURCE/experiments/torchao-npu" ]]; then
        search_paths+=(--search-path "$TORCHTITAN_NPU_SOURCE/experiments/torchao-npu")
    fi
fi
if [[ -n "${TORCHAO_NPU_SOURCE:-}" ]]; then
    search_paths+=(--search-path "$TORCHAO_NPU_SOURCE")
fi
python -m pyrefly check --config pyproject.toml "${search_paths[@]}"
