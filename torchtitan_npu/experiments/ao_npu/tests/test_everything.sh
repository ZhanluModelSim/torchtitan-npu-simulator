#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# terminate script on first error
set -e

BASE="torchtitan_npu/experiments/ao_npu/tests"

pytest "$BASE/test_param_swap_config.py" -s -v
pytest "$BASE/test_training.py" -s -v
pytest "$BASE/wrapper_tensors/test_param_swap_transform.py" -s -v
pytest "$BASE/wrapper_tensors/test_wrapper_tensor.py" -s -v
pytest "$BASE/wrapper_tensors/test_wrapper_ops.py" -s -v
pytest "$BASE/ops/test_mx_ops.py" -s -v
pytest "$BASE/ops/test_block_ops.py" -s -v

echo "all tests successful"
