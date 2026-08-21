# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Opt-in NPU overrides for TorchTitan."""

import torch_npu

try:
    _IS_A5 = torch_npu.npu.get_device_name().startswith("Ascend950")
except (AttributeError, TypeError):
    # CPU/UT runs may not have a current NPU device selected.  Keep the
    # non-A5 behavior in that case.
    _IS_A5 = False
