# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared test fixtures for the override test suite.

The override packages are opt-in. This root conftest intentionally does not
import ``torchtitan_npu`` so source-contract and registration tests can collect
without loading torch_npu or activating process-wide patches.
"""

from __future__ import annotations
