# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smoke-test-only startup hook for the NPU compile bypass backend."""

from npu_bypass_triton_codegen import install

install()
