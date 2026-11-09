# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 dense decoder configurations for TorchTitan v0.3.0."""

from .model import Qwen3Model, as_npu_config

__all__ = ["Qwen3Model", "as_npu_config"]
