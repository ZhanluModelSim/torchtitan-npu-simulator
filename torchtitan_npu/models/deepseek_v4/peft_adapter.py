# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""V4 keeps the shared DeepSeek PEFT conversion; the implementation lives in common."""

from torchtitan_npu.models.common.deepseek_lora import (
    DeepSeekPEFTAdapter,
    is_lora_parameter,
)

__all__ = ["DeepSeekPEFTAdapter", "is_lora_parameter"]
