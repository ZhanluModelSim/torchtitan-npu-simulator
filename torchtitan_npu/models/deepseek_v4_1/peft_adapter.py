# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""V4.1 entry point for the shared DeepSeek PEFT conversion."""

__all__ = ["DeepSeekPEFTAdapter", "is_lora_parameter"]

from torchtitan_npu.models.common.deepseek_lora import (
    DeepSeekPEFTAdapter,
    is_lora_parameter,
)
