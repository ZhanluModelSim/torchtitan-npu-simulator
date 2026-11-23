# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SLA2Attention + Multi-Head-MoE decoder (MAGI-2 preview style port)."""

from .model import (
    BlockRouter,
    InnerBlockSLA2,
    MultiHeadGate,
    MultiHeadMoE,
    MultiHeadMoEModel,
    SLA2Attention,
    TransformerBlock,
)
from .parallelize import parallelize_multi_head_moe

__all__ = [
    "BlockRouter",
    "InnerBlockSLA2",
    "MultiHeadGate",
    "MultiHeadMoE",
    "MultiHeadMoEModel",
    "SLA2Attention",
    "TransformerBlock",
    "parallelize_multi_head_moe",
]
