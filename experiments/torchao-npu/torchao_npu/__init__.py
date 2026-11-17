# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchao_npu.configs import (
    DeepSeekV4QuantLightningIndexerConfig,
    DeepSeekV41QuantCompressorConfig,
    DeepSeekV41QuantLightningIndexerConfig,
    DeepSeekV41QuantSparseAttentionConfig,
    ModuleSwapConfig,
    ParamSwapConfig,
)

__all__ = [
    "DeepSeekV4QuantLightningIndexerConfig",
    "DeepSeekV41QuantCompressorConfig",
    "DeepSeekV41QuantLightningIndexerConfig",
    "DeepSeekV41QuantSparseAttentionConfig",
    "ModuleSwapConfig",
    "ParamSwapConfig",
]


def normalize_dim(dim: int, ndim: int) -> int:
    """Return ``dim`` as a non-negative index into an ``ndim``-D tensor."""
    if not -ndim <= dim < ndim:
        raise IndexError(f"dim {dim} out of range for {ndim}D tensor")
    return dim % ndim
