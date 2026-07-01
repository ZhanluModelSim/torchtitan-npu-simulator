# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conversions between torch.Tensor metadata and the simulator's
framework-agnostic TensorMeta (L0 IR)."""

from __future__ import annotations

import torch

from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta

_DTYPE_STR_OVERRIDES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}

_DTYPE_BYTE_SIZES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
}

_DEFAULT_DTYPE_BYTE_SIZE = 4  # fall back to fp32-sized for unrecognized dtypes


def dtype_to_str(dtype: torch.dtype) -> str:
    """Canonical string name for a torch dtype (e.g. `torch.bfloat16` -> `"bfloat16"`)."""
    return _DTYPE_STR_OVERRIDES.get(dtype, str(dtype).replace("torch.", ""))


def dtype_byte_size(dtype_str: str) -> int:
    """Bytes per element for a canonical dtype string."""
    return _DTYPE_BYTE_SIZES.get(dtype_str, _DEFAULT_DTYPE_BYTE_SIZE)


def tensor_volume_bytes(shape: tuple[int, ...], dtype_str: str) -> int:
    """Total byte size of a tensor with the given shape and dtype."""
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel * dtype_byte_size(dtype_str)


def to_tensor_meta(tensor: torch.Tensor, name: str, is_parameter: bool = False) -> TensorMeta:
    """Build a TensorMeta from a live tensor (works for real, meta, or fake tensors --
    only `.shape`/`.dtype`/`.device` are read, never the underlying storage)."""
    return TensorMeta(
        name=name,
        shape=tuple(int(d) for d in tensor.shape),
        dtype=dtype_to_str(tensor.dtype),
        device=str(tensor.device),
        is_parameter=is_parameter,
    )
