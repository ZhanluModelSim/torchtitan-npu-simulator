# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Leaf quantize configs for NPU.

These are standalone dataclasses (no parent ``QATConfig`` machinery) consumed
by both training-time wrapper tensors and future PTQ algorithms.
"""

from dataclasses import dataclass, field

import torch
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase

NPU_SUPPORTED_ELEM_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float4_e2m1fn_x2,
]


@dataclass
class MXQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for MX quantization on NPU
    """

    block_size: int = 32

    # Dtypes for input and weights, supports FP8 and FP4 formats
    elem_dtype: torch.dtype = torch.float8_e4m3fn

    # How to cast to elem_dtype
    # −	for float4_e2m1fn_x2、float4_e1m2fn_x2, support "rint"、"floor"、"round"
    # −	for float8_e5m2、float8_e4m3fn，support "rint"
    round_mode: str = field(default="rint", compare=False)

    # Scale calculation algorithm.
    # - 0: standard max-abs scaling
    # - 1: CuBALS scaling (FP8 only)
    # - 2: DynamicDtypeRange implementation, fp4 only
    # When None, inferred from elem_dtype: 1 for FP8, 2 for FP4.
    scale_alg: int | None = field(default=None, compare=False)

    # Maximum value of the target dtype. 0.0 means auto-inferred from elem_dtype.
    # Required by NPU ops like npu_dynamic_mx_quant for FP4 quantization.
    dst_type_max: float = 0.0

    def __post_init__(self):
        assert self.block_size == 32, f"For MX formats, the block_size must be 32, block_size={self.block_size} passed."

        assert self.elem_dtype in NPU_SUPPORTED_ELEM_DTYPES, (
            f"elem_dtype must be one of {NPU_SUPPORTED_ELEM_DTYPES}, got {self.elem_dtype}"
        )

        if self.scale_alg is None:
            is_fp4 = self.elem_dtype in (torch.float4_e2m1fn_x2,)
            self.scale_alg = 2 if is_fp4 else 1


@dataclass
class BlockQuantizeConfig(FakeQuantizeConfigBase):
    """
    Config for block FP8 low-precision training.

    Most fields are constrained to a single value by the underlying kernels;
    :meth:`__post_init__` enforces these invariants so misconfiguration fails
    fast instead of erroring deep inside the op.
    """

    # Block size. Must be 32.
    block_size: int = 32

    # Target dtype for the quantized data. Either float8_e4m3fn or float8_e5m2.
    elem_dtype: torch.dtype = torch.float8_e4m3fn

    # Rounding mode. Must be "rint".
    round_mode: str = field(default="rint", compare=False)

    # Scale calculation algorithm. Must be 0.
    scale_alg: int = field(default=0, compare=False)

    # Maximum value of the target dtype. Must be 0.0 (auto-inferred from elem_dtype).
    dst_type_max: float = 0.0

    # When set, apply MXFP4 fake-quant to weights before the block FP8 matmul.
    mxfp4_fake_quantize_config: MXQuantizeConfig | None = None

    def __post_init__(self):
        # Kernel only supports block_size=32.
        assert self.block_size == 32, f"For block FP8, the block_size must be 32, block_size={self.block_size} passed."
        assert self.elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
            f"For block FP8, elem_dtype must be torch.float8_e4m3fn or torch.float8_e5m2, got {self.elem_dtype}"
        )
        assert self.round_mode == "rint", f"For block FP8, round_mode must be 'rint', got '{self.round_mode}'"
        assert self.scale_alg == 0, f"For block FP8, scale_alg must be 0, got {self.scale_alg}"
        assert self.dst_type_max == 0.0, f"For block FP8, dst_type_max must be 0.0, got {self.dst_type_max}"
        if self.mxfp4_fake_quantize_config is not None:
            assert self.mxfp4_fake_quantize_config.elem_dtype == torch.float4_e2m1fn_x2, (
                f"mxfp4_fake_quantize_config.elem_dtype must be FP4 (torch.float4_e2m1fn_x2), "
                f"got {self.mxfp4_fake_quantize_config.elem_dtype}"
            )
