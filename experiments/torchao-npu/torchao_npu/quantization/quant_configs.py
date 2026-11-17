# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Leaf quantize configs for NPU.

These are standalone dataclasses (no parent ``QATConfig`` machinery) consumed
by both training-time wrapper tensors and future PTQ algorithms.
"""

import warnings
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch_npu
from torchao.quantization.qat.fake_quantize_config import FakeQuantizeConfigBase

from torchao_npu.quantization import _NPU_DTYPE_DICT, _SUPPORTED_MX_ELEM_DTYPES

_DYNAMIC_QUANT_MODES = frozenset(("pertoken", "pertensor", "perchannel"))


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

    @property
    def scale_dtype(self) -> torch.dtype:
        return torch.float8_e8m0fnu

    # ``dst_type`` token the NPU quant ops expect; they do not accept
    # ``torch.float4_e2m1fn_x2``.
    @property
    def npu_elem_dtype(self) -> int:
        return _NPU_DTYPE_DICT[self.elem_dtype]

    # For ``npu_quant_matmul``'s ``x1_dtype``/``x2_dtype``: only FP4 (stored as
    # uint8) needs the explicit hint; FP8 dtypes must not be passed -- doing so
    # causes a RuntimeError under ``torch.compile``'s fake-tensor tracing.
    @property
    def npu_matmul_dtype(self) -> int | None:
        return self.npu_elem_dtype if self.elem_dtype is torch.float4_e2m1fn_x2 else None

    @property
    def npu_scale_dtype(self) -> int:
        return _NPU_DTYPE_DICT[self.scale_dtype]

    def __post_init__(self):
        assert self.block_size > 0 and self.block_size % 32 == 0, (
            f"For MX formats, the block_size must be a positive multiple of 32, block_size={self.block_size} passed."
        )

        assert self.elem_dtype in _SUPPORTED_MX_ELEM_DTYPES, (
            f"elem_dtype must be one of {_SUPPORTED_MX_ELEM_DTYPES}, got {self.elem_dtype}"
        )

        is_fp4 = self.elem_dtype is torch.float4_e2m1fn_x2
        if self.scale_alg is None:
            self.scale_alg = 2 if is_fp4 else 1

        if is_fp4:
            assert self.scale_alg in (0, 2), (
                f"{type(self).__name__} only supports scale_alg=0 or 2 for FP4, got {self.scale_alg}"
            )


@dataclass
class FP8QuantizeConfig(FakeQuantizeConfigBase):
    """E4M3FN dynamic quantization configuration."""

    elem_dtype: torch.dtype = torch.float8_e4m3fn
    quant_mode: str = "pertoken"

    @property
    def npu_elem_dtype(self) -> int:
        return _NPU_DTYPE_DICT[self.elem_dtype]

    def __post_init__(self):
        if self.elem_dtype is not torch.float8_e4m3fn:
            raise ValueError("FP8QuantizeConfig only supports torch.float8_e4m3fn")
        if self.quant_mode not in _DYNAMIC_QUANT_MODES:
            raise ValueError(f"FP8QuantizeConfig quant_mode must be one of {_DYNAMIC_QUANT_MODES}")


@dataclass
class HiF8QuantizeConfig(FakeQuantizeConfigBase):
    """HiFloat8 dynamic quantization configuration.

    Serves two call sites:

    - Lightning Indexer Q/K quantization (:mod:`torchao_npu.ops.li_ops`).
    - Dense/MoE weight+activation matmul quantization
      (:mod:`torchao_npu.ops.hif8_ops`, current/constant tensor scaling: the
      scale is recomputed fresh via ``npu_dynamic_quant`` on every call, no
      persisted per-parameter state).
    """

    elem_dtype: object = torch_npu.hifloat8
    dst_type_max: float = 0.0
    quant_mode: str = "pertensor"

    def __post_init__(self):
        if self.elem_dtype != torch_npu.hifloat8:
            raise ValueError("HiF8QuantizeConfig requires torch_npu.hifloat8")
        if self.quant_mode not in _DYNAMIC_QUANT_MODES:
            raise ValueError(f"HiF8QuantizeConfig quant_mode must be one of {_DYNAMIC_QUANT_MODES}")
        if self.dst_type_max not in (0.0, 15.0, 56.0, 224.0, 32768.0):
            raise ValueError("HiF8QuantizeConfig dst_type_max must be one of 0, 15, 56, 224, or 32768")


@dataclass
class BlockMXQuantizeConfig(MXQuantizeConfig):
    """
    Config for block MX low-precision training.

    Extends :class:`MXQuantizeConfig`, inheriting its fields and ``npu_*``
    properties.

    - Without ``mxfp4_fake_quantize_config``, the accepted values follow
      :class:`MXQuantizeConfig` except ``scale_alg`` for FP8: the block MX
      kernel only supports 0 (the inferred default; MX would infer 1). FP4
      keeps the inherited MX constraint (0 or 2, ``None`` infers 2).
    - When ``mxfp4_fake_quantize_config`` is set, the nested config's
      ``elem_dtype`` must be FP4 and this config's ``elem_dtype`` must be FP8;
      the fused FP4-to-block-MX kernel only consumes ``elem_dtype`` (its
      ``dst_type``), so ``dst_type_max`` and ``round_mode`` are ignored (with
      a warning when set to non-default values). ``scale_alg`` is not consumed
      either, but still follows the FP8 constraint above.
    """

    # When set, apply MXFP4 fake-quant to weights before the block MX matmul.
    mxfp4_fake_quantize_config: MXQuantizeConfig | None = None

    # When set, enable FSDP pre-quantization: the weight is quantized to block
    # MX in ``fsdp_pre_all_gather`` (before all_gather) and the quantized
    # weight + scales are all-gathered, so forward/backward reuse the
    # pre-quantized data instead of re-quantizing on the fly.
    fsdp_prequantize: bool = field(default=False, compare=False)
    fsdp_prequantize_dtype: Literal["fp8", "mxfp4"] = field(default="fp8", compare=False)

    def __post_init__(self):
        if self.fsdp_prequantize_dtype not in ("fp8", "mxfp4"):
            raise ValueError("fsdp_prequantize_dtype must be fp8 or mxfp4")
        if self.fsdp_prequantize_dtype == "mxfp4" and self.mxfp4_fake_quantize_config is None:
            raise ValueError(
                "fsdp_prequantize_dtype='mxfp4' requires mxfp4_fake_quantize_config; "
                "enable MXFP4 QAT for the selected weight"
            )
        # Mode invariants first: the branches below rely on mxfp4 mode
        # implying an FP8 ``elem_dtype``.
        if self.mxfp4_fake_quantize_config is not None:
            assert self.mxfp4_fake_quantize_config.elem_dtype == torch.float4_e2m1fn_x2, (
                f"mxfp4_fake_quantize_config.elem_dtype must be FP4 (torch.float4_e2m1fn_x2), "
                f"got {self.mxfp4_fake_quantize_config.elem_dtype}"
            )

            assert self.elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
                f"When fake quantization is enabled, elem_dtype must be torch.float8_e4m3fn or torch.float8_e5m2, "
                f"got {self.elem_dtype}"
            )

            if not (self.dst_type_max == 0 and self.round_mode == "rint"):
                warnings.warn(
                    "When fake quantization is enabled, ``dst_type_max`` and ``round_mode`` of "
                    "BlockMXQuantizeConfig are ignored.",
                    stacklevel=2,
                )

        if self.elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # FP8: the block MX kernel only supports scale_alg=0, so the
            # inference differs from the parent's (which would pick 1). The
            # mxfp4 fake-quantization path does not consume ``scale_alg``, but
            # the same check and normalization apply there so the field's
            # semantics do not depend on the mode. Must run before
            # ``super().__post_init__()`` to pre-empt its None -> 1 inference.
            assert self.scale_alg is None or self.scale_alg == 0, (
                f"BlockMXQuantizeConfig only supports scale_alg=0 for FP8, got {self.scale_alg}"
            )
            self.scale_alg = 0

        super().__post_init__()

        if self.mxfp4_fake_quantize_config is None:
            if self.elem_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                assert self.scale_alg == 0, (
                    f"BlockMXQuantizeConfig only supports scale_alg=0 for FP8, got {self.scale_alg}"
                )
            else:
                assert self.scale_alg in (0, 2), (
                    f"BlockMXQuantizeConfig only supports scale_alg=0 or 2 for FP4, got {self.scale_alg}"
                )


# Safe-unpickling allowlist: DCP loads checkpoints with torch.load(weights_only=True).
torch.serialization.add_safe_globals([MXQuantizeConfig, BlockMXQuantizeConfig, HiF8QuantizeConfig, FP8QuantizeConfig])
