# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped symbolic-Meta compatibility used only while tracing batch chunks."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode, ShapeEnv

if TYPE_CHECKING:
    from collections.abc import Iterator


_META_OVERRIDE_LOCK = threading.RLock()
_VENDOR_MX_SUPPORTS_UNBACKED: dict[str, bool] = {}


def _packed_mx_scale_extent(extent: int | torch.SymInt, block_size: int = 32) -> int | torch.SymInt:
    """Return packed scale pairs without specializing a symbolic extent."""
    # One scale covers ``block_size`` values and the NPU layout packs two
    # adjacent scales. This is equivalent to ceil(ceil(extent / block_size) / 2)
    # but remains integer/SymInt arithmetic throughout.
    packed_block_size = 2 * block_size
    return (extent + packed_block_size - 1) // packed_block_size


def _mx_quantized_output(input_dummy: torch.Tensor, dst_type: int) -> torch.Tensor:
    """Create the quantized output described by a TorchNPU MX dtype id."""
    if dst_type in (23, 291):
        return torch.empty_like(input_dummy, dtype=torch.float8_e5m2)
    if dst_type in (24, 292):
        return torch.empty_like(input_dummy, dtype=torch.float8_e4m3fn)

    trailing_extent = input_dummy.shape[-1]
    torch._check(
        trailing_extent % 2 == 0,
        lambda: "The trailing dimension must be divisible by 2 for packed FP4 output.",
    )
    return input_dummy.new_empty((*input_dummy.shape[:-1], trailing_extent // 2), dtype=torch.uint8)


def _dual_axis_scale_shapes(input_dummy: torch.Tensor) -> tuple[list[int | torch.SymInt], list[int | torch.SymInt]]:
    scale1_shape: list[int | torch.SymInt] = [*input_dummy.shape, 2]
    scale2_shape: list[int | torch.SymInt] = [*input_dummy.shape, 2]
    scale1_shape[-2] = _packed_mx_scale_extent(input_dummy.shape[-1])
    scale2_shape[-3] = _packed_mx_scale_extent(input_dummy.shape[-2])
    return scale1_shape, scale2_shape


def _symbolic_dynamic_mx_quant_meta(
    input_dummy: torch.Tensor,
    *,
    axis: int = -1,
    round_mode: str = "rint",
    dst_type: int = 296,
    block_size: int = 32,
    scale_alg: int = 0,
    dst_type_max: float | None = 0.0,
    max_low_bound: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shape-equivalent single-axis MX Meta kernel for symbolic quant axes."""
    del round_mode, dst_type_max, max_low_bound
    dim_num = input_dummy.dim()
    if not -dim_num <= axis < dim_num:
        raise RuntimeError(f"axis {axis} is out of range for an input with {dim_num} dimensions")
    if not (0 < block_size <= 1024 and block_size % 32 == 0):
        raise RuntimeError("block_size must be a positive multiple of 32 no greater than 1024")
    if scale_alg not in (0, 1, 2):
        raise RuntimeError(f"Invalid scale_alg value: {scale_alg}. Expected 0, 1, or 2.")

    axis = axis % dim_num
    scale_shape: list[int | torch.SymInt] = [*input_dummy.shape, 2]
    scale_shape[axis] = _packed_mx_scale_extent(input_dummy.shape[axis], block_size)
    output = _mx_quantized_output(input_dummy, dst_type)
    scale = input_dummy.new_empty(scale_shape, dtype=torch.uint8)
    return output, scale


def _symbolic_dynamic_mx_quant_with_dual_axis_meta(
    input_dummy: torch.Tensor,
    *,
    round_mode: str = "rint",
    dst_type: int = 296,
    scale_alg: int = 0,
    dst_type_max: float | None = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape-equivalent dual-axis MX Meta kernel that accepts unbacked SymInts."""
    del round_mode, dst_type_max
    if input_dummy.dim() < 2:
        raise RuntimeError("npu_dynamic_mx_quant_with_dual_axis requires an input with at least two dimensions")
    if scale_alg not in (0, 1, 2):
        raise RuntimeError(f"Invalid scale_alg value: {scale_alg}. Expected 0, 1, or 2.")

    scale1_shape, scale2_shape = _dual_axis_scale_shapes(input_dummy)

    y1 = _mx_quantized_output(input_dummy, dst_type)
    y2 = _mx_quantized_output(input_dummy, dst_type)
    scale1 = input_dummy.new_empty(scale1_shape, dtype=torch.uint8)
    scale2 = input_dummy.new_empty(scale2_shape, dtype=torch.uint8)
    return y1, scale1, y2, scale2


def _symbolic_dynamic_block_mx_quant_meta(
    input_dummy: torch.Tensor,
    *,
    round_mode: str = "rint",
    dst_type: int = 296,
    scale_alg: int = 0,
    dst_type_max: float | None = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shape-equivalent block-MX Meta kernel that accepts unbacked SymInts."""
    del round_mode, scale_alg, dst_type_max
    if input_dummy.dim() < 2:
        raise RuntimeError("npu_dynamic_block_mx_quant requires an input with at least two dimensions")

    scale1_shape, scale2_shape = _dual_axis_scale_shapes(input_dummy)
    output = _mx_quantized_output(input_dummy, dst_type)
    scale1 = input_dummy.new_empty(scale1_shape, dtype=torch.uint8)
    scale2 = input_dummy.new_empty(scale2_shape, dtype=torch.uint8)
    return output, scale1, scale2


_MX_META_PATCHES = {
    "npu_dynamic_mx_quant": _symbolic_dynamic_mx_quant_meta,
    "npu_dynamic_mx_quant_with_dual_axis": _symbolic_dynamic_mx_quant_with_dual_axis_meta,
    "npu_dynamic_block_mx_quant": _symbolic_dynamic_block_mx_quant_meta,
}


def _vendor_mx_supports_unbacked(op_name: str) -> bool:
    """Probe an installed TorchNPU Meta kernel once instead of version matching."""
    if op_name in _VENDOR_MX_SUPPORTS_UNBACKED:
        return _VENDOR_MX_SUPPORTS_UNBACKED[op_name]

    import torch_npu  # noqa: F401  # pyrefly: ignore[missing-import]

    op = getattr(torch.ops.npu, op_name, None)
    if op is None:
        _VENDOR_MX_SUPPORTS_UNBACKED[op_name] = True
        return True

    shape_env = ShapeEnv()
    unbacked_extent = shape_env.create_unbacked_symint()
    probe = torch.empty((unbacked_extent, 64), device="meta")
    kwargs = {"dst_type": 292, "scale_alg": 1}
    if op_name == "npu_dynamic_mx_quant":
        kwargs.update(axis=0, block_size=32)
    try:
        op.default(probe, **kwargs)
    except GuardOnDataDependentSymNode:
        supports_unbacked = False
    else:
        supports_unbacked = True
    _VENDOR_MX_SUPPORTS_UNBACKED[op_name] = supports_unbacked
    return supports_unbacked


@contextmanager
def batch_chunk_symbolic_meta_context(*, enabled: bool) -> Iterator[None]:
    """Temporarily repair unsupported Meta inference during a chunk trace."""
    if not enabled:
        yield
        return
    patches = {op_name: meta for op_name, meta in _MX_META_PATCHES.items() if not _vendor_mx_supports_unbacked(op_name)}
    if not patches:
        yield
        return

    # Dispatcher registrations are process-global, so serialize the short
    # trace window and destroy the override before leaving it. Library teardown
    # restores the vendor Meta kernel that was present before this context.
    with _META_OVERRIDE_LOCK:
        library = torch.library.Library("npu", "IMPL", "Meta")
        for op_name, meta in patches.items():
            library.impl(op_name, meta, allow_override=True)
        try:
            yield
        finally:
            library._destroy()
