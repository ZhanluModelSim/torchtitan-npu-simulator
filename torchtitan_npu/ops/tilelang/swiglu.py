# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile-safe custom-op boundary for the TileLang SwiGLU kernels.

The optional ``tile_kernels`` package is loaded only by the runtime custom-op
implementations. Fake implementations and the registered autograd formula
keep ``torch.compile``/AOT eager graph capture independent of that package.
"""

from functools import lru_cache
from typing import Any

import torch
from torch.distributed.tensor import DTensor


@lru_cache(maxsize=1)
def _load_tilelang_swiglu() -> tuple[Any, Any]:
    """Load the optional TileKernels entry points only when the op runs."""
    try:
        from tile_kernels.quant.swiglu_backward_kernel import swiglu_backward  # pyrefly: ignore [missing-import]
        from tile_kernels.quant.swiglu_forward_kernel import swiglu_forward  # pyrefly: ignore [missing-import]
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "TileLang SwiGLU requires the validated tile_kernels package and a loaded CANN environment"
        ) from error
    return swiglu_forward, swiglu_backward


def _validate_swiglu_input(x: torch.Tensor) -> int:
    if isinstance(x, DTensor):
        raise ValueError("TileLang SwiGLU does not support DTensor inputs; use the validated single-rank path")
    if x.device.type != "npu":
        raise ValueError(f"TileLang SwiGLU requires NPU input, got {x.device}")
    if x.ndim != 2 or not x.is_contiguous():
        raise ValueError("TileLang SwiGLU requires a contiguous 2-D [tokens, 2*hidden] tensor")
    if x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"TileLang SwiGLU requires BF16 or FP32 input, got {x.dtype}")
    if x.shape[-1] % 2:
        raise ValueError(f"TileLang SwiGLU requires an even last dimension, got {x.shape[-1]}")
    hidden = x.shape[-1] // 2
    if hidden % 64:
        raise ValueError(f"TileLang SwiGLU requires hidden size divisible by 64, got {hidden}")
    return hidden


def _clamp_sentinel(clamp_value: float | None) -> float:
    return -1.0 if clamp_value is None else float(clamp_value)


def _decode_clamp(clamp_value: float) -> float | None:
    return None if clamp_value < 0 else clamp_value


@torch.library.custom_op(
    "torchtitan_npu::tilelang_swiglu_forward",
    mutates_args=(),
    device_types="npu",
)
def tilelang_swiglu_forward_op(
    gate: torch.Tensor,
    up: torch.Tensor,
    clamp_value: float,
) -> torch.Tensor:
    """Run TileLang SwiGLU forward behind a compile-safe custom-op boundary."""
    if gate.shape != up.shape:
        raise ValueError(f"TileLang SwiGLU gate/up shapes must match, got {gate.shape} and {up.shape}")
    x = torch.cat((gate, up), dim=-1).contiguous()
    _validate_swiglu_input(x)
    forward, _ = _load_tilelang_swiglu()
    fmt = "bf16" if x.dtype == torch.bfloat16 else "fp32"
    return forward(x, fmt=fmt, clamp_value=_decode_clamp(clamp_value)).detach().clone()


@tilelang_swiglu_forward_op.register_fake
def _tilelang_swiglu_forward_fake(gate: torch.Tensor, up: torch.Tensor, clamp_value: float) -> torch.Tensor:
    if gate.ndim != 2 or up.ndim != 2 or gate.shape != up.shape:
        raise ValueError("TileLang SwiGLU requires matching 2-D gate/up tensors")
    return gate.new_empty(gate.shape)


@torch.library.custom_op(
    "torchtitan_npu::tilelang_swiglu_backward",
    mutates_args=(),
    device_types="npu",
)
def tilelang_swiglu_backward_op(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
    clamp_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TileLang SwiGLU backward behind a compile-safe custom-op boundary."""
    if gate.shape != up.shape:
        raise ValueError(f"TileLang SwiGLU gate/up shapes must match, got {gate.shape} and {up.shape}")
    x = torch.cat((gate, up), dim=-1).contiguous()
    _validate_swiglu_input(x)
    if grad_output.shape != gate.shape or grad_output.device != gate.device:
        raise ValueError("TileLang SwiGLU grad_output must match gate shape and device")
    _, backward = _load_tilelang_swiglu()
    grad_x = (
        backward(
            x,
            grad_output.to(dtype=x.dtype).contiguous(),
            fmt="bf16" if x.dtype == torch.bfloat16 else "fp32",
            clamp_value=_decode_clamp(clamp_value),
            do_recompute=False,
        )[0]
        .to(dtype=x.dtype)
        .contiguous()
    )
    hidden = x.shape[-1] // 2
    return grad_x[:, :hidden].contiguous(), grad_x[:, hidden:].contiguous()


@tilelang_swiglu_backward_op.register_fake
def _tilelang_swiglu_backward_fake(
    gate: torch.Tensor,
    up: torch.Tensor,
    grad_output: torch.Tensor,
    clamp_value: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if gate.ndim != 2 or up.ndim != 2 or gate.shape != up.shape:
        raise ValueError("TileLang SwiGLU requires matching 2-D gate/up tensors")
    if grad_output.shape != gate.shape:
        raise ValueError("TileLang SwiGLU grad_output must match gate shape")
    return torch.empty_like(gate), torch.empty_like(up)


def _tilelang_swiglu_setup_context(ctx: Any, inputs: tuple[Any, ...], output: torch.Tensor) -> None:
    del output
    gate, up, clamp_value = inputs
    ctx.save_for_backward(gate, up)
    ctx.clamp_value = clamp_value


def _tilelang_swiglu_backward(ctx: Any, grad_output: torch.Tensor | None) -> tuple[Any, Any, None]:
    if grad_output is None:
        return None, None, None
    gate, up = ctx.saved_tensors
    return (*tilelang_swiglu_backward_op(gate, up, grad_output, ctx.clamp_value), None)


torch.library.register_autograd(
    tilelang_swiglu_forward_op,
    _tilelang_swiglu_backward,
    setup_context=_tilelang_swiglu_setup_context,
)


def tilelang_swiglu(gate: torch.Tensor, up: torch.Tensor, clamp_value: float | None) -> torch.Tensor:
    """Apply the compile-safe TileLang SwiGLU custom op."""
    if gate.shape != up.shape:
        raise ValueError(f"TileLang SwiGLU gate/up shapes must match, got {gate.shape} and {up.shape}")
    if gate.device.type not in ("npu", "meta"):
        raise ValueError(f"TileLang SwiGLU requires NPU input, got {gate.device}")
    if gate.ndim != 2 or up.ndim != 2:
        raise ValueError("TileLang SwiGLU requires matching 2-D gate/up tensors")
    return tilelang_swiglu_forward_op(gate, up, _clamp_sentinel(clamp_value))
