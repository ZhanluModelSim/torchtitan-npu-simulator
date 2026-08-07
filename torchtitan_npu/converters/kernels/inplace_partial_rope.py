# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from collections.abc import Sequence

import torch
import torch.nn as nn
from torchtitan.models.common.rope import _maybe_wrap_positions

from ..model_custom_interface import ModelCustomConfig, ModelCustomConverter
from ..registry import register_model_converter
from .rope import _select_precomputed_rope_cache, _wrap_dtensor_like

logger = logging.getLogger(__name__)


def _reshape_partial_rotary_x_for_op(
    x: torch.Tensor,
    positions: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if positions is not None:
        position_index = positions.squeeze(0) if positions.dim() > 1 and positions.size(0) == 1 else positions
        if position_index.dim() == 1 and x.size(0) == position_index.numel():
            positions = position_index.unsqueeze(0)
            if x.ndim == 3:
                return x.unsqueeze(0), positions
            if x.ndim == 2:
                return x.unsqueeze(0).unsqueeze(2), positions

    if x.ndim == 3:
        return x.unsqueeze(2), positions
    if x.ndim == 4:
        return x, positions
    raise ValueError(f"Input tensor x's dim num should be 2, 3 or 4, actual {x.ndim}.")


def npu_apply_rotary_emb_partial_complex_(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    partial_slice: Sequence[int],
    inverse: bool = False,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Apply the fused inplace partial RoPE operation to ``x[..., start:end]``.

    ``freqs_cis`` must be a precomputed fp32 interleaved cos/sin cache with
    shape ``[2, cache_seq_len, end - start]``.

    The kernel updates only the selected slice. Autograd inputs are cloned
    because the custom operation cannot modify views in place.
    """
    from torch.distributed.tensor import DTensor

    if len(partial_slice) != 2:
        raise ValueError("partial_slice must contain exactly two integers.")
    start, end = partial_slice
    if start < 0 or end < start or end > x.shape[-1]:
        raise ValueError(f"Invalid partial_slice {partial_slice} for input last dim {x.shape[-1]}.")

    is_dtensor = isinstance(x, DTensor)
    x_local = x.to_local() if is_dtensor else x
    rope_cache_local = freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis
    # In-place updates on a leaf that requires grad violate autograd's versioning.
    if x_local.requires_grad and x_local.is_leaf:
        raise RuntimeError("in-place partial RoPE requires a non-leaf tensor when autograd is enabled.")

    positions = _maybe_wrap_positions(positions, x)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    x_for_op, positions = _reshape_partial_rotary_x_for_op(x_local, positions)
    materialized_for_autograd = x_for_op.requires_grad
    if materialized_for_autograd:
        x_for_op = x_for_op.clone()
    rope_width = end - start
    seqlen = x_for_op.shape[1]
    selected_cache = _select_precomputed_rope_cache(rope_cache_local, positions, seqlen, rope_width)
    cos, sin = selected_cache.unbind(0)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    if inverse:
        sin = -sin

    # pyrefly: ignore [missing-import]
    from cann_ops_transformer.ops import inplace_partial_rotary_mul

    cos = cos.to(device=x_for_op.device, dtype=torch.float32).contiguous()
    sin = sin.to(device=x_for_op.device, dtype=torch.float32).contiguous()
    inplace_partial_rotary_mul(
        x_for_op,
        cos,
        sin,
        rotary_mode="interleave",
        partial_slice=[start, end],
    )
    if not materialized_for_autograd:
        return x
    output_local = x_for_op.reshape(x_local.shape)
    return _wrap_dtensor_like(output_local, x, is_dtensor)


class NpuInplacePartialRoPEConverter(ModelCustomConverter):
    """Enable the CANN inplace partial RoPE implementation when applicable."""

    def convert(self, model: nn.Module):
        binding_name = "apply_partial_rotary_emb_"
        model_module = sys.modules.get(model.__class__.__module__)
        if model_module is None or not hasattr(model_module, binding_name):
            return

        try:
            # pyrefly: ignore [missing-import]
            from cann_ops_transformer.ops import inplace_partial_rotary_mul  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "npu_rope_inplace_partial requires a compatible cann_ops_transformer package providing "
                "inplace_partial_rotary_mul."
            ) from exc
        if getattr(model_module, binding_name) is npu_apply_rotary_emb_partial_complex_:
            return
        setattr(model_module, binding_name, npu_apply_rotary_emb_partial_complex_)
        logger.info(
            f"RoPEKernel: replaced {model_module.__name__}.{binding_name} "
            f"with {npu_apply_rotary_emb_partial_complex_.__name__}"
        )


@register_model_converter("npu_rope_inplace_partial")
class InplacePartialRoPEModelConfig(ModelCustomConfig):
    model_converter = NpuInplacePartialRoPEConverter
