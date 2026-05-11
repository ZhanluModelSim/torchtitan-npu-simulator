# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from collections.abc import Callable

import torch
import torch.nn as nn
import torch_npu

from torchtitan.models.common.rope import (
    _maybe_wrap_positions,
    _reshape_for_broadcast_cos_sin,
)

from ..base_converter import BaseConverter
from ..convert_utils import replace_functions
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


def _complex_to_interleaved_cos_sin(
    freqs_cis: torch.Tensor, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(dtype)
    return cos, sin


def _wrap_dtensor_like(
    out_local: torch.Tensor, original_tensor: torch.Tensor, is_dt: bool
) -> torch.Tensor:
    if is_dt:
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(
            out_local,
            device_mesh=original_tensor.device_mesh,
            placements=original_tensor.placements,
            run_check=False,
        )
    return out_local


def npu_apply_rotary_emb_cos_sin(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    xq_is_dt = isinstance(xq, DTensor)
    xk_is_dt = isinstance(xk, DTensor)
    xq_local = xq.to_local() if xq_is_dt else xq
    xk_local = xk.to_local() if xk_is_dt else xk
    rope_cache_local = (
        rope_cache.to_local() if isinstance(rope_cache, DTensor) else rope_cache
    )

    positions = _maybe_wrap_positions(positions, xq)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    head_dim = xq_local.shape[-1]
    rope_cache = _reshape_for_broadcast_cos_sin(rope_cache_local, xq_local, positions)
    cos = rope_cache[..., :head_dim].to(device=xq_local.device)
    sin = rope_cache[..., head_dim:].to(device=xq_local.device)

    xq_f = xq_local.float()
    xk_f = xk_local.float()
    xq_out = torch_npu.npu_rotary_mul(xq_f, cos, sin)
    xk_out = torch_npu.npu_rotary_mul(xk_f, cos, sin)

    xq_out = xq_out.type_as(xq_local)
    xk_out = xk_out.type_as(xk_local)

    xq_out = _wrap_dtensor_like(xq_out, xq, xq_is_dt)
    xk_out = _wrap_dtensor_like(xk_out, xk, xk_is_dt)

    return xq_out, xk_out


def npu_apply_rotary_emb_complex(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    xq_is_dt = isinstance(xq, DTensor)
    xk_is_dt = isinstance(xk, DTensor)
    xq_local = xq.to_local() if xq_is_dt else xq
    xk_local = xk.to_local() if xk_is_dt else xk
    freqs_cis_local = (
        freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis
    )

    positions = _maybe_wrap_positions(positions, xq)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    seqlen = xq_local.shape[1]
    if positions is None:
        freqs_cis = freqs_cis_local[0:seqlen]
    elif positions.size(0) == 1:
        freqs_cis = freqs_cis_local[positions.squeeze(0)]
    else:
        freqs_cis_expanded = freqs_cis_local[None, :, None, :].expand(
            xq_local.shape[0], -1, -1, -1
        )
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, freqs_cis.shape[-1]),
        ).squeeze(2)

    xq_f = xq_local.float()
    xk_f = xk_local.float()

    cos, sin = _complex_to_interleaved_cos_sin(freqs_cis, xq_f.dtype)
    xq_out = torch_npu.npu_rotary_mul(xq_f, cos, sin, rotary_mode="interleave").type_as(
        xq_local
    )
    xk_out = torch_npu.npu_rotary_mul(
        xk_f, cos.to(xk_f.dtype), sin.to(xk_f.dtype), rotary_mode="interleave"
    ).type_as(xk_local)

    xq_out = _wrap_dtensor_like(xq_out, xq, xq_is_dt)
    xk_out = _wrap_dtensor_like(xk_out, xk, xk_is_dt)

    return xq_out, xk_out


def npu_apply_rotary_emb_single_complex(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    is_dtensor = isinstance(x, DTensor)
    x_local = x.to_local() if is_dtensor else x
    freqs_cis_local = (
        freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis
    )

    positions = _maybe_wrap_positions(positions, x)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    seqlen = x_local.shape[1]
    if positions is None:
        freqs_cis = freqs_cis_local[0:seqlen]
    elif positions.size(0) == 1:
        freqs_cis = freqs_cis_local[positions.squeeze(0)]
    else:
        freqs_cis_expanded = freqs_cis_local[None, :, None, :].expand(
            x_local.shape[0], -1, -1, -1
        )
        freqs_cis = torch.gather(
            freqs_cis_expanded,
            dim=1,
            index=positions.unsqueeze(-1)
            .unsqueeze(-1)
            .expand(-1, -1, 1, freqs_cis.shape[-1]),
        ).squeeze(2)

    x_f = x_local.float()

    cos, sin = _complex_to_interleaved_cos_sin(freqs_cis, x_f.dtype)
    y = torch_npu.npu_rotary_mul(x_f, cos, sin, rotary_mode="interleave")
    y = y.to(x_local.dtype)

    if is_dtensor:
        from torch.distributed.tensor import DTensor as _DTensor

        y = _DTensor.from_local(
            y, device_mesh=x.device_mesh, placements=x.placements, run_check=False
        )

    return y


_UPSTREAM_ROPE_MODULE = "torchtitan.models.common.rope"

_ROPE_REPLACEMENTS = {
    "apply_rotary_emb_complex": npu_apply_rotary_emb_complex,
    "apply_rotary_emb_single_complex": npu_apply_rotary_emb_single_complex,
    "apply_rotary_emb_cos_sin": npu_apply_rotary_emb_cos_sin,
}


@register_npu_converter("npu_rope")
class RoPEKernel(BaseConverter):
    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        count = 0
        for func_name, impl in _ROPE_REPLACEMENTS.items():
            count += cls._replace_one(func_name, impl, model)
        return count

    @classmethod
    def _replace_one(cls, func_name: str, impl: Callable, model: nn.Module) -> int:
        mod = sys.modules.get(_UPSTREAM_ROPE_MODULE)
        if mod is not None and hasattr(mod, func_name):
            setattr(mod, func_name, impl)
            logger.info(
                f"RoPEKernel: replaced {_UPSTREAM_ROPE_MODULE}.{func_name} "
                f"with {impl.__name__}"
            )

        # from X import Y creates a local binding that setattr won't update
        count = replace_functions(func_name, impl, model=model)
        upstream_pkg = model.__class__.__module__.replace(
            "torchtitan_npu", "torchtitan"
        )
        if upstream_pkg != model.__class__.__module__:
            count += replace_functions(func_name, impl, package=upstream_pkg)

        if count == 0 and mod is None:
            logger.warning(
                f"RoPEKernel: function {func_name!r} not found, "
                f"skipping replacement"
            )
        return count
