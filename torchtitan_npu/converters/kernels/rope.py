# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
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

from ..convert_utils import replace_functions
from ..model_custom_interface import ModelCustomConfig, ModelCustomConverter
from ..registry import register_model_converter

logger = logging.getLogger(__name__)


def _complex_to_interleaved_cos_sin(freqs_cis: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(dtype)
    return cos, sin


def _select_freqs_cis(
    freqs_cis_local: torch.Tensor,
    positions: torch.Tensor | None,
    seqlen: int,
) -> torch.Tensor:
    """Select per-position complex ``freqs_cis`` as ``(seqlen, dim//2)``.

    NPU's gather/index kernels reject complex64, so index the *real* view
    (interleaved [real, imag] in the last dim) and convert back.

    The fused NPU interleaved RoPE op (``aclnnRotaryPositionEmbeddingV2``)
    only supports cos/sin broadcast over the batch dim, i.e. positions
    shared across batch. CP load balancing reorders sequential positions
    identically for every row, so a per-batch ``positions`` tensor
    ``(bsz, seqlen)`` carries the same row for all batch entries; use the
    first row. (Per-document positions under CP differ per row and are not
    supported by the fused op.)
    """
    freqs_cis_local = freqs_cis_local.resolve_conj()
    if positions is None:
        return freqs_cis_local[0:seqlen]
    pos = positions[0] if positions.dim() > 1 else positions
    freqs_cis_real = torch.view_as_real(freqs_cis_local)
    gathered = freqs_cis_real[pos]
    return torch.view_as_complex(gathered.contiguous())


def _wrap_dtensor_like(out_local: torch.Tensor, original_tensor: torch.Tensor, is_dt: bool) -> torch.Tensor:
    if is_dt:
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(
            out_local,
            device_mesh=original_tensor.device_mesh,  # pyrefly: ignore [missing-attribute]
            placements=original_tensor.placements,  # pyrefly: ignore [missing-attribute]
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
    rope_cache_local = rope_cache.to_local() if isinstance(rope_cache, DTensor) else rope_cache

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
    freqs_cis_local = freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis

    positions = _maybe_wrap_positions(positions, xq)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    seqlen = xq_local.shape[1]
    freqs_cis = _select_freqs_cis(freqs_cis_local, positions, seqlen)

    xq_f = xq_local.float()
    xk_f = xk_local.float()

    cos, sin = _complex_to_interleaved_cos_sin(freqs_cis, xq_f.dtype)
    xq_out = torch_npu.npu_rotary_mul(xq_f, cos, sin, rotary_mode="interleave").type_as(xq_local)
    xk_out = torch_npu.npu_rotary_mul(xk_f, cos.to(xk_f.dtype), sin.to(xk_f.dtype), rotary_mode="interleave").type_as(
        xk_local
    )

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
    freqs_cis_local = freqs_cis.to_local() if isinstance(freqs_cis, DTensor) else freqs_cis

    positions = _maybe_wrap_positions(positions, x)
    if isinstance(positions, DTensor):
        positions = positions.to_local()
    seqlen = x_local.shape[1]
    freqs_cis = _select_freqs_cis(freqs_cis_local, positions, seqlen)

    x_f = x_local.float()

    cos, sin = _complex_to_interleaved_cos_sin(freqs_cis, x_f.dtype)
    y = torch_npu.npu_rotary_mul(x_f, cos, sin, rotary_mode="interleave")
    y = y.to(x_local.dtype)

    if is_dtensor:
        from torch.distributed.tensor import DTensor as _DTensor

        y = _DTensor.from_local(y, device_mesh=x.device_mesh, placements=x.placements, run_check=False)

    return y


def reshape_for_broadcast_complex(
    freqs_cis: torch.Tensor,
    x: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """NPU-safe ``_reshape_for_broadcast_complex`` (patched onto common.rope at
    import time): select per-position freqs via _select_freqs_cis (real-view,
    dodges NPU complex64 index) and reshape to broadcast against complex x.
    """
    freqs_cis = _select_freqs_cis(freqs_cis, positions, x.shape[1])
    # Upstream has two branches: shared positions → (1, S, 1, D) via this
    # comprehension, per-sample positions → (B, S, 1, D) via explicit shape.
    # NPU only needs the shared branch because _select_freqs_cis always
    # returns (seqlen, d/2) — it forces batch-shared positions by taking
    # positions[0], matching the fused NPU RoPE op's constraint.
    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


_UPSTREAM_ROPE_MODULE = "torchtitan.models.common.rope"

_ROPE_REPLACEMENTS = {
    "apply_rotary_emb_complex": npu_apply_rotary_emb_complex,
    "apply_rotary_emb_single_complex": npu_apply_rotary_emb_single_complex,
    "apply_rotary_emb_cos_sin": npu_apply_rotary_emb_cos_sin,
}


def apply_reshape_for_broadcast_complex_patch() -> None:
    """Patch the inner helper (called by name within common.rope) so all callers
    are covered regardless of model from-imports. Fail loud if the target is
    missing -- a renamed/missing target must not silently disable the fix
    again (the previous patch went stale exactly that way).
    """
    mod = sys.modules.get(_UPSTREAM_ROPE_MODULE)
    target = "_reshape_for_broadcast_complex"
    if mod is None or not hasattr(mod, target):
        raise RuntimeError(
            f"RoPEKernel: target {_UPSTREAM_ROPE_MODULE}.{target} not found; "
            f"upstream likely renamed or moved it. Re-check this converter "
            f"against the current torchtitan version."
        )
    if getattr(mod, target) is reshape_for_broadcast_complex:
        return
    setattr(mod, target, reshape_for_broadcast_complex)
    logger.info(f"RoPEKernel: replaced {_UPSTREAM_ROPE_MODULE}.{target} with {reshape_for_broadcast_complex.__name__}")


class NpuRoPEConverter(ModelCustomConverter):
    @classmethod
    def _replace_one(cls, func_name: str, impl: Callable, model: nn.Module) -> int:
        mod = sys.modules.get(_UPSTREAM_ROPE_MODULE)
        if mod is not None and hasattr(mod, func_name):
            setattr(mod, func_name, impl)
            logger.info(f"RoPEKernel: replaced {_UPSTREAM_ROPE_MODULE}.{func_name} with {impl.__name__}")

        # from X import Y creates a local binding that setattr won't update;
        # replace_functions walks sys.modules to patch those local bindings.
        # The rope functions are imported by attention.py (and re-exported by
        # common/__init__.py), so we must search torchtitan.models.common in
        # addition to the model's own package.
        count = replace_functions(func_name, impl, model=model)
        upstream_pkg = model.__class__.__module__.replace("torchtitan_npu", "torchtitan")
        if upstream_pkg != model.__class__.__module__:
            count += replace_functions(func_name, impl, package=upstream_pkg)

        common_pkg = "torchtitan.models.common"
        if not upstream_pkg.startswith(common_pkg):
            count += replace_functions(func_name, impl, package=common_pkg)

        if count == 0 and mod is None:
            logger.warning(f"RoPEKernel: function {func_name!r} not found, skipping replacement")
        return count

    def convert(self, model: nn.Module):
        for func_name, impl in _ROPE_REPLACEMENTS.items():
            self._replace_one(func_name, impl, model)


@register_model_converter("npu_rope")
class RoPEModelConfig(ModelCustomConfig):
    model_converter = NpuRoPEConverter


apply_reshape_for_broadcast_complex_patch()
