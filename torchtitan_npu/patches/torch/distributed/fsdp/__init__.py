# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file adapts PyTorch composable FSDP internals
# (torch/distributed/fsdp/_fully_shard/_fsdp_param.py, _fsdp_param_group.py,
# and _fsdp_collectives.py, torch 2.15.0) for quantized multi-input payloads
# pending pytorch/pytorch#202101 and NPU compute-stream buffer reuse pending
# pytorch/pytorch#202343 and pytorch/pytorch#206795.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Upstream-PR: https://github.com/pytorch/pytorch/pull/206795

"""Compatibility and NPU buffer-lifetime adaptations for composable FSDP.

The ``all_gather_inputs`` relaxation lets quantized tensor subclasses return
multiple all-gather inputs (data + scales) under uneven sharding: the patched
check requires padded dim0 per input instead of the full padded shape.  Native
torch rejects those payloads; tracked upstream as
https://github.com/pytorch/pytorch/issues/202040, with the same relaxation
proposed in https://github.com/pytorch/pytorch/pull/202101.  Retire this patch
once a released torch carries the fix.

Optional compute-stream allocation adapts the pinned FSDP communication
interfaces to NPU stream-local memory reuse via a single switch
(``TORCHTITAN_NPU_FSDP_ALLOC_ON_COMPUTE_STREAM=1``, disabled by default and
read once at import time). When enabled, AG uses native asynchronous unshard
(moving allocation, cast, and copy-in to the current stream; configure explicit
forward prefetching to overlap communication with compute — the mechanism
exposed by upstream https://github.com/pytorch/pytorch/pull/202343), RS
redirects its output allocation to the compute stream while retaining native
communication streams (skipping CPU targets and HSDP partial-accumulation steps
where ``all_reduce_grads=False``; tracked upstream as
https://github.com/pytorch/pytorch/issues/206728 and proposed in
https://github.com/pytorch/pytorch/pull/206795), and
``MULTI_STREAM_MEMORY_REUSE`` defaults to ``2`` if unset or empty. Backend
requirements and supported paths are documented in
``docs/feature_guides/fsdp_compute_stream_allocation.md``. Retire the
compute-stream monkey patches once a released torch carries #202343 and #206795.
"""

from __future__ import annotations

import inspect
import os
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.distributed.fsdp._fully_shard import _fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_param import (
    FSDPParam,
    ShardedState,
    _to_dtype_if_needed,
)

from . import foreach_reduce as _foreach_reduce_owner

if TYPE_CHECKING:
    from collections.abc import Callable

_ORIGINAL_PARAM_GROUP_INIT = _fsdp_param_group.FSDPParamGroup.__init__
_ORIGINAL_ALL_GATHER_INPUTS = FSDPParam.all_gather_inputs
# ``property.fget`` is statically ``Callable | None``; this named property has a getter.
_ORIGINAL_ALL_GATHER_INPUTS_FGET = cast("Callable[[FSDPParam], Any]", _ORIGINAL_ALL_GATHER_INPUTS.fget)
_ALLOC_ON_COMPUTE_STREAM = os.getenv("TORCHTITAN_NPU_FSDP_ALLOC_ON_COMPUTE_STREAM", "0") == "1"
_DEFAULT_MULTI_STREAM_MEMORY_REUSE = "2"


class _ComputeStreamAllocator:
    """Allocate on compute; preserve native copy and communication streams."""

    def __init__(self, comm, device_handle, extra_streams=()):
        self._comm = comm
        self._device_handle = device_handle
        self._compute_stream = device_handle.current_stream()
        self._extra_streams = extra_streams

    def allocate(self, size, *, dtype, device):
        caller = self._device_handle.current_stream()
        with self._device_handle.stream(self._compute_stream):
            output = self._comm.allocate(size, dtype=dtype, device=device)
        if caller != self._compute_stream:
            # ``reduce_scatter_input`` is allocated on ``_compute_stream`` and never
            # read by ``all_reduce_stream``; only side-stream outputs need cross-stream tracking.
            caller.wait_stream(self._compute_stream)
            output.record_stream(caller)
            for stream in self._extra_streams:
                if stream is not None and stream != self._compute_stream and stream != caller:
                    output.record_stream(stream)
        return output

    def __call__(self, *args, **kwargs):
        return self._comm(*args, **kwargs)


@wraps(_ORIGINAL_PARAM_GROUP_INIT)
def _patched_param_group_init(self, *args, **kwargs):
    _ORIGINAL_PARAM_GROUP_INIT(self, *args, **kwargs)
    # Same native mode as FSDPModule._set_unshard_async_op(True), for every group.
    self.unshard_async_op = True


def _make_compute_stream_foreach_reduce(next_foreach_reduce: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(next_foreach_reduce)
    @torch.no_grad()
    def _patched_foreach_reduce(*args: Any, **kwargs: Any) -> Any:
        if not _ALLOC_ON_COMPUTE_STREAM:
            return next_foreach_reduce(*args, **kwargs)
        bound = _foreach_reduce_owner.FOREACH_REDUCE_SIGNATURE.bind(*args, **kwargs)
        values = bound.arguments
        device = values["device"]
        if device.type == "cpu" or (values["all_reduce_group"] is not None and not values["all_reduce_grads"]):
            # HSDP partial-accumulation outputs stay internal to reduce_scatter_stream
            # rather than backing sharded_param.grad; keep them on the RS stream.
            return next_foreach_reduce(*args, **kwargs)
        uses_all_reduce_stream = values["all_reduce_group"] is not None or values["all_reduce_hook"] is not None
        extra_streams = (values["all_reduce_stream"],) if uses_all_reduce_stream else ()
        values["reduce_scatter_comm"] = _ComputeStreamAllocator(
            values["reduce_scatter_comm"],
            torch.get_device_module(device),
            extra_streams,
        )
        return next_foreach_reduce(*bound.args, **bound.kwargs)

    return _patched_foreach_reduce


def _check_all_gather_inputs(fsdp_param: FSDPParam, all_gather_inputs) -> None:
    """Uneven-sharding contract: multi-input payloads (data + scales) are
    checked for padded dim0 only, keeping all-gather buffers equal-sized.

    Stands in for the native check pending upstream pytorch/pytorch#202040
    (fix proposed in pytorch/pytorch#202101).
    """
    local_tensor = fsdp_param._sharded_local_tensor
    padded = fsdp_param.padded_sharded_param_size
    if local_tensor.size() == padded:
        return  # even sharding (or already padded): nothing to check
    if len(all_gather_inputs) == 1:
        ok = all_gather_inputs[0].size() == padded
    else:
        ok = all(t.size(0) == padded[0] for t in all_gather_inputs)
    if not ok:
        # Raised only on padding ranks; elsewhere it manifests as an HCCL timeout.
        raise AssertionError(
            "When a parameter is unevenly sharded by FSDP "
            f"(orig size={fsdp_param._orig_size}, FSDP world size={fsdp_param.mesh_info.mesh.size()}), "
            "fsdp_pre_all_gather must return all-gather inputs padded to the "
            f"padded sharded size {padded} (single input: full shape; multiple "
            f"inputs: dim0) but got {[t.size() for t in all_gather_inputs]}"
        )


@wraps(_ORIGINAL_ALL_GATHER_INPUTS_FGET)
def _patched_all_gather_inputs(self: FSDPParam):
    # Mirrors the native property (torch 2.14.0.dev); uneven-sharding check delegated below.
    self._assert_in_states(ShardedState.SHARDED, ShardedState.SHARDED_POST_FORWARD)
    if self.sharded_state == ShardedState.SHARDED:
        if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather"):
            sharded_local_tensor = self._sharded_local_tensor
            if self.offload_to_cpu:
                sharded_local_tensor = sharded_local_tensor.to(self.device, non_blocking=True)
            pre_all_gather_signature = inspect.signature(
                # pyrefly: ignore [missing-attribute]
                sharded_local_tensor.fsdp_pre_all_gather
            )
            num_fn_params = len(pre_all_gather_signature.parameters)
            # Old signature only passes mesh; keep for BC for now
            if num_fn_params not in (1, 5):
                raise AssertionError(
                    f"Invalid fsdp_pre_all_gather: {pre_all_gather_signature}\n"
                    "Expects fsdp_pre_all_gather(self, mesh: DeviceMesh, "
                    "outer_size: torch.Size, outer_stride: tuple[int, ...], "
                    "module: nn.Module, mp_policy: MixedPrecisionPolicy)"
                )
            if num_fn_params == 1:
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                    # pyrefly: ignore [missing-attribute]
                ) = sharded_local_tensor.fsdp_pre_all_gather(self.shard_mesh_from_root)
            else:
                (
                    all_gather_inputs,
                    self._extensions_data.all_gather_metadata,
                    # pyrefly: ignore [missing-attribute]
                ) = sharded_local_tensor.fsdp_pre_all_gather(
                    self.shard_mesh_from_root,
                    self._orig_size,
                    self._contiguous_orig_stride,
                    self._module_info.module,
                    self.mp_policy,
                )
                _check_all_gather_inputs(self, all_gather_inputs)
            self._extensions_data.all_gather_input_sizes = [t.size() for t in all_gather_inputs]
            return [t.view(-1) for t in all_gather_inputs]
        sharded_param_data = self._sharded_param_data
        if self.offload_to_cpu:
            sharded_param_data = sharded_param_data.to(self.device, non_blocking=True)
        return [_to_dtype_if_needed(sharded_param_data, self.param_dtype)]
    return _ORIGINAL_ALL_GATHER_INPUTS_FGET(self)


def apply() -> None:
    if _ALLOC_ON_COMPUTE_STREAM and not os.environ.get("MULTI_STREAM_MEMORY_REUSE"):
        # Read lazily by torch_npu's ProcessGroupHCCL on the first collective;
        # defaulting here lets a single switch activate stream reuse cleanly.
        os.environ["MULTI_STREAM_MEMORY_REUSE"] = _DEFAULT_MULTI_STREAM_MEMORY_REUSE
    _foreach_reduce_owner.set_layer(
        "compute_stream",
        _make_compute_stream_foreach_reduce if _ALLOC_ON_COMPUTE_STREAM else None,
    )
    torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup.__init__ = (
        _patched_param_group_init if _ALLOC_ON_COMPUTE_STREAM else _ORIGINAL_PARAM_GROUP_INIT
    )
    torch.distributed.fsdp._fully_shard._fsdp_param.FSDPParam.all_gather_inputs = property(_patched_all_gather_inputs)


apply()
