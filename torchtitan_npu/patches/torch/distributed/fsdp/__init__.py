# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file adapts composable FSDP's _fsdp_param.py all-gather check for
# quantized multi-input payloads pending pytorch/pytorch#202101.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility for quantized multi-input composable-FSDP all-gathers.

The ``all_gather_inputs`` relaxation lets quantized tensor subclasses return
multiple all-gather inputs (data + scales) under uneven sharding: the patched
check requires padded dim0 per input instead of the full padded shape.  Native
torch rejects those payloads; tracked upstream as
https://github.com/pytorch/pytorch/issues/202040, with the same relaxation
proposed in https://github.com/pytorch/pytorch/pull/202101.  Retire this patch
once a released torch carries the fix.
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from torch.distributed.fsdp._fully_shard._fsdp_param import (
    FSDPParam,
    ShardedState,
    _to_dtype_if_needed,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_ORIGINAL_ALL_GATHER_INPUTS = FSDPParam.all_gather_inputs
# ``property.fget`` is statically ``Callable | None``; this named property has a getter.
_ORIGINAL_ALL_GATHER_INPUTS_FGET = cast("Callable[[FSDPParam], Any]", _ORIGINAL_ALL_GATHER_INPUTS.fget)


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
    FSDPParam.all_gather_inputs = property(_patched_all_gather_inputs)


apply()
