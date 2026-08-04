# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 support for preserving selected parameters in their master dtype."""

import inspect

import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as fsdp_collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

from torchtitan_npu.distributed.fsdp_parameter_precision import (
    FSDP_PARAMETER_PRESERVE_DTYPE_ATTR,
)

_ORIGINAL_PARAM_INIT = FSDPParam.__init__
_ORIGINAL_INIT_DTYPE_ATTRS = FSDPParam.init_dtype_attrs
_ORIGINAL_FOREACH_REDUCE = fsdp_collectives.foreach_reduce
_FSDP_PARAM_PRESERVE_DTYPE_ATTR = "_torchtitan_npu_preserve_parameter_dtype"

_EXPECTED_PARAM_INIT_PARAMETERS = (
    "self",
    "param",
    "module_info",
    "mesh_info",
    "post_forward_mesh_info",
    "device",
    "shard_placement_fn",
    "mp_policy",
    "offload_policy",
)
_EXPECTED_INIT_DTYPE_ATTRS_PARAMETERS = ("self", "mp_policy")
_EXPECTED_FOREACH_REDUCE_PARAMETERS = (
    "fsdp_params",
    "unsharded_grads",
    "reduce_scatter_group",
    "reduce_scatter_stream",
    "reduce_scatter_comm",
    "orig_dtype",
    "reduce_dtype",
    "device",
    "gradient_divide_factor",
    "all_reduce_group",
    "all_reduce_stream",
    "all_reduce_grads",
    "partial_reduce_output",
    "all_reduce_hook",
    "force_sum_reduction_for_comms",
)


def _validate_signature(function, expected_parameters: tuple[str, ...], name: str) -> None:
    actual_parameters = tuple(inspect.signature(function).parameters)
    if actual_parameters != expected_parameters:
        raise RuntimeError(f"unsupported FSDP {name} signature for parameter precision patch: {actual_parameters}")


def _patched_param_init(self, param, *args, **kwargs):
    preserve_dtype = getattr(param, FSDP_PARAMETER_PRESERVE_DTYPE_ATTR, False)
    if not isinstance(preserve_dtype, bool):
        raise ValueError("invalid FSDP parameter dtype-preservation marker")
    _ORIGINAL_PARAM_INIT(self, param, *args, **kwargs)
    setattr(self, _FSDP_PARAM_PRESERVE_DTYPE_ATTR, preserve_dtype)


def _patched_init_dtype_attrs(self, mp_policy):
    _ORIGINAL_INIT_DTYPE_ATTRS(self, mp_policy)
    preserve_dtype = getattr(
        self,
        _FSDP_PARAM_PRESERVE_DTYPE_ATTR,
        False,
    )
    if not preserve_dtype:
        return

    if not self.orig_dtype.is_floating_point:
        raise ValueError(f"parameter dtype preservation requires a floating-point parameter, got {self.orig_dtype}")
    if hasattr(self._sharded_local_tensor, "fsdp_pre_all_gather") or hasattr(
        self._sharded_local_tensor,
        "fsdp_post_all_gather",
    ):
        raise ValueError("parameter dtype preservation cannot target an FSDP all-gather extension")

    unit_param_dtype = self.param_dtype or self.orig_dtype
    changes_compute_dtype = unit_param_dtype != self.orig_dtype
    if changes_compute_dtype and self.sharded_param.requires_grad and mp_policy.reduce_dtype != self.orig_dtype:
        raise ValueError(
            "preserving a trainable parameter's original dtype requires "
            "MixedPrecisionPolicy.reduce_dtype to match that original dtype; "
            f"got original dtype {self.orig_dtype} and reduce dtype {mp_policy.reduce_dtype}"
        )

    # Native FSDP interprets None as no cast before all-gather.
    self.param_dtype = None


def _contains_preserved_parameter(fsdp_params: list[FSDPParam]) -> bool:
    return any(getattr(param, _FSDP_PARAM_PRESERVE_DTYPE_ATTR, False) for param in fsdp_params)


@torch.no_grad()
def _patched_foreach_reduce(
    fsdp_params,
    unsharded_grads,
    reduce_scatter_group,
    reduce_scatter_stream,
    reduce_scatter_comm,
    orig_dtype,
    reduce_dtype,
    device,
    gradient_divide_factor,
    all_reduce_group,
    all_reduce_stream,
    all_reduce_grads,
    partial_reduce_output,
    all_reduce_hook,
    force_sum_reduction_for_comms=False,
):
    grad_dtypes = {grad.dtype for grad in unsharded_grads}
    if len(grad_dtypes) > 1 and _contains_preserved_parameter(fsdp_params):
        if reduce_dtype != orig_dtype:
            raise RuntimeError(
                "mixed FSDP gradients for preserved parameters require reduce_dtype "
                f"to match orig_dtype, got reduce_dtype={reduce_dtype} and orig_dtype={orig_dtype}"
            )
        unsharded_grads[:] = [grad if grad.dtype == reduce_dtype else grad.to(reduce_dtype) for grad in unsharded_grads]
    return _ORIGINAL_FOREACH_REDUCE(
        fsdp_params,
        unsharded_grads,
        reduce_scatter_group,
        reduce_scatter_stream,
        reduce_scatter_comm,
        orig_dtype,
        reduce_dtype,
        device,
        gradient_divide_factor,
        all_reduce_group,
        all_reduce_stream,
        all_reduce_grads,
        partial_reduce_output,
        all_reduce_hook,
        force_sum_reduction_for_comms,
    )


def apply_patch() -> None:
    if getattr(FSDPParam, "_torchtitan_npu_parameter_precision_patch", False):
        return
    _validate_signature(
        _ORIGINAL_PARAM_INIT,
        _EXPECTED_PARAM_INIT_PARAMETERS,
        "FSDPParam.__init__",
    )
    _validate_signature(
        _ORIGINAL_INIT_DTYPE_ATTRS,
        _EXPECTED_INIT_DTYPE_ATTRS_PARAMETERS,
        "FSDPParam.init_dtype_attrs",
    )
    _validate_signature(
        _ORIGINAL_FOREACH_REDUCE,
        _EXPECTED_FOREACH_REDUCE_PARAMETERS,
        "foreach_reduce",
    )

    FSDPParam.__init__ = _patched_param_init
    FSDPParam.init_dtype_attrs = _patched_init_dtype_attrs
    fsdp_collectives.foreach_reduce = _patched_foreach_reduce
    fsdp_param_group.foreach_reduce = _patched_foreach_reduce
    FSDPParam._torchtitan_npu_parameter_precision_patch = True


apply_patch()
