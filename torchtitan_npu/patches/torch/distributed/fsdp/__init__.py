# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file wraps PyTorch composable FSDP internals
# (torch/distributed/fsdp/_fully_shard/_fsdp_param.py and
# _fsdp_collectives.py, torch 2.14.0.dev+git194fcb011f) as the downstream part
# of https://github.com/pytorch/pytorch/issues/156784.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-parameter mixed-precision policies for composable FSDP.

A ``MixedPrecisionPolicy`` holds one dtype for a whole parameter group, but
DeepSeek-V4 keeps a few parameters in FP32 while the rest of the model computes
in the training dtype.  :func:`policy_overrides` installs an exact-FQN policy
mapping for the duration of a block, and every policy built inside that block
carries the mapping, so it reaches ``fully_shard`` through the ordinary
``mp_policy`` argument.  ``init_dtype_attrs`` then resolves each parameter's own
policy during lazy initialization, and ``foreach_reduce`` unifies the resulting
mixed gradients before the native single-dtype reduce-scatter.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard import _fsdp_collectives, _fsdp_param_group
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

_ORIGINAL_POLICY_INIT = MixedPrecisionPolicy.__init__
_ORIGINAL_INIT_DTYPE_ATTRS = FSDPParam.init_dtype_attrs
_ORIGINAL_FOREACH_REDUCE = _fsdp_collectives.foreach_reduce


@contextmanager
def policy_overrides(overrides: Mapping[str, MixedPrecisionPolicy]) -> Iterator[None]:
    """Attach ``overrides`` to every policy built inside the block."""

    def init(self, *args, **kwargs) -> None:
        _ORIGINAL_POLICY_INIT(self, *args, **kwargs)
        object.__setattr__(self, "overrides", dict(overrides))

    previous, MixedPrecisionPolicy.__init__ = MixedPrecisionPolicy.__init__, init
    try:
        yield
    finally:
        MixedPrecisionPolicy.__init__ = previous


@wraps(_ORIGINAL_INIT_DTYPE_ATTRS)
def _patched_init_dtype_attrs(self, mp_policy: MixedPrecisionPolicy) -> None:
    overrides = getattr(mp_policy, "overrides", None)
    if not overrides:
        return _ORIGINAL_INIT_DTYPE_ATTRS(self, mp_policy)

    fqn = self._param_fqn
    if "_checkpoint_wrapped_module" in fqn:
        # Activation checkpointing module is prefixed with
        # ``_checkpoint_wrapped_module``
        fqn = ".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module")
    policy = overrides.get(fqn) or mp_policy
    self.mp_policy = policy  # extension hooks read the parameter's own policy
    _ORIGINAL_INIT_DTYPE_ATTRS(self, policy)
    # Use the group's (default) reduce dtype so all parameters in the group
    # share a single reduce-scatter dtype regardless of per-parameter override.
    self.reduce_dtype = mp_policy.reduce_dtype


@wraps(_ORIGINAL_FOREACH_REDUCE)
@torch.no_grad()
def _patched_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs):
    # Backward computes in each parameter's own dtype, so the gradients are mixed
    # (FP32 for the overridden families, BF16 for the rest) while native
    # ``foreach_reduce`` asserts a single dtype: unify them first.
    reduce_dtype = fsdp_params[0].reduce_dtype
    if reduce_dtype is not None and len({grad.dtype for grad in unsharded_grads}) > 1:
        unsharded_grads[:] = [grad if grad.dtype == reduce_dtype else grad.to(reduce_dtype) for grad in unsharded_grads]
    return _ORIGINAL_FOREACH_REDUCE(fsdp_params, unsharded_grads, *args, **kwargs)


def apply() -> None:
    FSDPParam.init_dtype_attrs = _patched_init_dtype_attrs
    # ``_fsdp_param_group`` binds ``foreach_reduce`` by value at import time,
    # so both references must be patched.
    _fsdp_collectives.foreach_reduce = _patched_foreach_reduce
    _fsdp_param_group.foreach_reduce = _patched_foreach_reduce


apply()
