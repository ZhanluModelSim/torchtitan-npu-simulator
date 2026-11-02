# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preserve caller FSDP policies around native chunked-loss temporary overrides."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, cast

from torch.distributed.fsdp import FSDPModule

from torchtitan_simulator.compat.patches import PatchScope

if TYPE_CHECKING:
    from torch import nn


@contextmanager
def _preserve_head_policies(head: FSDPModule) -> Iterator[None]:
    # The native reshard setters recurse, while its gradient-sync setter does
    # not. Preserve mesh objects themselves: bool conversion loses partial
    # reshard policies, and the head can own more than one parameter group.
    modules = cast("nn.Module", head).modules()
    states = [module._get_fsdp_state() for module in modules if isinstance(module, FSDPModule)]
    forward_policies = [(state, state._auto_reshard_after_forward) for state in states]
    group_policies = [
        (group, group.post_forward_mesh_info, group.reshard_after_backward)
        for state in states
        for group in state._fsdp_param_groups
    ]
    sync_policies = [
        (group, group.reduce_grads, group.all_reduce_grads) for group in head._get_fsdp_state()._fsdp_param_groups
    ]
    try:
        yield
    finally:
        for state, auto_reshard in forward_policies:
            state._auto_reshard_after_forward = auto_reshard
        for group, mesh_info, reshard_backward in group_policies:
            group.post_forward_mesh_info = mesh_info
            group.reshard_after_backward = reshard_backward
        for group, reduce_grads, all_reduce_grads in sync_policies:
            group.reduce_grads = reduce_grads
            group.all_reduce_grads = all_reduce_grads


def install_chunked_loss_policy_restore(scope: PatchScope) -> None:
    """Keep native chunk execution intact; restore only policies it overrides."""
    from torchtitan.components.loss import ChunkedLossWrapper

    original = ChunkedLossWrapper.__call__

    @wraps(original)
    def call(self, *args, **kwargs):
        head = self.lm_head
        if not isinstance(head, FSDPModule):
            return original(self, *args, **kwargs)
        with _preserve_head_policies(head):
            return original(self, *args, **kwargs)

    scope.replace(ChunkedLossWrapper, "__call__", call)
