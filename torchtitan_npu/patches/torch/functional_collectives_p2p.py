# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/_functional_collectives.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fix functional P2P peer-rank semantics on sub-groups.

Patches ``isend_inplace`` / ``irecv_inplace`` in
``torch.distributed._functional_collectives``.

The traceable functional P2P helpers forward a global peer rank to
``torch.ops._c10d_functional.isend`` / ``irecv``, but that op (like the eager
path) expects a group-local rank. On the WORLD group the two coincide; on a
sub-group (e.g. a context-parallel group carved from a larger world) the global
rank exceeds the group size and the P2P call fails
(see https://github.com/pytorch/pytorch/pull/161213).

Fix: convert the peer to group-local before the call -- a global ``dst`` /
``src`` via ``get_group_rank``, a ``group_dst`` / ``group_src`` kept as-is --
and default ``dst`` / ``src`` to ``None`` so the group-relative convention also
binds under ``torch.compile``.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed._functional_collectives import (
    _are_we_tracing,
    _maybe_wrap_tensor,
    _resolve_group_name,
)

_C10D_FUNCTIONAL_NS = "_c10d_functional"
_c10d_functional = getattr(torch.ops, _C10D_FUNCTIONAL_NS)


def _to_group_local(group, peer, group_peer):
    """Return the GROUP-LOCAL peer rank expected by ProcessGroup send/recv."""
    if group_peer is not None and group_peer != -1:
        if peer is not None:
            raise ValueError(
                "Cannot specify both a global peer (dst/src) and a group-relative peer (group_dst/group_src)"
            )
        return group_peer
    return c10d.get_group_rank(group, peer)


def isend_inplace(tensor, dst=None, tag=0, group=None, group_dst=None):
    if group is None:
        group = dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")
    local_dst = _to_group_local(group, dst, group_dst)
    group_name = _resolve_group_name(group)
    tensor = _c10d_functional.isend(tensor, local_dst, tag, group_name)
    if _are_we_tracing():
        return tensor
    return _maybe_wrap_tensor(tensor)


def irecv_inplace(tensor, src=None, tag=0, group=None, group_src=None):
    if group is None:
        group = dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")
    local_src = _to_group_local(group, src, group_src)
    group_name = _resolve_group_name(group)
    tensor = _c10d_functional.irecv(tensor, local_src, tag, group_name)
    return _maybe_wrap_tensor(tensor)


def apply() -> None:
    """Monkey-patch the functional P2P helpers in-place.

    ``_remapped_isend`` / ``_remapped_irecv`` (the entries Dynamo uses) call
    ``isend_inplace`` / ``irecv_inplace`` by name from this module's globals, so
    replacing the module attributes is picked up on both eager and compiled
    paths.
    """
    funcol.isend_inplace = isend_inplace
    funcol.irecv_inplace = irecv_inplace


apply()
