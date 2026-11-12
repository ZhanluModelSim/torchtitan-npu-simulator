# Original backport upstream PR: https://github.com/pytorch/torchtitan/pull/4897
# Original backport upstream PR: https://github.com/pytorch/torchtitan/pull/4617
# v0.4 uses native metadata handling. Only the inactive legacy CP helpers remain
# until the NPU CP path migrates to the native ParallelismContext protocol.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy CP helpers retained for the deferred NPU CP path."""

from collections.abc import Hashable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch

AttentionMetadataMap = Mapping[Hashable, Any]


@dataclass(frozen=True)
class CPShardContext:
    """What a backend's CP transform reads here (upstream: ``ParallelismContext``).

    Upstream's ``prepare_cp_metadata`` takes the permutation alone because the
    mesh and the load balancer are ambient in the ``ParallelismContext`` its
    model is called with.  0.3.0 has neither, so ``Decoder._cp_shard`` publishes
    the three things its pinned CP entry cannot recover -- the mesh it sharded
    with, the load-balancer type it sharded by, and the *pre-shard* positions
    the rank-local copy is derived from -- for the duration of the transform.
    """

    cp_mesh: Any
    load_balancer_type: str | None
    positions: torch.Tensor | None


_CP_SHARD_CONTEXT: ContextVar[CPShardContext | None] = ContextVar(
    "torchtitan_npu_cp_shard_context",
    default=None,
)


def cp_shard_context() -> CPShardContext:
    """The context ``Decoder._cp_shard`` published for this transform."""
    context = _CP_SHARD_CONTEXT.get()
    if context is None:
        raise RuntimeError("prepare_cp_metadata reads the pinned shard context, which only Decoder._cp_shard publishes")
    return context


@contextmanager
def cp_shard_context_set(
    *,
    cp_mesh: Any,
    load_balancer_type: str | None,
    positions: torch.Tensor | None,
):
    """Publish the pinned shard context around one metadata transform."""
    token = _CP_SHARD_CONTEXT.set(CPShardContext(cp_mesh, load_balancer_type, positions))
    try:
        yield
    finally:
        _CP_SHARD_CONTEXT.reset(token)


def cp_layout_of(attention_metadata: AttentionMetadataMap | None) -> str | None:
    """The layout these metadata describe, when the run names none (``cp_layout``).

    A stack's rank-local metadata exists for one layout only: the frames it carries address one
    arrangement of the row.  Declaring it on the metadata class lets a run leave
    ``--parallelism.context-parallel-load-balancer`` unset and still be sharded by the layout its
    own attention reads.  A stack that declares nothing keeps the pinned behaviour, and asking for
    a layout the metadata cannot describe is refused by the transform itself.
    """
    for metadata in (attention_metadata or {}).values():
        layout = getattr(type(metadata), "cp_layout", None)
        if layout is not None:
            return layout
    return None


def resolve_cp_layout(load_balancer_type: str | None, attention_metadata: AttentionMetadataMap | None) -> str | None:
    """The layout this run shards by: the configured one, or the stack's declared default.

    Every ``cp_shard`` call site goes through this, so a run that leaves the flag unset shards
    *every* replicated tensor by the same layout -- the model's own inputs and the MTP depths
    alike.  Resolving it at one call site only would silently cut the other one contiguously.
    """
    return load_balancer_type or cp_layout_of(attention_metadata)


def prepare_cp_metadata_map(
    attention_metadata: AttentionMetadataMap,
    *,
    permutation: torch.Tensor | None,
) -> dict[Hashable, Any]:
    """Rank-local metadata for context parallel (upstream #4617's CP half).

    One rule, per map entry: a metadata that carries ``prepare_cp_metadata`` is rebuilt by
    it, and every entry that does not passes through unchanged (the Ulysses case, where the
    metadata stays global).  A backend that cannot be sharded raises from its own metadata's
    transform rather than letting a layer read a layout its rank does not hold.

    Upstream hosts the hook on the backend class and finds it through the map key.  Here it
    lives on the metadata class, because that is the object being rebuilt -- same signature,
    same staticmethod shape, one owner -- and because it makes the key pure addressing:
    anything hashable may address an entry, and no key has to look like a class.

    The transform is handed the permutation alone, as upstream does; the pinned
    version has no ambient parallelism context, so the mesh, the load-balancer
    type and the pre-shard positions come from :func:`cp_shard_context`, which
    ``Decoder._cp_shard`` publishes around this call.
    """
    prepared: dict[Hashable, Any] = {}
    for key, metadata in attention_metadata.items():
        hook = getattr(type(metadata), "prepare_cp_metadata", None)
        prepared[key] = metadata if hook is None else hook(metadata, permutation=permutation)
    return prepared
