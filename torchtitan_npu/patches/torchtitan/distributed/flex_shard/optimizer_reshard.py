# Backport of upstream TorchTitan PR #4122:
# https://github.com/pytorch/torchtitan/pull/4122
# Upstream commit: 6ee6b10b934730df00cbcd0cab5c93d04ae2e38a
# Remove this patch when the pinned TorchTitan dependency includes PR #4122.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Monkey-patch the PR #4122 ComputeLayout backport."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import torchtitan.distributed.flex_shard as flex_shard
import torchtitan.distributed.flex_shard.dist_muon as dist_muon
import torchtitan.distributed.flex_shard.optimizer_reshard as optimizer_reshard
from torch.distributed.tensor import Replicate, Shard

Owned = optimizer_reshard.Owned
BlockShard = optimizer_reshard.BlockShard
_ComputeSharding = Owned | Replicate | Shard | BlockShard
_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class _FrozenConfigMapping(Mapping[_KeyT, _ValueT]):
    """Small immutable mapping that remains safe to copy with configs.

    ``ComputeLayout`` normalizes both of its mapping fields into this type:
    ``shardings_by_mesh_axis`` keyed by mesh axis name, and
    ``shard_order_by_tensor_dim`` keyed by tensor dimension. Both must stay
    hashable and must survive a deep copy of the surrounding config without
    becoming a mutable alias.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Mapping[_KeyT, _ValueT]) -> None:
        self._items = tuple(items.items())

    def __getitem__(self, key: _KeyT) -> _ValueT:
        for candidate_key, value in self._items:
            if candidate_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[_KeyT]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __hash__(self) -> int:
        return hash(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenConfigMapping[_KeyT, _ValueT]:
        return self


_ShardingsByMeshAxis = _FrozenConfigMapping[str, _ComputeSharding]

_ShardOrderByTensorDim = _FrozenConfigMapping[int, tuple[str, ...]]

_DEFAULT_SHARD_ORDER = _ShardOrderByTensorDim({})


@dataclass(frozen=True, slots=True)
class ComputeLayout:
    """Describe temporary compute shardings on named storage-mesh axes.

    These shardings apply before any optimizer-local tensor view.

    Each named axis uses one of four compute shardings: ``Owned`` assigns the
    complete subgroup-local logical tensor to one dynamically selected owner rank,
    ``Replicate`` assigns it to every rank, ``Shard`` partitions one tensor
    dimension, and ``BlockShard`` partitions fixed-size blocks without changing
    the tensor shape.
    Multiple ``Owned`` shardings select one owner rank from their joint Cartesian
    group.
    Omitted axes preserve their storage placement. Extra declarations may target
    mesh variants not used by a particular parameter, but at least one declaration
    must apply when the layout is resolved.

    ``shard_order_by_tensor_dim`` maps a tensor dimension to the mesh axes that
    shard it, outermost first: the leading axis partitions the whole dimension
    and every following axis partitions its predecessor's shard. Declaring an
    order is only necessary when it differs from the default, which applies the
    axes in storage-mesh order. Each named axis must declare ``Shard`` on that
    same tensor dimension, written as a non-negative index.

    Examples:
        Shard a logical batch of matrices over EP first, then split each
        EP-local batch over EFSDP even though EFSDP precedes EP in the storage
        mesh::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "efsdp": Shard(0),
                    "ep": Shard(0),
                },
                shard_order_by_tensor_dim={0: ("ep", "efsdp")},
            )

        Assign the complete subgroup-local logical tensor to one owner rank
        along ``dp_shard``::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": Owned(),
                }
            )

        Shard complete four-row blocks along tensor dimension 0::

            ComputeLayout(
                shardings_by_mesh_axis={
                    "dp_shard": BlockShard(dim=0, block_size=4),
                }
            )
    """

    shardings_by_mesh_axis: Mapping[str, _ComputeSharding]
    shard_order_by_tensor_dim: Mapping[int, tuple[str, ...]] = _DEFAULT_SHARD_ORDER

    def __post_init__(self) -> None:
        shardings_by_mesh_axis = dict(self.shardings_by_mesh_axis)
        if not shardings_by_mesh_axis:
            raise ValueError("ComputeLayout must declare a compute sharding")
        for axis_name, sharding in shardings_by_mesh_axis.items():
            if not isinstance(axis_name, str):
                raise ValueError("ComputeLayout.shardings_by_mesh_axis keys must be strings")
            if type(sharding) not in (Owned, Replicate, Shard, BlockShard):
                raise ValueError(
                    "ComputeLayout.shardings_by_mesh_axis values must be Owned, Replicate, Shard, or BlockShard"
                )
        normalized_shardings_by_mesh_axis = dict(sorted(shardings_by_mesh_axis.items()))
        object.__setattr__(
            self,
            "shardings_by_mesh_axis",
            _ShardingsByMeshAxis(normalized_shardings_by_mesh_axis),
        )
        object.__setattr__(
            self,
            "shard_order_by_tensor_dim",
            _ShardOrderByTensorDim(self._validated_shard_order(normalized_shardings_by_mesh_axis)),
        )

    def _validated_shard_order(
        self,
        shardings_by_mesh_axis: Mapping[str, _ComputeSharding],
    ) -> dict[int, tuple[str, ...]]:
        validated_shard_order: dict[int, tuple[str, ...]] = {}
        for tensor_dim, axis_names in dict(self.shard_order_by_tensor_dim).items():
            if isinstance(tensor_dim, bool) or not isinstance(tensor_dim, int) or tensor_dim < 0:
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim keys must be "
                    f"non-negative tensor dimensions; got {tensor_dim!r}"
                )
            if isinstance(axis_names, str) or not isinstance(axis_names, Sequence):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim values must be "
                    f"sequences of mesh axis names; got {axis_names!r}"
                )
            ordered_axis_names = tuple(axis_names)
            if len(ordered_axis_names) < 2:
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim must order at least "
                    f"two mesh axes; tensor dimension {tensor_dim} lists "
                    f"{list(ordered_axis_names)}"
                )
            if len(set(ordered_axis_names)) != len(ordered_axis_names):
                raise ValueError(
                    "ComputeLayout.shard_order_by_tensor_dim must not repeat a "
                    f"mesh axis; tensor dimension {tensor_dim} lists "
                    f"{list(ordered_axis_names)}"
                )
            for axis_name in ordered_axis_names:
                sharding = shardings_by_mesh_axis.get(axis_name)
                if sharding is None:
                    raise ValueError(
                        "ComputeLayout.shard_order_by_tensor_dim names mesh axis "
                        f"{axis_name!r}, which shardings_by_mesh_axis does not "
                        "declare"
                    )
                if type(sharding) is not Shard or sharding.dim != tensor_dim:
                    raise ValueError(
                        "ComputeLayout.shard_order_by_tensor_dim requires "
                        f"Shard({tensor_dim}) on mesh axis {axis_name!r}; got "
                        f"{sharding!r}"
                    )
            validated_shard_order[tensor_dim] = ordered_axis_names
        return dict(sorted(validated_shard_order.items()))

    def to_dict(self) -> dict:
        """Serialize for JSON logging."""
        return {"repr": repr(self)}


def apply() -> None:
    optimizer_reshard.ComputeLayout = ComputeLayout
    flex_shard.ComputeLayout = ComputeLayout
    dist_muon.ComputeLayout = ComputeLayout


apply()
