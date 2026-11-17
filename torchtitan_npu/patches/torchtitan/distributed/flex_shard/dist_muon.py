# Backport of upstream TorchTitan PR #4122:
# https://github.com/pytorch/torchtitan/pull/4122
# Upstream commit: 6ee6b10b934730df00cbcd0cab5c93d04ae2e38a
# NPU admission issue: https://github.com/pytorch/torchtitan/issues/4405
# Remove this patch when the pinned TorchTitan dependency includes PR #4122.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Monkey-patch the PR #4122 DistMuon backport and NPU storage admission."""

from __future__ import annotations

from types import FunctionType
from typing import TYPE_CHECKING

import torch
import torchtitan.distributed.flex_shard.dist_muon as dist_muon
import torchtitan.distributed.flex_shard.optimizer_reshard as optimizer_reshard
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

Owned = optimizer_reshard.Owned
BlockShard = optimizer_reshard.BlockShard


def _build_parameter_redistribution_plan(
    compute_layout: _ParameterComputeLayout,
    group: _RedistributionGroup,
    owner_rank: int | None,
) -> _RedistributionPlan | None:
    transition = compute_layout.storage_to_compute_transition
    if isinstance(transition, _NoRedistributionTransition):
        return None
    assert isinstance(transition, _RedistributionTransition)

    group_local_storage_shape, storage_regions = _dtensor_storage_regions(
        compute_layout.param,
        group.participants,
        required_storage_mesh_axis=(compute_layout.redistribution_storage_mesh_axis),
    )
    compute_sharding = compute_layout.compute_sharding
    if type(compute_sharding) is Owned:
        assert owner_rank is not None
        assert owner_rank in group.participants
        return _build_owned_redistribution_plan(
            storage_regions,
            participants=group.participants,
            owner_rank=owner_rank,
            logical_shape=tuple(compute_layout.param.shape),
        )

    assert owner_rank is None
    assert type(compute_sharding) is Shard
    if tuple(compute_layout.global_compute_shape) == tuple(compute_layout.param.shape):
        return _build_dim0_shard_redistribution_plan(
            storage_regions,
            participants=group.participants,
            shard_participants=group.mesh_axis_participants,
            logical_shape=group_local_storage_shape,
        )

    return _build_batched_matrix_redistribution_plan(
        storage_regions,
        participants=group.participants,
        storage_shape=tuple(compute_layout.param.shape),
        compute_shape=tuple(compute_layout.global_compute_shape),
    )


def _lower_shard_order_to_strided_shards(
    param: DTensor,
    compute_layout: ComputeLayout,
    storage_axis_by_name: Mapping[str, int],
    shardings_by_storage_mesh_axis: Mapping[int, _LoweredComputeSharding],
) -> dict[int, _LoweredComputeSharding]:
    """Encode a declared shard order as DTensor placements.

    DTensor applies same-dimension shard axes in storage-mesh order, so an axis
    that the compute layout applies later than that default becomes a
    ``_StridedShard`` whose split factor is the product of the mesh sizes of the
    axes it now follows but that sit to its right in the storage mesh. Axes the
    layout declares for other mesh variants are absent here and drop out of the
    order.
    """
    lowered_shardings = dict(shardings_by_storage_mesh_axis)
    for tensor_dim, axis_names in compute_layout.shard_order_by_tensor_dim.items():
        ordered_mesh_axes = [
            storage_axis_by_name[axis_name] for axis_name in axis_names if axis_name in storage_axis_by_name
        ]
        for order_index, storage_mesh_axis in enumerate(ordered_mesh_axes):
            split_factor = math.prod(
                param.device_mesh.size(preceding_mesh_axis)
                for preceding_mesh_axis in ordered_mesh_axes[:order_index]
                if preceding_mesh_axis > storage_mesh_axis
            )
            if split_factor > 1:
                lowered_shardings[storage_mesh_axis] = _StridedShard(tensor_dim, split_factor=split_factor)
    return lowered_shardings


def _validate_shard_order_compute_targets(
    fqn: str,
    param: DTensor,
    target_sharding_by_storage_mesh_axis: Mapping[int, _AxisComputeSharding],
    owned_storage_mesh_axes: Sequence[int],
) -> None:
    """Reject declared shard orders that DistMuon cannot lower to a transport plan.

    DistMuon supports reordering a mesh axis behind exactly one later mesh axis
    that shards the same tensor dimension, because that axis is the one whose
    storage ownership the redistribution preserves.
    """
    mesh_axis_names = param.device_mesh.mesh_dim_names
    assert mesh_axis_names is not None
    owned_axis_set = set(owned_storage_mesh_axes)
    for (
        storage_mesh_axis,
        target_sharding,
    ) in target_sharding_by_storage_mesh_axis.items():
        if type(target_sharding) is not _StridedShard:
            continue
        target_dim = _normalize_dim(target_sharding.dim, param.ndim)
        rightward_shard_axes = []
        for rightward_mesh_axis in range(
            storage_mesh_axis + 1,
            param.device_mesh.ndim,
        ):
            if rightward_mesh_axis in owned_axis_set:
                continue
            rightward_sharding = target_sharding_by_storage_mesh_axis.get(rightward_mesh_axis)
            if rightward_sharding is None:
                rightward_sharding = _normalize_storage_placement(
                    param.placements[rightward_mesh_axis],
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(rightward_mesh_axis),
                )
            if type(rightward_sharding) is Shard and _normalize_dim(rightward_sharding.dim, param.ndim) == target_dim:
                rightward_shard_axes.append(rightward_mesh_axis)

        if len(rightward_shard_axes) != 1:
            raise ValueError(
                f"Muon parameter {fqn!r} orders mesh axis "
                f"{mesh_axis_names[storage_mesh_axis]!r} after another axis on "
                f"tensor dimension {target_dim}; DistMuon requires exactly one "
                "later mesh axis to shard that dimension"
            )
        preserved_mesh_axis_size = param.device_mesh.size(rightward_shard_axes[0])
        if target_sharding.split_factor != preserved_mesh_axis_size:
            raise ValueError(
                f"Muon parameter {fqn!r} must order mesh axis "
                f"{mesh_axis_names[storage_mesh_axis]!r} directly after "
                f"{mesh_axis_names[rightward_shard_axes[0]]!r} on tensor "
                f"dimension {target_dim}; DistMuon does not support ordering it "
                "after further mesh axes"
            )


def _is_supported_orthogonal_dim0_shard_redistribution(
    *,
    ndim: int,
    compute_view: _MatrixBatchView | None,
    source_storage_placement: object,
    target_compute_sharding: _AxisComputeSharding | None,
    preserved_storage_placement: object,
) -> bool:
    if compute_view is not None or ndim != 3:
        return False
    if (
        type(source_storage_placement) is not Shard
        or type(target_compute_sharding) is not _StridedShard
        or type(preserved_storage_placement) is not Shard
    ):
        return False

    storage_dim = _normalize_dim(source_storage_placement.dim, ndim)
    compute_dim = _normalize_dim(target_compute_sharding.dim, ndim)
    preserved_dim = _normalize_dim(preserved_storage_placement.dim, ndim)
    return storage_dim == 1 and compute_dim == preserved_dim == 0


def _resolve_storage_to_compute_transition(
    fqn: str,
    param: DTensor,
    global_compute_shape: torch.Size,
    compute_view: _MatrixBatchView | None,
    compute_layout: ComputeLayout,
) -> _ResolvedStorageToComputeTransition:
    """Validate one storage layout and resolve its concrete compute transition."""
    local = param.to_local()
    if (
        len(global_compute_shape) not in (2, 3)
        or torch.is_complex(param)
        or param.ndim < 2
        or not local.is_contiguous()
    ):
        _raise_unsupported_layout(fqn)

    mesh_axis_names = param.device_mesh.mesh_dim_names
    if mesh_axis_names is None:
        raise ValueError(f"Muon parameter {fqn!r} requires a storage mesh with named axes")
    storage_axis_by_name = {axis_name: storage_mesh_axis for storage_mesh_axis, axis_name in enumerate(mesh_axis_names)}
    applicable_compute_shardings_by_storage_mesh_axis: dict[int, _LoweredComputeSharding] = {
        storage_axis_by_name[axis_name]: sharding
        for axis_name, sharding in compute_layout.shardings_by_mesh_axis.items()
        if axis_name in storage_axis_by_name
    }
    if not applicable_compute_shardings_by_storage_mesh_axis:
        declared_axes = sorted(compute_layout.shardings_by_mesh_axis)
        raise ValueError(
            f"Muon compute layout for parameter {fqn!r} declares no axis in "
            f"storage mesh {list(mesh_axis_names)}; declared axes: {declared_axes}"
        )
    applicable_compute_shardings_by_storage_mesh_axis = _lower_shard_order_to_strided_shards(
        param,
        compute_layout,
        storage_axis_by_name,
        applicable_compute_shardings_by_storage_mesh_axis,
    )

    applicable_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis, sharding in (applicable_compute_shardings_by_storage_mesh_axis.items())
        if type(sharding) is Owned
    )

    if compute_view is not None:
        if applicable_owned_storage_mesh_axes:
            raise ValueError(f"Muon owned compute for parameter {fqn!r} requires a 2D matrix")
        shard_axes = [
            mesh_axis_names[storage_mesh_axis]
            for storage_mesh_axis, sharding in (applicable_compute_shardings_by_storage_mesh_axis.items())
            if type(sharding) in (Shard, _StridedShard)
        ]
        if shard_axes:
            raise ValueError(
                f"Muon parameter {fqn!r} with matrix-batch compute requires "
                f"BlockShard instead of Shard on mesh axes {shard_axes}"
            )

    replicated_axes = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, sharding in (applicable_compute_shardings_by_storage_mesh_axis.items())
        if type(sharding) is Replicate
    ]
    if replicated_axes:
        raise NotImplementedError(
            f"Muon parameter {fqn!r} requests explicit replicated compute on "
            f"mesh axes {replicated_axes}; replicated compute is not implemented"
        )

    normalized_target_sharding_by_storage_mesh_axis: dict[int, _AxisComputeSharding] = {}
    declared_shard_dims = []
    for (
        storage_mesh_axis,
        sharding,
    ) in applicable_compute_shardings_by_storage_mesh_axis.items():
        if type(sharding) is Owned:
            continue
        placement = cast(_AxisComputeSharding, sharding)
        if type(placement) is Shard or type(placement) is _StridedShard:
            declared_shard_dims.append(_normalize_dim(placement.dim, param.ndim))
        elif type(placement) is BlockShard:
            declared_shard_dims.append(0)
        target_sharding = _normalize_compute_placement(
            placement,
            ndim=param.ndim,
            mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
        )
        normalized_target_sharding_by_storage_mesh_axis[storage_mesh_axis] = target_sharding

    _validate_shard_order_compute_targets(
        fqn,
        param,
        normalized_target_sharding_by_storage_mesh_axis,
        applicable_owned_storage_mesh_axes,
    )
    changed_storage_mesh_axes = _resolve_storage_to_compute_redistribution_requirement(
        fqn,
        param,
        compute_view,
        normalized_target_sharding_by_storage_mesh_axis,
        tuple(applicable_compute_shardings_by_storage_mesh_axis),
    )
    active_owned_storage_mesh_axes = tuple(
        storage_mesh_axis
        for storage_mesh_axis in applicable_owned_storage_mesh_axes
        if param.device_mesh.size(storage_mesh_axis) > 1
    )
    transport_mesh_axes = tuple(sorted(set(changed_storage_mesh_axes).union(active_owned_storage_mesh_axes)))
    if len(transport_mesh_axes) > 1:
        axis_names = [mesh_axis_names[axis] for axis in transport_mesh_axes]
        raise NotImplementedError(
            f"Muon parameter {fqn!r} requires compute redistribution or "
            f"owned compute on multiple mesh axes {axis_names}; multi-axis "
            "transport is not implemented"
        )

    redistribution_storage_mesh_axis = transport_mesh_axes[0] if transport_mesh_axes else None
    uses_supported_orthogonal_shard_redistribution = False
    if redistribution_storage_mesh_axis is not None:
        redistribution_axis_name = mesh_axis_names[redistribution_storage_mesh_axis]
        for storage_mesh_axis, placement in enumerate(param.placements):
            if storage_mesh_axis == redistribution_storage_mesh_axis:
                if type(placement) not in (Replicate, Shard):
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute "
                        f"{type(placement).__name__} storage on mesh axis "
                        f"{redistribution_axis_name!r}"
                    )
            else:
                preserved_storage_sharding = _normalize_storage_placement(
                    placement,
                    ndim=param.ndim,
                    mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
                )
                if type(preserved_storage_sharding) is Replicate:
                    continue
                redistribution_storage_placement = param.placements[redistribution_storage_mesh_axis]
                redistribution_compute_sharding = normalized_target_sharding_by_storage_mesh_axis.get(
                    redistribution_storage_mesh_axis
                )
                if (
                    not uses_supported_orthogonal_shard_redistribution
                    and _is_supported_orthogonal_dim0_shard_redistribution(
                        ndim=param.ndim,
                        compute_view=compute_view,
                        source_storage_placement=redistribution_storage_placement,
                        target_compute_sharding=redistribution_compute_sharding,
                        preserved_storage_placement=placement,
                    )
                ):
                    uses_supported_orthogonal_shard_redistribution = True
                    continue
                if (
                    type(redistribution_storage_placement) is Shard
                    and type(redistribution_compute_sharding) in (Shard, _StridedShard, BlockShard)
                    and type(placement) is Shard
                ):
                    target_compute_shard = cast(
                        Shard | _StridedShard | BlockShard,
                        redistribution_compute_sharding,
                    )
                    storage_dim = _normalize_dim(redistribution_storage_placement.dim, param.ndim)
                    target_dim = _normalize_dim(target_compute_shard.dim, param.ndim)
                    preserved_dim = _normalize_dim(placement.dim, param.ndim)
                    if (
                        storage_dim == 1
                        and target_dim == preserved_dim == 0
                        and type(redistribution_compute_sharding) is Shard
                        and redistribution_storage_mesh_axis < storage_mesh_axis
                    ):
                        preserved_axis_name = mesh_axis_names[storage_mesh_axis]
                        raise ValueError(
                            f"Muon parameter {fqn!r} must declare "
                            "shard_order_by_tensor_dim={0: "
                            f"({preserved_axis_name!r}, "
                            f"{redistribution_axis_name!r})}} when preserving "
                            f"Shard(0) on mesh axis {preserved_axis_name!r}"
                        )
                    raise NotImplementedError(
                        f"Muon parameter {fqn!r} cannot redistribute storage on "
                        f"mesh axis {redistribution_axis_name!r} from "
                        f"Shard({storage_dim}) to "
                        f"{redistribution_compute_sharding!r} while "
                        f"preserving Shard({preserved_dim}) storage on mesh axis "
                        f"{mesh_axis_names[storage_mesh_axis]!r}; orthogonal-shard "
                        "redistribution is not implemented"
                    )
                raise NotImplementedError(
                    f"Muon parameter {fqn!r} cannot redistribute mesh axis "
                    f"{redistribution_axis_name!r} while storage mesh axis "
                    f"{mesh_axis_names[storage_mesh_axis]!r} has non-replicated "
                    f"placement {placement}; this implementation requires every "
                    "other storage mesh axis to be replicated"
                )

    reordered_axis_names = [
        mesh_axis_names[storage_mesh_axis]
        for storage_mesh_axis, target_sharding in (normalized_target_sharding_by_storage_mesh_axis.items())
        if type(target_sharding) is _StridedShard
    ]
    if (
        reordered_axis_names
        and redistribution_storage_mesh_axis is not None
        and not uses_supported_orthogonal_shard_redistribution
    ):
        raise ValueError(
            f"Muon parameter {fqn!r} has an unsupported shard order on mesh "
            f"axes {reordered_axis_names}; DistMuon currently reorders a mesh "
            "axis only when a preceding redistribution axis preserves one "
            "rightward Shard axis on the same tensor dimension"
        )

    resolved_target_signature = []
    resolved_shard_dims = []
    owned_axis_set = set(applicable_owned_storage_mesh_axes)
    for storage_mesh_axis, axis_name in enumerate(mesh_axis_names):
        if storage_mesh_axis in owned_axis_set:
            target_sharding: Owned | _AxisComputeSharding | _UnsupportedStoragePlacement = Owned()
        elif storage_mesh_axis in normalized_target_sharding_by_storage_mesh_axis:
            target_sharding = normalized_target_sharding_by_storage_mesh_axis[storage_mesh_axis]
        else:
            target_sharding = _normalize_storage_placement(
                param.placements[storage_mesh_axis],
                ndim=param.ndim,
                mesh_axis_size=param.device_mesh.size(storage_mesh_axis),
            )
        if type(target_sharding) is _UnsupportedStoragePlacement:
            raise NotImplementedError(
                f"Muon parameter {fqn!r} has unsupported compute placement "
                f"{target_sharding.type_name!r} "
                f"({target_sharding.representation}) on mesh axis {axis_name!r}"
            )
        resolved_target_signature.append((axis_name, target_sharding))
        if type(target_sharding) is Shard or type(target_sharding) is _StridedShard:
            resolved_shard_dims.append(target_sharding.dim)
        elif type(target_sharding) is BlockShard:
            resolved_shard_dims.append(0)

    resolved_compute_layout_signature = tuple(resolved_target_signature)
    compute_shard_dims = [*resolved_shard_dims, *declared_shard_dims]
    if applicable_owned_storage_mesh_axes and (len(global_compute_shape) != 2 or param.ndim != 2):
        raise ValueError(f"Muon owned compute for parameter {fqn!r} requires a 2D matrix")
    if active_owned_storage_mesh_axes:
        compute_sharding: _ResolvedComputeSharding = Owned()
    elif compute_shard_dims:
        if compute_view is None and len(global_compute_shape) == 2:
            raise ValueError(
                f"Muon parameter {fqn!r}: 2D Muon compute cannot use Shard; "
                "use Owned() for one matrix or "
                "BlockShard(dim=0, block_size=R) for row-concatenated matrices"
            )
        if len(global_compute_shape) != 3 or any(shard_dim != 0 for shard_dim in compute_shard_dims):
            raise ValueError(
                f"Muon sharded compute for parameter {fqn!r} requires a 3D "
                "batch-first tensor sharded only on tensor dimension 0"
            )
        compute_sharding = Shard(0)
    elif applicable_owned_storage_mesh_axes:
        compute_sharding = Owned()
    else:
        raise ValueError(f"unsupported storage-to-compute layout for {fqn!r}")

    if redistribution_storage_mesh_axis is not None and type(compute_sharding) is Shard:
        source_sharding = _normalize_storage_placement(
            param.placements[redistribution_storage_mesh_axis],
            ndim=param.ndim,
            mesh_axis_size=param.device_mesh.size(redistribution_storage_mesh_axis),
        )
        target_sharding = normalized_target_sharding_by_storage_mesh_axis[redistribution_storage_mesh_axis]
        if (
            type(source_sharding) is not Replicate
            and type(target_sharding) is not BlockShard
            and source_sharding != target_sharding
            and not uses_supported_orthogonal_shard_redistribution
        ):
            axis_name = mesh_axis_names[redistribution_storage_mesh_axis]
            raise NotImplementedError(
                f"Muon parameter {fqn!r} cannot yet change tensor sharding "
                f"from {source_sharding} to {target_sharding} on mesh axis "
                f"{axis_name!r}"
            )

    if redistribution_storage_mesh_axis is None:
        return _ResolvedStorageToComputeTransition(
            compute_sharding=compute_sharding,
            storage_to_compute_transition=_NoRedistributionTransition(),
            resolved_compute_layout_signature=resolved_compute_layout_signature,
        )

    return _ResolvedStorageToComputeTransition(
        compute_sharding=compute_sharding,
        storage_to_compute_transition=_RedistributionTransition(),
        resolved_compute_layout_signature=resolved_compute_layout_signature,
        redistribution_storage_mesh_axis=redistribution_storage_mesh_axis,
    )


def _normalize_compute_placement(
    placement: _AxisComputeSharding,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> _AxisComputeSharding:
    if type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        return Shard(normalized_dim)
    if type(placement) is _StridedShard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if mesh_axis_size == 1:
            return Replicate()
        if placement.split_factor == 1:
            return Shard(normalized_dim)
        return _StridedShard(
            normalized_dim,
            split_factor=placement.split_factor,
        )
    assert type(placement) is BlockShard
    normalized_dim = _normalize_dim(placement.dim, ndim)
    if mesh_axis_size == 1:
        return Replicate()
    return BlockShard(normalized_dim, placement.block_size)


def _normalize_storage_placement(
    placement: object,
    *,
    ndim: int,
    mesh_axis_size: int,
) -> Replicate | Shard | _StridedShard | _UnsupportedStoragePlacement:
    if mesh_axis_size == 1 or type(placement) is Replicate:
        return Replicate()
    if type(placement) is Shard:
        return Shard(_normalize_dim(placement.dim, ndim))
    if type(placement) is _StridedShard:
        normalized_dim = _normalize_dim(placement.dim, ndim)
        if placement.split_factor == 1:
            return Shard(normalized_dim)
        return _StridedShard(
            normalized_dim,
            split_factor=placement.split_factor,
        )
    return _UnsupportedStoragePlacement(
        type_name=type(placement).__name__,
        representation=repr(placement),
    )


def _rebind(function, namespace):
    rebound = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebound.__annotations__ = function.__annotations__
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__doc__ = function.__doc__
    return rebound


def _validate_parameter_storage(self) -> torch.device:
    local_devices = set()
    for group in self.param_groups:
        for param in group["params"]:
            if not isinstance(param, DTensor):
                raise TypeError("DistMuon requires DTensor parameters")
            local_device = param.to_local().device
            if local_device.type != "npu":
                raise ValueError("DistMuon requires NPU parameters")
            local_devices.add(local_device)
    if len(local_devices) != 1:
        raise ValueError("DistMuon requires one device per process")
    return local_devices.pop()


def apply() -> None:
    dist_muon._LoweredComputeSharding = Owned | Replicate | Shard | _StridedShard | BlockShard
    dist_muon._AxisComputeSharding = Replicate | Shard | _StridedShard | BlockShard
    for function in (
        _build_parameter_redistribution_plan,
        _lower_shard_order_to_strided_shards,
        _validate_shard_order_compute_targets,
        _is_supported_orthogonal_dim0_shard_redistribution,
        _resolve_storage_to_compute_transition,
        _normalize_compute_placement,
        _normalize_storage_placement,
    ):
        setattr(dist_muon, function.__name__, _rebind(function, dist_muon.__dict__))
    dist_muon.DistMuon._validate_parameter_storage = _validate_parameter_storage


apply()
