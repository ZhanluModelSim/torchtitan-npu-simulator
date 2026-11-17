# Backport of upstream TorchTitan PR #4122:
# https://github.com/pytorch/torchtitan/pull/4122
# Upstream commit: 6ee6b10b934730df00cbcd0cab5c93d04ae2e38a
# Remove this patch when the pinned TorchTitan dependency includes PR #4122.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Monkey-patch the PR #4122 optimizer reshard schedule backport."""

from __future__ import annotations

from dataclasses import dataclass
from types import FunctionType
from typing import TYPE_CHECKING

import torchtitan.distributed.flex_shard._optimizer_reshard_schedule as schedule
import torchtitan.distributed.flex_shard.dist_muon as dist_muon

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class _RedistributionGroup:
    process_group: dist.ProcessGroup
    participants: tuple[int, ...]
    mesh_axis_participants: tuple[int, ...]
    local_participant: int


def _tensor_region_intersection_numel(first: _TensorRegion, second: _TensorRegion) -> int:
    intersection = _tensor_region_intersection(first, second)
    return 0 if intersection is None else intersection.numel


def _tensor_region_intersection(
    first: _TensorRegion,
    second: _TensorRegion,
) -> _TensorRegion | None:
    if len(first.shape) != len(second.shape):
        return None
    intersection_offsets = tuple(
        max(first_offset, second_offset)
        for first_offset, second_offset in zip(
            first.offsets,
            second.offsets,
            strict=True,
        )
    )
    intersection_shape = tuple(
        max(
            0,
            min(first_offset + first_size, second_offset + second_size) - max(first_offset, second_offset),
        )
        for first_offset, first_size, second_offset, second_size in zip(
            first.offsets,
            first.shape,
            second.offsets,
            second.shape,
            strict=True,
        )
    )
    if not math.prod(intersection_shape):
        return None
    return _TensorRegion(
        offsets=intersection_offsets,
        shape=intersection_shape,
    )


def _build_dim0_shard_redistribution_plan(
    storage_regions: Sequence[_StorageRegionMapping],
    *,
    participants: tuple[int, ...],
    shard_participants: tuple[int, ...],
    logical_shape: tuple[int, ...],
) -> _RedistributionPlan:
    """Route storage regions to dim-0 compute shards."""
    participant_set = set(participants)
    _require_valid_plan(
        len(shard_participants) == len(participants) and set(shard_participants) == participant_set,
        "shard participants must order the redistribution participants",
    )
    storage_endpoints = []
    storage_by_participant = {}
    for holders, logical_region in storage_regions:
        _require_valid_plan(
            bool(holders) and len(set(holders)) == len(holders) and set(holders) <= participant_set,
            "storage region holders must be unique redistribution participants",
        )
        storage_endpoints.append((holders, logical_region))
        for holder in holders:
            _require_valid_plan(
                holder not in storage_by_participant,
                "multiple storage regions per participant are not supported",
            )
            storage_by_participant[holder] = logical_region
    _require_valid_plan(
        set(storage_by_participant) == participant_set,
        "storage regions must cover every redistribution participant",
    )
    storage_partitions = tuple(
        _ParticipantPartition(
            participant=participant,
            tensor_shape=storage_by_participant[participant].shape,
            logical_regions=(storage_by_participant[participant],),
        )
        for participant in participants
    )

    compute_partitions = []
    storage_to_compute_routes = []
    compute_endpoints = []
    shard_index_by_participant = {participant: index for index, participant in enumerate(shard_participants)}
    for participant in participants:
        participant_index = shard_index_by_participant[participant]
        local_dim0, dim0_offset = Shard.local_shard_size_and_offset(
            logical_shape[0],
            len(participants),
            participant_index,
        )
        local_shape = (local_dim0, *logical_shape[1:])
        logical_region = _TensorRegion(
            offsets=(dim0_offset,) + (0,) * (len(logical_shape) - 1),
            shape=local_shape,
        )
        compute_partitions.append(
            _ParticipantPartition(
                participant=participant,
                tensor_shape=local_shape,
                logical_regions=(logical_region,),
            )
        )
        compute_endpoints.append((participant, logical_region))

    for source_holders, storage_region in storage_endpoints:
        for destination, compute_region in compute_endpoints:
            logical_region = _tensor_region_intersection(
                storage_region,
                compute_region,
            )
            if logical_region is None:
                continue
            storage_tensor_region = _TensorRegion(
                offsets=tuple(
                    logical_offset - storage_offset
                    for logical_offset, storage_offset in zip(
                        logical_region.offsets,
                        storage_region.offsets,
                        strict=True,
                    )
                ),
                shape=logical_region.shape,
            )
            compute_tensor_region = _TensorRegion(
                offsets=tuple(
                    logical_offset - compute_offset
                    for logical_offset, compute_offset in zip(
                        logical_region.offsets,
                        compute_region.offsets,
                        strict=True,
                    )
                ),
                shape=logical_region.shape,
            )
            storage_to_compute_routes.append(
                _TensorRegionRoute(
                    logical_region=logical_region,
                    source=_RouteEndpoint(storage_tensor_region, source_holders),
                    destination=_RouteEndpoint(
                        compute_tensor_region,
                        (destination,),
                    ),
                )
            )

    return _RedistributionPlan(
        participants=participants,
        logical_shape=logical_shape,
        storage_partitions=storage_partitions,
        compute_partitions=tuple(compute_partitions),
        storage_to_compute_routes=tuple(storage_to_compute_routes),
    )


def _redistribution_group(mesh: DeviceMesh) -> _RedistributionGroup:
    process_group = mesh.get_group()
    participants = tuple(dist.get_process_group_ranks(process_group))
    mesh_axis_participants = _device_mesh_ranks(mesh)
    if set(mesh_axis_participants) != set(participants):
        raise ValueError("bucket mesh and process group participants do not match")
    return _RedistributionGroup(
        process_group=process_group,
        participants=participants,
        mesh_axis_participants=mesh_axis_participants,
        local_participant=participants[dist.get_rank(process_group)],
    )


def _dtensor_storage_regions(
    tensor: DTensor,
    participants: tuple[int, ...],
    *,
    required_storage_mesh_axis: int | None,
) -> _DTensorStorageDomain:
    """Return the transport-local shape and holder-to-region mappings."""
    storage_mesh = tensor.device_mesh
    storage_ranks = storage_mesh.mesh
    reference_locations = (storage_ranks == participants[0]).nonzero()
    if tuple(reference_locations.shape) != (1, storage_mesh.ndim):
        raise ValueError("bucket mesh participants must belong to the DTensor storage mesh")

    reference_coordinate = reference_locations[0].tolist()
    matching_storage_mesh_axes = []
    participant_set = set(participants)
    for storage_mesh_axis in range(storage_mesh.ndim):
        coordinate = list(reference_coordinate)
        coordinate[storage_mesh_axis] = slice(None)
        axis_participants = tuple(storage_ranks[tuple(coordinate)].flatten().tolist())
        if len(axis_participants) == len(participants) and set(axis_participants) == participant_set:
            matching_storage_mesh_axes.append(storage_mesh_axis)

    if required_storage_mesh_axis is not None:
        if required_storage_mesh_axis not in matching_storage_mesh_axes:
            raise ValueError("bucket mesh participants do not match the parameter storage shard axis")
        storage_mesh_axis = required_storage_mesh_axis
    elif len(matching_storage_mesh_axes) == 1 or (len(participants) == 1 and matching_storage_mesh_axes):
        storage_mesh_axis = matching_storage_mesh_axes[0]
    else:
        raise ValueError("bucket mesh participants must match exactly one DTensor storage mesh axis")

    transport_placement = tensor.placements[storage_mesh_axis]
    preserved_shard_axis = None
    for mesh_axis, placement in enumerate(tensor.placements):
        if mesh_axis == storage_mesh_axis:
            if type(placement) not in (Replicate, Shard):
                raise ValueError(
                    "redistributed optimizer storage requires exact Shard or Replicate on the communication mesh axis"
                )
        elif storage_mesh.size(mesh_axis) != 1 and type(placement) is not Replicate:
            if not (
                preserved_shard_axis is None
                and type(transport_placement) is Shard
                and type(placement) is Shard
                and transport_placement.dim % tensor.ndim != placement.dim % tensor.ndim
            ):
                raise ValueError(
                    "redistributed optimizer storage requires Replicate outside "
                    "the communication mesh axis, except for one orthogonal "
                    "exact Shard"
                )
            preserved_shard_axis = mesh_axis

    domain_shape = list(tensor.shape)
    domain_offsets = [0] * tensor.ndim
    if preserved_shard_axis is not None:
        placement = cast(Shard, tensor.placements[preserved_shard_axis])
        tensor_dim = placement.dim % tensor.ndim
        local_size, global_offset = Shard.local_shard_size_and_offset(
            tensor.shape[tensor_dim],
            storage_mesh.size(preserved_shard_axis),
            reference_coordinate[preserved_shard_axis],
        )
        domain_shape[tensor_dim] = local_size
        domain_offsets[tensor_dim] = global_offset

    holders_by_region: dict[_TensorRegion, list[int]] = {}
    for participant in participants:
        global_region = _dtensor_storage_region_for_participant(tensor, participant)
        region = _TensorRegion(
            offsets=tuple(
                offset - domain_offset
                for offset, domain_offset in zip(
                    global_region.offsets,
                    domain_offsets,
                    strict=True,
                )
            ),
            shape=global_region.shape,
        )
        holders_by_region.setdefault(region, []).append(participant)
    regions = tuple((tuple(holders), region) for region, holders in holders_by_region.items())
    _validate_tensor_region_partition(
        tuple(region for _holders, region in regions),
        tuple(domain_shape),
        direction="transport-subgroup storage",
    )
    return tuple(domain_shape), regions


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


def apply() -> None:
    schedule._RedistributionGroup = _RedistributionGroup
    for function in (
        _tensor_region_intersection_numel,
        _tensor_region_intersection,
        _build_dim0_shard_redistribution_plan,
        _redistribution_group,
        _dtensor_storage_regions,
    ):
        setattr(schedule, function.__name__, _rebind(function, schedule.__dict__))
    # dist_muon imported these symbols by value before this package patch ran.
    dist_muon._RedistributionGroup = _RedistributionGroup
    dist_muon._build_dim0_shard_redistribution_plan = schedule._build_dim0_shard_redistribution_plan
    dist_muon._dtensor_storage_regions = schedule._dtensor_storage_regions


apply()
