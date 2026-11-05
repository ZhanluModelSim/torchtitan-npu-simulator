# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""Equivalent metadata planning for the pinned TorchTitan FlexShard implementation.

This module has no activation side effects. Caches live only within one bucket
planning call. Timing observations are host wall time, never simulated costs.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import TYPE_CHECKING, cast

import torchtitan.distributed.flex_shard._optimizer_reshard_schedule as schedule
import torchtitan.distributed.flex_shard.dist_muon as muon
from torch.distributed.tensor import Replicate, Shard

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor

# Captured after package compatibility patches and before extension activation.
# Kept immutable as the pinned baseline for differential validation.
NATIVE_BUILD_BUCKET_PLANS = schedule._build_bucket_plans
NATIVE_VALIDATE_PLAN = schedule._validate_redistribution_plan
NATIVE_VALIDATE_REGIONS = schedule._validate_disjoint_regions_in_bounds
NATIVE_PACK = schedule._build_packed_all_to_all_schedule
NATIVE_STORAGE_REGION = schedule._dtensor_storage_region_for_participant
NATIVE_STORAGE_REGIONS = schedule._dtensor_storage_regions
NATIVE_RESOLVE_PLANS = muon._resolve_muon_redistribution_plans


@dataclass
class PlanningTiming:
    """Inclusive elapsed seconds and number of invocations of a planning stage."""

    wall_seconds: float = 0.0
    calls: int = 0


_observer: ContextVar[dict[str, PlanningTiming] | None] = ContextVar("muon_planning_timings", default=None)
_mesh_coordinates: ContextVar[dict[int, tuple[DeviceMesh, dict[int, tuple[int, ...]]]] | None] = ContextVar(
    "muon_planning_coordinates", default=None
)


@contextmanager
def collect_planning_timings() -> Iterator[dict[str, PlanningTiming]]:
    """Collect inclusive host timings in this context; nested stages must not be summed."""
    observations: dict[str, PlanningTiming] = {}
    token = _observer.set(observations)
    try:
        yield observations
    finally:
        _observer.reset(token)


def _observed[ResultT](name: str, call: Callable[[], ResultT]) -> ResultT:
    observations = _observer.get()
    if observations is None:
        return call()
    start = perf_counter()
    try:
        return call()
    finally:
        observation = observations.setdefault(name, PlanningTiming())
        observation.wall_seconds += perf_counter() - start
        observation.calls += 1


def validate_disjoint_regions_in_bounds(
    regions: tuple[schedule._TensorRegion, ...], bounds_shape: tuple[int, ...], *, direction: str
) -> None:
    # Bounds must be checked before sorting: malformed regions must keep the
    # native diagnostic instead of leaking IndexError or being silently accepted.
    schedule._require_valid_plan(
        not any(size < 0 for size in bounds_shape)
        and not any(
            len(region.offsets) != len(bounds_shape)
            or len(region.shape) != len(bounds_shape)
            or any(
                offset < 0 or size < 0 or offset + size > bound
                for offset, size, bound in zip(region.offsets, region.shape, bounds_shape, strict=True)
            )
            for region in regions
        ),
        f"{direction} regions must be in bounds",
    )
    positive = tuple(region for region in regions if region.numel)
    if len(positive) < 2:
        return
    # Separation along one axis is a sufficient proof for arbitrary rectangles.
    # Failure to prove it says nothing about overlap; retain the general check.
    for axis in range(len(bounds_shape)):
        ordered = sorted(positive, key=lambda region: region.offsets[axis])
        if all(first.offsets[axis] + first.shape[axis] <= second.offsets[axis] for first, second in pairwise(ordered)):
            return
    NATIVE_VALIDATE_REGIONS(regions, bounds_shape, direction=direction)


def _validate_plan(plan: schedule._RedistributionPlan) -> None:
    participant_set = set(plan.participants)
    schedule._require_valid_plan(len(participant_set) == len(plan.participants), "participants must be unique")
    for name, partitions in (("storage", plan.storage_partitions), ("compute", plan.compute_partitions)):
        schedule._require_valid_plan(
            tuple(partition.participant for partition in partitions) == plan.participants,
            f"{name} partitions must follow the redistribution participants",
        )
        for partition in partitions:
            schedule._validate_participant_logical_partition(
                partition, plan.logical_shape, direction=f"{name} logical partition {partition.participant}"
            )
    routes = plan.storage_to_compute_routes
    source_index: dict[int, list[schedule._TensorRegionRoute]] = defaultdict(list)
    destination_index: dict[int, list[schedule._TensorRegionRoute]] = defaultdict(list)
    for route in routes:
        schedule._require_valid_plan(
            bool(route.source.participants and route.destination.participants),
            "routes require source and destination participants",
        )
        schedule._require_valid_plan(
            len(set(route.source.participants)) == len(route.source.participants)
            and len(set(route.destination.participants)) == len(route.destination.participants),
            "route endpoint participants must be unique",
        )
        schedule._require_valid_plan(
            set(route.source.participants) <= participant_set
            and set(route.destination.participants) <= participant_set,
            "route references an unknown participant",
        )
        schedule._require_valid_plan(
            route.logical_region.numel == route.source.tensor_region.numel == route.destination.tensor_region.numel,
            "route endpoint regions must have equal numel",
        )
        for participant in route.source.participants:
            source_index[participant].append(route)
        for participant in route.destination.participants:
            destination_index[participant].append(route)
    schedule._validate_tensor_region_partition(
        tuple(route.logical_region for route in routes),
        plan.logical_shape,
        direction=schedule._RedistributionDirection.STORAGE_TO_COMPUTE.value,
    )
    for storage, compute in zip(plan.storage_partitions, plan.compute_partitions, strict=True):
        participant = storage.participant
        source_routes = source_index.get(participant, ())
        schedule._validate_regions_cover_partition(
            tuple(route.logical_region for route in source_routes),
            storage.logical_regions,
            plan.logical_shape,
            direction=f"storage source {participant}",
        )
        schedule._validate_tensor_region_partition(
            tuple(route.source.tensor_region for route in source_routes),
            storage.tensor_shape,
            direction=f"storage source tensor {participant}",
        )
        destination_routes = destination_index.get(participant, ())
        schedule._validate_regions_cover_partition(
            tuple(route.logical_region for route in destination_routes),
            compute.logical_regions,
            plan.logical_shape,
            direction=f"compute destination {participant}",
        )
        schedule._validate_tensor_region_partition(
            tuple(route.destination.tensor_region for route in destination_routes),
            compute.tensor_shape,
            direction=f"compute destination tensor {participant}",
        )


def validate_redistribution_plan(plan: schedule._RedistributionPlan) -> None:
    _observed("validation", lambda: _validate_plan(plan))


def _pack(
    plans: tuple[schedule._RedistributionPlan, ...],
    *,
    direction: schedule._RedistributionDirection,
    process_group: dist.ProcessGroup,
    local_participant: int,
) -> schedule._PackedAllToAllSchedule:
    participants = plans[0].participants
    sends: dict[int, list[tuple[int, schedule._TensorRegion]]] = defaultdict(list)
    receives: dict[int, list[tuple[int, schedule._TensorRegion]]] = defaultdict(list)
    remote = False
    for parameter_index, plan in enumerate(plans):
        for source, destination, source_region, destination_region in schedule._resolve_routes_to_transfers(
            direction.routes(plan), participants
        ):
            # This flag describes the whole group, including transfers not
            # involving this rank; it controls native collective participation.
            remote |= source != destination
            if source == local_participant:
                sends[destination].append((parameter_index, source_region))
            if destination == local_participant:
                receives[source].append((parameter_index, destination_region))

    def pack_side(
        by_peer: dict[int, list[tuple[int, schedule._TensorRegion]]],
    ) -> tuple[list[int], tuple[tuple[schedule._PackedSpan, ...], ...]]:
        cursor = 0
        splits: list[int] = []
        spans: list[list[schedule._PackedSpan]] = [[] for _ in plans]
        for peer in participants:
            start = cursor
            for parameter_index, region in by_peer.get(peer, ()):
                spans[parameter_index].append(schedule._PackedSpan(region, cursor))
                cursor += region.numel
            splits.append(cursor - start)
        return splits, tuple(tuple(parameter_spans) for parameter_spans in spans)

    input_splits, input_spans = pack_side(sends)
    output_splits, output_spans = pack_side(receives)
    return schedule._PackedAllToAllSchedule(
        process_group=process_group,
        participants=participants,
        local_participant=local_participant,
        has_remote_transfers=remote,
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
        input_spans_by_parameter=input_spans,
        output_spans_by_parameter=output_spans,
    )


def build_packed_all_to_all_schedule(
    redistribution_plans: tuple[schedule._RedistributionPlan, ...],
    *,
    direction: schedule._RedistributionDirection,
    process_group: dist.ProcessGroup,
    local_participant: int,
) -> schedule._PackedAllToAllSchedule:
    return _observed(
        "packing",
        lambda: _pack(
            redistribution_plans, direction=direction, process_group=process_group, local_participant=local_participant
        ),
    )


def dtensor_storage_region_for_participant(tensor: DTensor, participant: int) -> schedule._TensorRegion:
    cache = _mesh_coordinates.get()
    if cache is None:
        return NATIVE_STORAGE_REGION(tensor, participant)
    mesh = tensor.device_mesh
    entry = cache.get(id(mesh))
    if entry is None:
        coordinates: dict[int, tuple[int, ...]] = {}
        for flat_index, rank in enumerate(schedule._device_mesh_ranks(mesh)):
            coordinate = [0] * mesh.ndim
            index = flat_index
            for axis in range(mesh.ndim - 1, -1, -1):
                index, coordinate[axis] = divmod(index, mesh.shape[axis])
            # Keep tuple.index's first-occurrence semantics for malformed meshes.
            coordinates.setdefault(rank, tuple(coordinate))
        entry = (mesh, coordinates)
        cache[id(mesh)] = entry
    coordinate = entry[1].get(participant)
    if coordinate is None:
        return NATIVE_STORAGE_REGION(tensor, participant)
    local_shape = list(tensor.shape)
    offsets = [0] * tensor.ndim
    for axis, placement in enumerate(tensor.placements):
        if mesh.shape[axis] == 1 or type(placement) is Replicate:
            continue
        schedule._require_valid_plan(type(placement) is Shard, "storage placement must be exact Shard or Replicate")
        placement = cast("Shard", placement)
        dimension = placement.dim % tensor.ndim
        local_shape[dimension], offsets[dimension] = Shard.local_shard_size_and_offset(
            tensor.shape[dimension], mesh.shape[axis], coordinate[axis]
        )
    return schedule._TensorRegion(offsets=tuple(offsets), shape=tuple(local_shape))


def dtensor_storage_regions(tensor: DTensor, participants: tuple[int, ...], *, required_storage_mesh_axis: int | None):
    return _observed(
        "geometry",
        lambda: NATIVE_STORAGE_REGIONS(tensor, participants, required_storage_mesh_axis=required_storage_mesh_axis),
    )


def resolve_muon_redistribution_plans(
    contexts: tuple[schedule._BucketPlanningContext[muon._ParameterComputeLayout], ...],
    *,
    ns_steps_by_group: Sequence[int],
) -> tuple[tuple[schedule._RedistributionPlan | None, ...], ...]:
    return _observed("routes", lambda: NATIVE_RESOLVE_PLANS(contexts, ns_steps_by_group=ns_steps_by_group))


def build_bucket_plans[ItemT](
    items: Sequence[ItemT],
    specs: Sequence[schedule._BucketSpec],
    *,
    get_fqn: Callable[[ItemT], str],
    get_storage_dtensor: Callable[[ItemT], DTensor],
    requires_redistribution: Callable[[ItemT], bool],
    resolve_redistribution_plans: Callable[
        [tuple[schedule._BucketPlanningContext[ItemT], ...]],
        Sequence[Sequence[schedule._RedistributionPlan | None]],
    ],
) -> schedule._BucketPlanningResult[ItemT]:
    token = _mesh_coordinates.set({})
    try:
        return _observed(
            "build",
            lambda: NATIVE_BUILD_BUCKET_PLANS(
                items,
                specs,
                get_fqn=get_fqn,
                get_storage_dtensor=get_storage_dtensor,
                requires_redistribution=requires_redistribution,
                resolve_redistribution_plans=resolve_redistribution_plans,
            ),
        )
    finally:
        _mesh_coordinates.reset(token)
