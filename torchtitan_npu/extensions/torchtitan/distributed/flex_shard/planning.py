# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause
"""Equivalent metadata planning for the pinned TorchTitan FlexShard implementation.

This module has no activation side effects. Caches live only within one bucket
planning call. Timing observations are host wall time, never simulated costs.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
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
NATIVE_BATCHED_PLAN = muon._build_batched_matrix_redistribution_plan
NATIVE_VALIDATE_TENSOR_PARTITION = schedule._validate_tensor_region_partition
NATIVE_VALIDATE_PARTICIPANT = schedule._validate_participant_logical_partition
NATIVE_VALIDATE_COVER = schedule._validate_regions_cover_partition

# Bound retained Python metadata independently of participant count. Oversized
# or nonstandard inputs still receive the full original check without caching.
_MAX_VALIDATION_FACTS = 1024
_MAX_VALIDATION_WEIGHT = 16384
_MAX_REGIONS_PER_FACT = 64
_MAX_IMMUTABLE_INPUTS = 4096
_MAX_IMMUTABLE_WEIGHT = 65536


@dataclass
class _ValidationFacts:
    successes: set[tuple[object, ...]] = field(default_factory=set)
    order: deque[tuple[tuple[object, ...], int]] = field(default_factory=deque)
    weight: int = 0
    immutable_inputs: dict[int, tuple[object, int]] = field(default_factory=dict)
    immutable_weight: int = 0
    trusted_geometry: bool = False

    def remember_immutable(self, value: object, weight: int) -> None:
        if (
            len(self.immutable_inputs) < _MAX_IMMUTABLE_INPUTS
            and self.immutable_weight + weight <= _MAX_IMMUTABLE_WEIGHT
        ):
            self.immutable_inputs[id(value)] = (value, weight)
            self.immutable_weight += weight

    def add(self, key: tuple[object, ...], weight: int) -> None:
        while self.order and (
            len(self.successes) >= _MAX_VALIDATION_FACTS or self.weight + weight > _MAX_VALIDATION_WEIGHT
        ):
            previous, previous_weight = self.order.popleft()
            self.successes.remove(previous)
            self.weight -= previous_weight
        if weight <= _MAX_VALIDATION_WEIGHT and _MAX_VALIDATION_FACTS > 0:
            self.successes.add(key)
            self.order.append((key, weight))
            self.weight += weight

    def clear(self) -> None:
        self.successes.clear()
        self.order.clear()
        self.weight = 0
        self.immutable_inputs.clear()
        self.immutable_weight = 0
        self.trusted_geometry = False


def _shape_weight(shape: tuple[int, ...], facts: _ValidationFacts) -> int | None:
    if type(shape) is not tuple or len(shape) > 16:
        return None
    entry = facts.immutable_inputs.get(id(shape))
    if entry is not None and entry[0] is shape:
        return entry[1]
    for size in shape:
        if type(size) is not int:
            return None
    weight = len(shape) + 1
    facts.remember_immutable(shape, weight)
    return weight


_validation_facts: ContextVar[_ValidationFacts | None] = ContextVar("muon_validation_facts", default=None)


def _fact_weight(
    regions: tuple[schedule._TensorRegion, ...],
    expected: tuple[schedule._TensorRegion, ...],
    bounds: tuple[int, ...],
    tensor_shape: tuple[int, ...],
    facts: _ValidationFacts,
) -> int | None:
    # Exact immutable types prevent a mutable/custom hash or overridden numel
    # from borrowing a proof obtained for a standard native region.
    if type(regions) is not tuple or type(expected) is not tuple:
        return None
    if len(regions) + len(expected) > _MAX_REGIONS_PER_FACT:
        return None
    weight = 1
    for shape in (bounds, tensor_shape):
        shape_weight = _shape_weight(shape, facts)
        if shape_weight is None:
            return None
        weight += shape_weight
    for group in (regions, expected):
        for region in group:
            if type(region) is not schedule._TensorRegion:
                return None
            entry = facts.immutable_inputs.get(id(region))
            if entry is not None and entry[0] is region:
                weight += entry[1]
                continue
            region_weight = 0
            for values in (region.offsets, region.shape):
                shape_weight = _shape_weight(values, facts)
                if shape_weight is None:
                    return None
                region_weight += shape_weight
            facts.remember_immutable(region, region_weight)
            weight += region_weight
    return weight


def _validate_fact(
    kind: str,
    regions: tuple[schedule._TensorRegion, ...],
    bounds: tuple[int, ...],
    check: Callable[[], None],
    *,
    expected: tuple[schedule._TensorRegion, ...] = (),
    tensor_shape: tuple[int, ...] = (),
) -> None:
    facts = _validation_facts.get()
    if facts is not None and facts.trusted_geometry:
        # The enclosing native plan's immutable geometry was checked once.
        # Value-based proof keys remain unchanged; only qualification is hoisted.
        if len(regions) + len(expected) > _MAX_REGIONS_PER_FACT:
            check()
            return
        key = (kind, regions, expected, bounds, tensor_shape)
        if key in facts.successes:
            return
        weight = 3 + len(bounds) + len(tensor_shape)
        for group in (regions, expected):
            for region in group:
                weight += 2 + len(region.offsets) + len(region.shape)
    else:
        weight = _fact_weight(regions, expected, bounds, tensor_shape, facts) if facts is not None else None
    if weight is None:
        check()
        return
    key = (kind, regions, expected, bounds, tensor_shape)
    assert facts is not None
    if key not in facts.successes:
        check()
        # Failed checks never reach this point; direction is therefore not part
        # of the proof key and the native first-failure diagnostic is preserved.
        facts.add(key, weight)


def validate_tensor_region_partition(
    regions: tuple[schedule._TensorRegion, ...], logical_shape: tuple[int, ...], *, direction: str
) -> None:
    _validate_fact(
        "tensor",
        regions,
        logical_shape,
        lambda: NATIVE_VALIDATE_TENSOR_PARTITION(regions, logical_shape, direction=direction),
    )


def validate_participant_logical_partition(
    partition: schedule._ParticipantPartition, logical_shape: tuple[int, ...], *, direction: str
) -> None:
    _validate_fact(
        "participant",
        partition.logical_regions,
        logical_shape,
        lambda: NATIVE_VALIDATE_PARTICIPANT(partition, logical_shape, direction=direction),
        tensor_shape=partition.tensor_shape,
    )


def validate_regions_cover_partition(
    regions: tuple[schedule._TensorRegion, ...],
    expected: tuple[schedule._TensorRegion, ...],
    logical_shape: tuple[int, ...],
    *,
    direction: str,
) -> None:
    _validate_fact(
        "cover",
        regions,
        logical_shape,
        lambda: NATIVE_VALIDATE_COVER(regions, expected, logical_shape, direction=direction),
        expected=expected,
    )


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


def _validate_disjoint_regions_in_bounds(
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


def validate_disjoint_regions_in_bounds(
    regions: tuple[schedule._TensorRegion, ...], bounds_shape: tuple[int, ...], *, direction: str
) -> None:
    _validate_fact(
        "disjoint",
        regions,
        bounds_shape,
        lambda: _validate_disjoint_regions_in_bounds(regions, bounds_shape, direction=direction),
    )


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


def _native_plan_geometry(plan: schedule._RedistributionPlan, facts: _ValidationFacts) -> bool:
    """Qualify geometry once; this proves no partition or route invariant."""
    if type(plan) is not schedule._RedistributionPlan or _shape_weight(plan.logical_shape, facts) is None:
        return False

    def regions_valid(regions: tuple[schedule._TensorRegion, ...]) -> bool:
        if type(regions) is not tuple:
            return False
        for region in regions:
            if (
                type(region) is not schedule._TensorRegion
                or _shape_weight(region.offsets, facts) is None
                or _shape_weight(region.shape, facts) is None
            ):
                return False
        return True

    for partitions in (plan.storage_partitions, plan.compute_partitions):
        if type(partitions) is not tuple:
            return False
        for partition in partitions:
            if (
                type(partition) is not schedule._ParticipantPartition
                or _shape_weight(partition.tensor_shape, facts) is None
                or not regions_valid(partition.logical_regions)
            ):
                return False
    if type(plan.storage_to_compute_routes) is not tuple:
        return False
    for route in plan.storage_to_compute_routes:
        if (
            type(route) is not schedule._TensorRegionRoute
            or type(route.source) is not schedule._RouteEndpoint
            or type(route.destination) is not schedule._RouteEndpoint
            or not regions_valid((route.logical_region, route.source.tensor_region, route.destination.tensor_region))
        ):
            return False
    return True


def validate_redistribution_plan(plan: schedule._RedistributionPlan) -> None:
    facts = _ValidationFacts()
    token = _validation_facts.set(facts)
    try:

        def validate() -> None:
            facts.trusted_geometry = _native_plan_geometry(plan, facts)
            _validate_plan(plan)

        _observed("validation", validate)
    finally:
        facts.clear()
        _validation_facts.reset(token)


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


def _storage_region_reader(tensor: DTensor, participants: tuple[int, ...]) -> Callable[[int], schedule._TensorRegion]:
    # Native validation of the subgroup and placements precedes this helper.
    # Initialize the existing build-local coordinate map, then hoist metadata
    # that is invariant for every participant in this parameter's domain.
    first = dtensor_storage_region_for_participant(tensor, participants[0])
    cache = _mesh_coordinates.get()
    entry = cache.get(id(tensor.device_mesh)) if cache is not None else None
    if entry is None:
        return lambda rank: dtensor_storage_region_for_participant(tensor, rank)
    coordinates = entry[1]
    shape = tuple(tensor.shape)
    ndim = len(shape)
    mesh_shape = tensor.device_mesh.shape
    axes = []
    for axis, placement in enumerate(tensor.placements):
        size = mesh_shape[axis]
        if size == 1 or type(placement) is Replicate:
            continue
        schedule._require_valid_plan(type(placement) is Shard, "storage placement must be exact Shard or Replicate")
        axes.append((axis, size, cast("Shard", placement).dim % ndim))

    def region(rank: int) -> schedule._TensorRegion:
        if rank == participants[0]:
            return first
        coordinate = coordinates.get(rank)
        if coordinate is None:
            return dtensor_storage_region_for_participant(tensor, rank)
        local_shape = list(shape)
        offsets = [0] * ndim
        for axis, size, dimension in axes:
            local_shape[dimension], offsets[dimension] = Shard.local_shard_size_and_offset(
                shape[dimension], size, coordinate[axis]
            )
        return schedule._TensorRegion(offsets=tuple(offsets), shape=tuple(local_shape))

    return region


def _storage_regions(
    tensor: DTensor,
    participants: tuple[int, ...],
    *,
    required_storage_mesh_axis: int | None,
) -> tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], schedule._TensorRegion], ...]]:
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
        placement = cast("Shard", tensor.placements[preserved_shard_axis])
        tensor_dim = placement.dim % tensor.ndim
        local_size, global_offset = Shard.local_shard_size_and_offset(
            tensor.shape[tensor_dim],
            storage_mesh.size(preserved_shard_axis),
            reference_coordinate[preserved_shard_axis],
        )
        domain_shape[tensor_dim] = local_size
        domain_offsets[tensor_dim] = global_offset

    reader = _storage_region_reader(tensor, participants)
    holders_by_region: dict[schedule._TensorRegion, list[int]] = {}
    for participant in participants:
        global_region = reader(participant)
        region = schedule._TensorRegion(
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
    schedule._validate_tensor_region_partition(
        tuple(region for _holders, region in regions),
        tuple(domain_shape),
        direction="transport-subgroup storage",
    )
    return tuple(domain_shape), regions


def dtensor_storage_regions(tensor: DTensor, participants: tuple[int, ...], *, required_storage_mesh_axis: int | None):
    builder = _storage_regions if _mesh_coordinates.get() is not None else NATIVE_STORAGE_REGIONS
    return _observed(
        "geometry", lambda: builder(tensor, participants, required_storage_mesh_axis=required_storage_mesh_axis)
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


def build_batched_matrix_redistribution_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], schedule._TensorRegion]],
    *,
    participants: tuple[int, ...],
    storage_shape: tuple[int, ...],
    compute_shape: tuple[int, ...],
) -> schedule._RedistributionPlan:
    """Prune disjoint matrix candidates, retaining native order and validation."""
    # The interval proof requires positive matrix dimensions and bounded row
    # regions. Delegate all other inputs to preserve native degenerate behavior
    # and diagnostics instead of introducing division-by-zero or early errors.
    eligible = (
        len(storage_shape) == 2
        and len(compute_shape) == 3
        and all(size > 0 for size in compute_shape)
        and storage_shape == (compute_shape[0] * compute_shape[1], compute_shape[2])
        and bool(participants)
        and all(
            len(region.offsets) == 2
            and len(region.shape) == 2
            and region.offsets[0] >= 0
            and region.shape[0] >= 0
            and region.offsets[0] + region.shape[0] <= storage_shape[0]
            and region.offsets[1] == 0
            and region.shape[1] == storage_shape[1]
            for _, region in storage_regions
        )
    )
    builder = _build_batched_matrix_plan if eligible else NATIVE_BATCHED_PLAN
    return builder(storage_regions, participants=participants, storage_shape=storage_shape, compute_shape=compute_shape)


def _build_batched_matrix_plan(
    storage_regions: Sequence[tuple[tuple[int, ...], schedule._TensorRegion]],
    *,
    participants: tuple[int, ...],
    storage_shape: tuple[int, ...],
    compute_shape: tuple[int, ...],
) -> schedule._RedistributionPlan:
    """Build flat 2D routes whose destination shards contain complete matrices."""
    schedule._require_valid_plan(
        len(storage_shape) == 2
        and len(compute_shape) == 3
        and storage_shape[0] == compute_shape[0] * compute_shape[1]
        and storage_shape[1] == compute_shape[2],
        "matrix-batch redistribution requires flat 2D storage plus a 3D matrix-batch compute shape",
    )

    num_matrices, matrix_rows, matrix_columns = compute_shape
    storage_by_participant = {}
    storage_endpoints = []
    for holders, logical_region in storage_regions:
        schedule._require_valid_plan(
            bool(holders)
            and len(logical_region.shape) == 2
            and logical_region.offsets[1] == 0
            and logical_region.shape[1] == matrix_columns,
            "matrix-batch redistribution requires row-sharded 2D storage",
        )
        tensor_region = schedule._TensorRegion(
            offsets=(0, 0),
            shape=logical_region.shape,
        )
        storage_endpoints.append((holders, logical_region, tensor_region))
        for participant in holders:
            schedule._require_valid_plan(
                participant in participants and participant not in storage_by_participant,
                "matrix-batch storage holders must partition participants",
            )
            storage_by_participant[participant] = (logical_region, tensor_region)

    schedule._require_valid_plan(
        set(storage_by_participant) == set(participants),
        "matrix-batch storage must cover every participant",
    )

    storage_partitions = tuple(
        schedule._ParticipantPartition(
            participant=participant,
            tensor_shape=storage_by_participant[participant][1].shape,
            logical_regions=(storage_by_participant[participant][0],),
        )
        for participant in participants
    )

    compute_endpoints = []
    compute_partitions_list = []
    for mesh_rank, participant in enumerate(participants):
        local_num_matrices, matrix_offset = Shard.local_shard_size_and_offset(
            num_matrices,
            len(participants),
            mesh_rank,
        )
        logical_region = schedule._TensorRegion(
            offsets=(matrix_offset * matrix_rows, 0),
            shape=(local_num_matrices * matrix_rows, matrix_columns),
        )
        compute_endpoints.append(((participant,), matrix_offset, local_num_matrices))
        compute_partitions_list.append(
            schedule._ParticipantPartition(
                participant=participant,
                tensor_shape=logical_region.shape,
                logical_regions=(logical_region,),
            )
        )
    compute_partitions = tuple(compute_partitions_list)

    # Shard assigns increasing matrix intervals in participant order. Keep empty
    # partitions above; omit them only from this search index, never from the plan.
    nonempty_endpoints = tuple(endpoint for endpoint in compute_endpoints if endpoint[2] > 0)
    compute_row_ends = tuple((offset + count) * matrix_rows for _, offset, count in nonempty_endpoints)
    storage_to_compute_routes = []
    for source_holders, storage_region, storage_tensor_base in storage_endpoints:
        storage_row_offset = storage_region.offsets[0]
        storage_row_end = storage_row_offset + storage_region.shape[0]
        if storage_row_end <= storage_row_offset:
            continue
        endpoint_index = bisect_right(compute_row_ends, storage_row_offset)
        while endpoint_index < len(nonempty_endpoints):
            destination_holders, matrix_offset, local_num_matrices = nonempty_endpoints[endpoint_index]
            endpoint_index += 1
            if matrix_offset * matrix_rows >= storage_row_end:
                break
            first_matrix = max(0, storage_row_offset // matrix_rows - matrix_offset)
            last_matrix = min(local_num_matrices, (storage_row_end + matrix_rows - 1) // matrix_rows - matrix_offset)
            for local_matrix_index in range(first_matrix, last_matrix):
                matrix_index = matrix_offset + local_matrix_index
                matrix_row_offset = matrix_index * matrix_rows
                route_row_offset = max(storage_row_offset, matrix_row_offset)
                route_row_end = min(
                    storage_row_end,
                    matrix_row_offset + matrix_rows,
                )
                route_rows = route_row_end - route_row_offset
                if route_rows <= 0:
                    continue

                logical_region = schedule._TensorRegion(
                    offsets=(route_row_offset, 0),
                    shape=(route_rows, matrix_columns),
                )
                storage_tensor_region = schedule._TensorRegion(
                    offsets=(
                        storage_tensor_base.offsets[0] + route_row_offset - storage_row_offset,
                        0,
                    ),
                    shape=(route_rows, matrix_columns),
                )
                compute_tensor_region = schedule._TensorRegion(
                    offsets=(
                        local_matrix_index * matrix_rows + route_row_offset - matrix_row_offset,
                        0,
                    ),
                    shape=(route_rows, matrix_columns),
                )
                storage_to_compute_routes.append(
                    schedule._TensorRegionRoute(
                        logical_region=logical_region,
                        source=schedule._RouteEndpoint(
                            storage_tensor_region,
                            source_holders,
                        ),
                        destination=schedule._RouteEndpoint(
                            compute_tensor_region,
                            destination_holders,
                        ),
                    )
                )

    return schedule._RedistributionPlan(
        participants=participants,
        logical_shape=storage_shape,
        storage_partitions=storage_partitions,
        compute_partitions=compute_partitions,
        storage_to_compute_routes=tuple(storage_to_compute_routes),
    )
