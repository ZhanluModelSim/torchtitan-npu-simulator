"""Native plan equivalence and actual Gloo transport for optimized metadata planning."""

from __future__ import annotations

import random
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchtitan.distributed.flex_shard._optimizer_reshard_schedule as schedule
import torchtitan.distributed.flex_shard.dist_muon as muon
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

import torchtitan_npu  # noqa: F401
from torchtitan_npu.extensions.torchtitan.distributed.flex_shard import planning


def _regions(shape, ranks):
    regions = []
    for index, rank in enumerate(ranks):
        size, offset = Shard.local_shard_size_and_offset(shape[0], len(ranks), index)
        regions.append(((rank,), schedule._TensorRegion((offset, 0), (size, shape[1]))))
    return tuple(regions)


def _native_validation(monkeypatch):
    context = monkeypatch.context()
    patcher = context.__enter__()
    patcher.setattr(schedule, "_validate_disjoint_regions_in_bounds", planning.NATIVE_VALIDATE_REGIONS)
    patcher.setattr(schedule, "_validate_redistribution_plan", planning.NATIVE_VALIDATE_PLAN)
    patcher.setattr(muon, "_build_batched_matrix_redistribution_plan", planning.NATIVE_BATCHED_PLAN)
    return context


def _outcome(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (RuntimeError, ValueError) as error:
        return type(error), str(error)
    return None


@pytest.mark.parametrize("seed", [0, 11, 29], ids=lambda seed: f"seed{seed}")
def test_region_validation_matches_native_for_rectangles_and_invalid_bounds(seed):
    rng = random.Random(seed)
    for _ in range(100):
        regions = tuple(
            schedule._TensorRegion(
                tuple(rng.randrange(-1, 12) for _ in range(2)), tuple(rng.randrange(-1, 5) for _ in range(2))
            )
            for _ in range(rng.randrange(0, 20))
        )
        assert _outcome(planning.validate_disjoint_regions_in_bounds, regions, (12, 12), direction="test") == _outcome(
            planning.NATIVE_VALIDATE_REGIONS, regions, (12, 12), direction="test"
        )


@pytest.mark.parametrize(
    "regions,bounds,valid",
    [
        ([((0, 0), (2, 2)), ((2, 0), (2, 2))], (4, 2), True),
        ([((0, 0), (2, 2)), ((1, 1), (2, 2))], (4, 4), False),
        ([((0, 0), (2, 1)), ((1, 1), (1, 2)), ((0, 2), (1, 1))], (2, 3), True),
        ([((0, 0), (0, 4)), ((0, 0), (2, 4))], (2, 4), True),
        ([((), ())], (), True),
        ([((), ()), ((), ())], (), False),
        ([((0,), (1, 2))], (2, 2), False),
        ([], (-1,), False),
    ],
    ids=[
        "axis-proof",
        "overlap",
        "general-fallback",
        "zero-volume",
        "scalar",
        "scalar-overlap",
        "bad-rank",
        "negative-bound",
    ],
)
def test_region_validation_preserves_geometric_acceptance(regions, bounds, valid):
    regions = tuple(schedule._TensorRegion(*region) for region in regions)
    result = _outcome(planning.validate_disjoint_regions_in_bounds, regions, bounds, direction="test")
    assert (result is None) is valid
    assert result == _outcome(planning.NATIVE_VALIDATE_REGIONS, regions, bounds, direction="test")


@pytest.mark.parametrize("layout", ["owned", "replicate", "shard", "block"], ids=str)
@pytest.mark.parametrize("ranks", [(9, 2, 7, 1), tuple(range(8))], ids=["permuted", "empty-shards"])
def test_plan_and_all_rank_schedules_match_native(monkeypatch, layout, ranks):
    shape = (6, 3)
    regions = _regions(shape, ranks)

    def build():
        if layout in {"owned", "replicate"}:
            return schedule._build_whole_tensor_redistribution_plan(
                regions,
                participants=ranks,
                compute_participants=(ranks[-1],) if layout == "owned" else ranks,
                logical_shape=shape,
            )
        if layout == "shard":
            return schedule._build_dim0_shard_redistribution_plan(
                regions, participants=ranks, shard_participants=tuple(reversed(ranks)), logical_shape=shape
            )
        return muon._build_batched_matrix_redistribution_plan(
            regions, participants=ranks, storage_shape=shape, compute_shape=(3, 2, 3)
        )

    context = _native_validation(monkeypatch)
    try:
        expected = build()
    finally:
        context.__exit__(None, None, None)
    actual = build()
    assert actual == expected
    assert schedule._validate_redistribution_plan is planning.validate_redistribution_plan
    assert muon._build_bucket_plans is planning.build_bucket_plans
    for rank in ranks:
        for direction in schedule._RedistributionDirection:
            kwargs = dict(direction=direction, process_group=None, local_participant=rank)
            # Two parameters protect grouping order, not just a single tensor's split sizes.
            assert schedule._build_packed_all_to_all_schedule((actual, actual), **kwargs) == planning.NATIVE_PACK(
                (expected, expected), **kwargs
            )


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate-participant",
        "partition-order",
        "unknown-source",
        "duplicate-source",
        "empty-source",
        "wrong-numel",
        "missing-route",
    ],
)
def test_plan_validation_rejects_corruption_with_native_diagnostics(monkeypatch, corruption):
    plan = schedule._build_whole_tensor_redistribution_plan(
        _regions((4, 3), (0, 1)), participants=(0, 1), compute_participants=(0,), logical_shape=(4, 3)
    )
    route = plan.storage_to_compute_routes[0]
    kwargs = {}
    if corruption == "duplicate-participant":
        kwargs["participants"] = (0, 0)
    elif corruption == "partition-order":
        kwargs["storage_partitions"] = tuple(reversed(plan.storage_partitions))
    elif corruption == "missing-route":
        kwargs["storage_to_compute_routes"] = plan.storage_to_compute_routes[1:]
    else:
        endpoint = route.source
        if corruption == "unknown-source":
            endpoint = replace(endpoint, participants=(5,))
        elif corruption == "duplicate-source":
            endpoint = replace(endpoint, participants=(0, 0))
        elif corruption == "empty-source":
            endpoint = replace(endpoint, participants=())
        else:
            endpoint = replace(endpoint, tensor_region=schedule._TensorRegion((0, 0), (1, 3)))
        kwargs["storage_to_compute_routes"] = (replace(route, source=endpoint), *plan.storage_to_compute_routes[1:])
    expected = None
    with monkeypatch.context() as patcher:
        patcher.setattr(schedule, "_validate_disjoint_regions_in_bounds", planning.NATIVE_VALIDATE_REGIONS)
        patcher.setattr(schedule, "_validate_redistribution_plan", planning.NATIVE_VALIDATE_PLAN)
        expected = _outcome(replace, plan, **kwargs)
    actual = _outcome(replace, plan, **kwargs)
    assert actual is not None
    assert actual == expected


def test_bucket_scope_releases_cache_on_failure_and_observer_is_local(monkeypatch):
    def failing(*args, **kwargs):
        assert planning._mesh_coordinates.get() == {}
        raise RuntimeError("planned failure")

    monkeypatch.setattr(planning, "NATIVE_BUILD_BUCKET_PLANS", failing)
    with planning.collect_planning_timings() as timings, pytest.raises(RuntimeError, match="planned failure"):
        planning.build_bucket_plans(
            (),
            (),
            get_fqn=str,
            get_storage_dtensor=lambda x: x,
            requires_redistribution=lambda x: False,
            resolve_redistribution_plans=lambda x: (),
        )
    assert planning._mesh_coordinates.get() is None
    assert timings["build"].calls == 1
    assert timings["build"].wall_seconds >= 0
    with planning.collect_planning_timings() as second:
        assert second == {}
    assert planning._observer.get() is None


def _slice(region):
    return tuple(slice(offset, offset + size) for offset, size in zip(region.offsets, region.shape, strict=True))


def _transport(schedule_value, source, destination_shape):
    packed = torch.empty(schedule_value.input_buffer_numel)
    for span in schedule_value.input_spans_by_parameter[0]:
        packed[span.buffer_offset : span.buffer_offset + span.numel] = source[_slice(span.region)].flatten()
    received = torch.empty(schedule_value.output_buffer_numel)
    dist.all_to_all_single(
        received,
        packed,
        output_split_sizes=schedule_value.output_split_sizes,
        input_split_sizes=schedule_value.input_split_sizes,
    )
    output = torch.empty(destination_shape)
    for span in schedule_value.output_spans_by_parameter[0]:
        output[_slice(span.region)] = received[span.buffer_offset : span.buffer_offset + span.numel].reshape(
            span.region.shape
        )
    return output


def _gloo_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        ranks = (0, 1)
        plan = schedule._build_whole_tensor_redistribution_plan(
            _regions((6, 3), ranks), participants=ranks, compute_participants=(0,), logical_shape=(6, 3)
        )
        source = torch.arange(18, dtype=torch.float32).reshape(6, 3)[rank * 3 : (rank + 1) * 3].clone()
        for packer in [planning.NATIVE_PACK, schedule._build_packed_all_to_all_schedule]:
            inbound = packer(
                (plan,),
                direction=schedule._RedistributionDirection.STORAGE_TO_COMPUTE,
                process_group=dist.group.WORLD,
                local_participant=rank,
            )
            compute = _transport(inbound, source, plan.compute_partition(rank).tensor_shape)
            if rank == 0:
                torch.testing.assert_close(compute, torch.arange(18, dtype=torch.float32).reshape(6, 3), rtol=0, atol=0)
            compute.add_(100)
            outbound = packer(
                (plan,),
                direction=schedule._RedistributionDirection.COMPUTE_TO_STORAGE,
                process_group=dist.group.WORLD,
                local_participant=rank,
            )
            restored = _transport(outbound, compute, source.shape)
            torch.testing.assert_close(restored, source + 100, rtol=0, atol=0)
        # Real DTensor/DeviceMesh metadata protects uneven shards and coordinate ordering.
        mesh = DeviceMesh("cpu", torch.tensor([[1, 0]]), mesh_dim_names=("replica", "shard"))
        tensor = DTensor.from_local(
            torch.empty(3 if rank == 1 else 2, 3),
            mesh,
            (Replicate(), Shard(0)),
            shape=torch.Size((5, 3)),
            stride=(3, 1),
            run_check=False,
        )
        token = planning._mesh_coordinates.set({})
        try:
            for participant in ranks:
                expected = planning.NATIVE_STORAGE_REGION(tensor, participant)
                actual = schedule._dtensor_storage_region_for_participant(tensor, participant)
                assert actual == expected
                assert actual.offsets == ((0, 0) if participant == 1 else (3, 0))
            assert len(planning._mesh_coordinates.get()) == 1
        finally:
            planning._mesh_coordinates.reset(token)
    finally:
        dist.destroy_process_group()


def test_all_to_all_schedule_round_trip_matches_values_with_two_rank_gloo(tmp_path):
    mp.spawn(_gloo_worker, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)


def _mesh_coordinate_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=4)
    try:
        # Both axes are nontrivial and physical mesh order differs from rank order.
        mesh = DeviceMesh("cpu", torch.tensor([[3, 1], [2, 0]]), mesh_dim_names=("replica", "shard"))
        coordinates = {3: (0, 0), 1: (0, 1), 2: (1, 0), 0: (1, 1)}
        row, column = coordinates[rank]
        tensors = (
            DTensor.from_local(
                torch.empty(3 if column == 0 else 2, 7),
                mesh,
                (Replicate(), Shard(0)),
                shape=torch.Size((5, 7)),
                stride=(7, 1),
                run_check=False,
            ),
            DTensor.from_local(
                torch.empty(3 if row == 0 else 2, 4 if column == 0 else 3),
                mesh,
                (Shard(0), Shard(1)),
                shape=torch.Size((5, 7)),
                stride=(7, 1),
                run_check=False,
            ),
        )
        expected = [[planning.NATIVE_STORAGE_REGION(tensor, peer) for peer in range(4)] for tensor in tensors]
        flattened_meshes = 0
        native_flatten = schedule._device_mesh_ranks

        def counted_flatten(device_mesh):
            nonlocal flattened_meshes
            flattened_meshes += 1
            return native_flatten(device_mesh)

        token = planning._mesh_coordinates.set({})
        try:
            with patch.object(schedule, "_device_mesh_ranks", counted_flatten):
                for _ in range(2):
                    for index, tensor in enumerate(tensors):
                        for peer, (row_coordinate, column_coordinate) in coordinates.items():
                            actual = schedule._dtensor_storage_region_for_participant(tensor, peer)
                            assert actual == expected[index][peer]
                            if index == 0:
                                assert actual.offsets == (0 if column_coordinate == 0 else 3, 0)
                                assert actual.shape == (3 if column_coordinate == 0 else 2, 7)
                            else:
                                assert actual.offsets == (
                                    0 if row_coordinate == 0 else 3,
                                    0 if column_coordinate == 0 else 4,
                                )
                                assert actual.shape == (
                                    3 if row_coordinate == 0 else 2,
                                    4 if column_coordinate == 0 else 3,
                                )
                for index, tensor in enumerate(tensors):
                    peers = tuple(mesh.mesh[coordinates[rank][0]].tolist())
                    expected_domain = planning.NATIVE_STORAGE_REGIONS(tensor, peers, required_storage_mesh_axis=1)
                    actual_domain = planning.dtensor_storage_regions(tensor, peers, required_storage_mesh_axis=1)
                    assert actual_domain == expected_domain
                    assert actual_domain[0] == ((5, 7) if index == 0 else (3 if row == 0 else 2, 7))
                assert flattened_meshes == 1
        finally:
            planning._mesh_coordinates.reset(token)
        assert planning._mesh_coordinates.get() is None
    finally:
        dist.destroy_process_group()


def test_mesh_coordinates_preserve_replica_and_orthogonal_shards_on_four_rank_gloo(tmp_path):
    mp.spawn(_mesh_coordinate_worker, args=(f"file://{tmp_path / 'mesh-rendezvous'}",), nprocs=4, join=True)


@pytest.mark.parametrize("seed", [7, 29, 113])
def test_batched_routes_match_native_for_uneven_replicated_and_empty_partitions(seed):
    rng = random.Random(seed)
    for _ in range(40):
        count, matrices, rows, columns = (rng.randint(1, 24), rng.randint(1, 45), rng.randint(1, 19), rng.randint(1, 9))
        ranks = list(range(count))
        rng.shuffle(ranks)
        groups = rng.randint(1, count)
        holders = [tuple(ranks[index::groups]) for index in range(groups)]
        cuts = sorted([0, matrices * rows] + [rng.randint(0, matrices * rows) for _ in range(groups - 1)])
        regions = [
            (group, schedule._TensorRegion((start, 0), (end - start, columns)))
            for group, start, end in zip(holders, cuts[:-1], cuts[1:], strict=True)
        ]
        rng.shuffle(regions)
        kwargs = dict(
            participants=tuple(ranks), storage_shape=(matrices * rows, columns), compute_shape=(matrices, rows, columns)
        )
        expected = planning.NATIVE_BATCHED_PLAN(regions, **kwargs)
        actual = muon._build_batched_matrix_redistribution_plan(regions, **kwargs)
        assert actual == expected
        for rank in ranks:
            for direction in schedule._RedistributionDirection:
                packing = dict(direction=direction, process_group=None, local_participant=rank)
                assert planning.build_packed_all_to_all_schedule((actual,), **packing) == planning.NATIVE_PACK(
                    (expected,), **packing
                )
    assert muon._build_batched_matrix_redistribution_plan is planning.build_batched_matrix_redistribution_plan


@pytest.mark.parametrize(
    "compute_shape",
    [(16, 1024, 4096), (128, 512, 1536), (64, 128, 1536)],
    ids=["matrices16", "matrices128", "matrices64"],
)
def test_batched_routes_preserve_pro_geometry_and_full_plan(compute_shape):
    matrices, rows, columns = compute_shape
    shape = (matrices * rows, columns)
    ranks = tuple(range(2048))
    regions = _regions(shape, ranks)
    kwargs = dict(participants=ranks, storage_shape=shape, compute_shape=compute_shape)
    actual = planning.build_batched_matrix_redistribution_plan(regions, **kwargs)
    assert actual == planning.NATIVE_BATCHED_PLAN(regions, **kwargs)
    assert len(actual.storage_to_compute_routes) == 2048
    assert sum(bool(part.logical_regions[0].numel) for part in actual.compute_partitions) == matrices


@pytest.mark.parametrize(
    "regions,storage_shape,compute_shape",
    [
        ([((0,), ((0, 0), (0, 3)))], (0, 3), (0, 4, 3)),
        ([((0,), ((0, 0), (4, 3)))], (4, 3), (2, 3, 3)),
        ([((0,), ((-1, 0), (5, 3)))], (4, 3), (1, 4, 3)),
        ([((0,), ((0,), (4, 3)))], (4, 3), (1, 4, 3)),
        ([], (4, 3), (1, 4, 3)),
        ([((0,), ((0, 0), (4, 3))), ((0,), ((0, 0), (4, 3)))], (4, 3), (1, 4, 3)),
    ],
    ids=["zero-matrices", "bad-shape", "negative-offset", "bad-offset-rank", "missing-holder", "duplicate-holder"],
)
def test_batched_route_degenerate_and_invalid_inputs_preserve_native_diagnostics(regions, storage_shape, compute_shape):
    regions = tuple((holders, schedule._TensorRegion(*region)) for holders, region in regions)
    kwargs = dict(participants=(0,), storage_shape=storage_shape, compute_shape=compute_shape)

    def outcome(builder):
        try:
            return builder(regions, **kwargs)
        except (RuntimeError, ValueError, IndexError) as error:
            return type(error), str(error)

    assert outcome(planning.build_batched_matrix_redistribution_plan) == outcome(planning.NATIVE_BATCHED_PLAN)


def _batched_gloo_worker(rank, rendezvous, world):
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world)
    try:
        ranks = tuple(range(world))
        # Uneven row shards split matrices; single-matrix case has empty compute
        # participants. Replicas exercise native source selection on the return.
        for matrices, rows, replicated in [(5, 2, False), (1, 3, False), (5, 2, True)]:
            shape = (matrices * rows, 3)
            global_values = torch.arange(shape[0] * 3, dtype=torch.float32).reshape(shape)
            groups = world // 2 if replicated else world
            chunk = (shape[0] + groups - 1) // groups
            regions = tuple(
                (
                    tuple(range(index, world, groups)),
                    schedule._TensorRegion(
                        (min(index * chunk, shape[0]), 0), (max(0, min(chunk, shape[0] - index * chunk)), 3)
                    ),
                )
                for index in range(groups)
            )
            storage_index = rank % groups
            source = global_values[storage_index * chunk : (storage_index + 1) * chunk].clone()
            matrix_chunk = (matrices + world - 1) // world
            expected_compute = global_values[rank * matrix_chunk * rows : (rank + 1) * matrix_chunk * rows]
            kwargs = dict(participants=ranks, storage_shape=shape, compute_shape=(matrices, rows, 3))
            expected = planning.NATIVE_BATCHED_PLAN(regions, **kwargs)
            actual = muon._build_batched_matrix_redistribution_plan(regions, **kwargs)
            assert actual == expected
            for plan in (expected, actual):
                inbound = planning.build_packed_all_to_all_schedule(
                    (plan,),
                    direction=schedule._RedistributionDirection.STORAGE_TO_COMPUTE,
                    process_group=dist.group.WORLD,
                    local_participant=rank,
                )
                compute = _transport(inbound, source, plan.compute_partition(rank).tensor_shape)
                torch.testing.assert_close(compute, expected_compute, rtol=0, atol=0)
                compute.add_(100)
                outbound = planning.build_packed_all_to_all_schedule(
                    (plan,),
                    direction=schedule._RedistributionDirection.COMPUTE_TO_STORAGE,
                    process_group=dist.group.WORLD,
                    local_participant=rank,
                )
                restored = _transport(outbound, compute, source.shape)
                torch.testing.assert_close(restored, source + 100, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_batched_routes_round_trip_with_gloo(tmp_path, world):
    mp.spawn(_batched_gloo_worker, args=(f"file://{tmp_path / 'batch-rendezvous'}", world), nprocs=world, join=True)


def test_validation_facts_preserve_failures_and_fall_back_for_mutable_inputs(monkeypatch):
    calls = []
    native = planning.NATIVE_VALIDATE_TENSOR_PARTITION

    def counted(*args, **kwargs):
        calls.append(kwargs["direction"])
        return native(*args, **kwargs)

    monkeypatch.setattr(planning, "NATIVE_VALIDATE_TENSOR_PARTITION", counted)
    facts = planning._ValidationFacts()
    token = planning._validation_facts.set(facts)
    try:
        regions = (schedule._TensorRegion((0,), (4,)),)
        for direction in ("first", "second"):
            planning.validate_tensor_region_partition(regions, (4,), direction=direction)
        assert calls == ["first"]
        for direction in ("bad-one", "bad-two"):
            with pytest.raises(RuntimeError, match=f"{direction} regions must be in bounds"):
                planning.validate_tensor_region_partition(regions, (3,), direction=direction)
        assert calls[-2:] == ["bad-one", "bad-two"]
        mutable = [schedule._TensorRegion([0], [4])]
        for _ in range(2):
            planning.validate_tensor_region_partition(mutable, [4], direction="mutable")
        assert calls[-2:] == ["mutable", "mutable"]
        mutable[0].shape[0] = 5
        with pytest.raises(RuntimeError, match="mutable regions must be in bounds"):
            planning.validate_tensor_region_partition(mutable, [4], direction="mutable")
    finally:
        planning._validation_facts.reset(token)


def test_validation_facts_distinguish_tensor_shape_and_expected_partition():
    regions = (schedule._TensorRegion((0,), (4,)),)
    partition = schedule._ParticipantPartition(participant=0, tensor_shape=(4,), logical_regions=regions)
    token = planning._validation_facts.set(planning._ValidationFacts())
    try:
        planning.validate_participant_logical_partition(partition, (8,), direction="participant")
        invalid = replace(partition, tensor_shape=(3,))
        actual = _outcome(planning.validate_participant_logical_partition, invalid, (8,), direction="participant")
        assert actual is not None
        assert actual == _outcome(planning.NATIVE_VALIDATE_PARTICIPANT, invalid, (8,), direction="participant")
        planning.validate_regions_cover_partition(regions, regions, (8,), direction="cover")
        # Equal volume but a different location must still fail containment.
        shifted = (schedule._TensorRegion((4,), (4,)),)
        actual = _outcome(planning.validate_regions_cover_partition, regions, shifted, (8,), direction="cover")
        assert actual is not None
        assert actual == _outcome(planning.NATIVE_VALIDATE_COVER, regions, shifted, (8,), direction="cover")
    finally:
        planning._validation_facts.reset(token)


def test_validation_facts_have_bounded_retention_and_eviction_keeps_checks(monkeypatch):
    monkeypatch.setattr(planning, "_MAX_VALIDATION_FACTS", 8)
    monkeypatch.setattr(planning, "_MAX_VALIDATION_WEIGHT", 64)
    monkeypatch.setattr(planning, "_MAX_IMMUTABLE_INPUTS", 8)
    monkeypatch.setattr(planning, "_MAX_IMMUTABLE_WEIGHT", 32)
    facts = planning._ValidationFacts()
    token = planning._validation_facts.set(facts)
    try:
        for offset in range(512):
            planning.validate_disjoint_regions_in_bounds(
                (schedule._TensorRegion((offset,), (1,)),), (512,), direction="bounded"
            )
            assert len(facts.successes) <= 8
            assert len(facts.order) == len(facts.successes)
            assert facts.weight <= 64
            assert len(facts.immutable_inputs) <= 8
            assert facts.immutable_weight <= 32
        with pytest.raises(RuntimeError, match="bad regions must be in bounds"):
            planning.validate_disjoint_regions_in_bounds(
                (schedule._TensorRegion((512,), (1,)),), (512,), direction="bad"
            )
        facts.clear()
        assert not facts.successes and not facts.order and facts.weight == 0
        assert not facts.immutable_inputs and facts.immutable_weight == 0
    finally:
        planning._validation_facts.reset(token)


@pytest.mark.parametrize("fail", [False, True], ids=["success", "exception"])
def test_validation_fact_scope_clears_references_and_restores_nested_scope(monkeypatch, fail):
    observed = []
    outer = planning._ValidationFacts()

    def validate(_plan):
        current = planning._validation_facts.get()
        assert current is not outer
        observed.append(current)
        planning.validate_disjoint_regions_in_bounds((schedule._TensorRegion((0,), (1,)),), (1,), direction="inner")
        if fail:
            raise ValueError("scope failure")

    monkeypatch.setattr(planning, "_validate_plan", validate)
    token = planning._validation_facts.set(outer)
    try:
        for _ in range(3):
            if fail:
                with pytest.raises(ValueError, match="scope failure"):
                    planning.validate_redistribution_plan(None)
            else:
                planning.validate_redistribution_plan(None)
            assert planning._validation_facts.get() is outer
        assert len({id(item) for item in observed}) == 3
        assert all(not item.successes and not item.order and item.weight == 0 for item in observed)
        assert all(not item.immutable_inputs and item.immutable_weight == 0 for item in observed)
    finally:
        planning._validation_facts.reset(token)
    assert planning._validation_facts.get() is None


def test_immutable_qualification_requires_identity_and_exact_types():
    facts = planning._ValidationFacts()
    shape = (4,)
    facts.immutable_inputs[id(shape)] = ((5,), 999)
    assert planning._shape_weight(shape, facts) == 2
    assert facts.immutable_inputs[id(shape)][0] is shape
    assert planning._shape_weight([4], facts) is None
    assert planning._shape_weight((True,), facts) is None
    region = schedule._TensorRegion([0], [4])
    assert planning._fact_weight((region,), (), (4,), (), facts) is None
    assert id(region) not in facts.immutable_inputs


def test_native_plan_qualification_falls_back_for_mutable_geometry():
    plan = schedule._build_whole_tensor_redistribution_plan(
        _regions((4, 2), (0, 1)), participants=(0, 1), compute_participants=(0,), logical_shape=(4, 2)
    )
    assert planning._native_plan_geometry(plan, planning._ValidationFacts())
    first = plan.storage_partitions[0]
    region = first.logical_regions[0]
    mutable = schedule._TensorRegion(list(region.offsets), list(region.shape))
    changed = replace(first, logical_regions=(mutable,))
    plan = replace(plan, storage_partitions=(changed, *plan.storage_partitions[1:]))
    assert not planning._native_plan_geometry(plan, planning._ValidationFacts())
    planning.validate_redistribution_plan(plan)
    mutable.shape[0] = 99
    actual = _outcome(planning.validate_redistribution_plan, plan)
    assert actual is not None
    assert actual == _outcome(planning._validate_plan, plan)
