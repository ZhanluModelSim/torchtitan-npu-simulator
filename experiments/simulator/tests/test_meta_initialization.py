"""Scoped Meta initialization preserves native FSDP consumers and session state."""

import gc
import logging
import re
import weakref
from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.fsdp._fully_shard._fsdp_param as fsdp_param
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torchtitan_simulator import runner
from torchtitan_simulator.compat import meta_initialization as initialization
from torchtitan_simulator.config import SimulationConfig

try:
    from spmd_types import MeshAxis
    from torch.distributed._mesh_layout import _FlatLayout
except ImportError:
    MeshAxis = _FlatLayout = None

has_axis_cache = MeshAxis is not None and _FlatLayout is not None
requires_axis_cache = pytest.mark.skipif(not has_axis_cache, reason="requires MeshAxis and _FlatLayout")


def _metadata(tensor):
    return tensor.shape, tensor.stride(), tensor.storage_offset(), tensor.dtype, tensor.device


@pytest.fixture
def fake_world():
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_meta_submeshes_share_their_own_ordered_root_ranks(fake_world):
    layout = [[2, 0], [3, 1]]
    root = DeviceMesh("meta", layout, mesh_dim_names=("dp", "tp"))
    cpu = DeviceMesh("cpu", layout, mesh_dim_names=("dp", "tp"))
    native = root["tp"]
    with initialization.meta_initialization():
        child = root["tp"]
        sibling = root["dp"]
        other = DeviceMesh("meta", mesh=[[0, 2], [3, 1]], mesh_dim_names=("dp", "tp"))
        reordered = other["tp"]
        assert child is not native and child is not sibling
        assert child._flatten_rank_map is sibling._flatten_rank_map is root._flatten_rank_map
        assert reordered._flatten_rank_map is other._flatten_rank_map
        assert child != reordered and child != sibling
        assert child == native and hash(child) == hash(native)
        assert child.mesh.tolist() == [2, 0]
        assert child.get_coordinate() == (1,)
        assert child.get_group() is native.get_group()
        assert cpu["tp"]._flatten_rank_map is not cpu._flatten_rank_map


def test_meta_submesh_keeps_its_snapshot_after_rank_tensor_changes(fake_world):
    ranks = torch.tensor([[2, 0], [3, 1]], dtype=torch.int)
    root = DeviceMesh("meta", ranks, mesh_dim_names=("dp", "tp"))
    ranks[1] = torch.tensor([1, 3])
    with initialization.meta_initialization():
        child = root["tp"]
    assert root._flatten_rank_map == (2, 0, 3, 1)
    assert child._flatten_rank_map == (2, 0, 1, 3)


@requires_axis_cache
def test_axis_subset_cache_reuses_equal_layouts_and_resets_after_failure(monkeypatch):
    native = MeshAxis.__le__
    calls = []

    def observed(left, right):
        calls.append((left, right))
        return native(left, right)

    monkeypatch.setattr(MeshAxis, "__le__", observed)
    even = MeshAxis.of(2, 2)  # {0, 2}
    dense = MeshAxis.of(3, 1)  # {0, 1, 2}
    far = MeshAxis.of(2, 3)  # {0, 3}
    with pytest.raises(RuntimeError, match="scope failure"), initialization.meta_initialization():
        assert even <= dense
        assert MeshAxis.of(2, 2) <= MeshAxis.of(3, 1)
        assert not dense <= even
        assert not far <= dense
        assert not MeshAxis.of(2, 3) <= dense
        assert calls == [(even, dense), (dense, even), (far, dense)]
        raise RuntimeError("scope failure")
    assert MeshAxis.__le__ is observed
    with initialization.meta_initialization():
        assert even <= dense
    assert calls[-1] == (even, dense) and len(calls) == 4
    assert MeshAxis.__le__ is observed


@requires_axis_cache
def test_axis_subset_cache_preserves_native_dispatch_for_other_types():
    class UnhashableAxis(MeshAxis):
        __hash__ = None

    axis = MeshAxis.of(4, 1)
    child = UnhashableAxis(MeshAxis.of(2, 1).layout)
    with initialization.meta_initialization():
        assert child <= axis
        assert axis.__le__(child) is False
        assert axis.__le__([]) is NotImplemented


@pytest.mark.parametrize(
    "shape",
    [(5, 3), (1, 3), (0, 3), (6, 7), (2, 0)],
    ids=["uneven", "empty-ranks", "zero-rows", "dense", "zero-columns"],
)
@pytest.mark.parametrize("dim", [0, 1], ids=["rows", "columns"])
def test_meta_chunks_preserve_native_metadata_and_do_not_materialize_unused_empty_ranks(shape, dim):
    tensor = torch.empty(shape, device="meta")
    for count in (1, 2, 4, 16):
        native = fsdp_param._chunk_with_empty(tensor, count, dim)
        candidate = initialization._InitializationChunks(tensor, count, dim)
        assert len(candidate) == len(native)
        assert candidate._empty == {}
        for rank in range(count):
            assert _metadata(candidate[rank]) == _metadata(native[rank])
            assert candidate[rank] is candidate[rank]
        assert _metadata(candidate[-1]) == _metadata(native[-1])
        assert [_metadata(value) for value in candidate[1:3]] == [_metadata(value) for value in native[1:3]]
        with pytest.raises(IndexError, match="list index out of range"):
            candidate[count]


def test_large_world_initialization_materializes_only_requested_distinct_empty_chunks():
    chunks = initialization._InitializationChunks(torch.empty(1, 3, device="meta"), 4096, 0)
    assert len(chunks._chunks) == 1
    assert _metadata(chunks[0])[:2] == (torch.Size((1, 3)), (3, 1))
    assert chunks[4095].shape == torch.Size((0,))
    assert len(chunks._empty) == 1
    assert chunks[4094] is not chunks[4095]
    assert len(chunks._empty) == 2
    refs = [weakref.ref(value) for value in (*chunks._chunks, *chunks._empty.values())]
    del chunks
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_cpu_fallback_keeps_native_values_and_list_contract():
    tensor = torch.arange(15).reshape(5, 3)
    native = fsdp_param._chunk_with_empty
    result = initialization._initialization_chunks(tensor, 8, 0, native=native)
    assert type(result) is list
    expected = native(tensor, 8, 0)
    for actual, reference in zip(result, expected, strict=True):
        torch.testing.assert_close(actual, reference)
        assert _metadata(actual) == _metadata(reference)


def test_native_post_forward_consumer_and_exception_restore_bindings():
    original_main = fsdp_param.FSDPParam._init_sharded_param
    original_post = fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata
    original_helper = fsdp_param._chunk_with_empty
    mesh = object.__new__(fsdp_param.FSDPMeshInfo)
    mesh.shard_mesh_size = 8
    mesh.shard_mesh_rank = 7
    state = SimpleNamespace(post_forward_mesh_info=mesh, fsdp_placement=Shard(0))
    with initialization.meta_initialization():
        assert fsdp_param._chunk_with_empty is original_helper
        fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata(state, torch.empty(3, 5, device="meta"))
        assert state.sharded_post_forward_size == torch.Size((0, 5))
        assert state.contiguous_sharded_post_forward_stride == (5, 1)
    assert fsdp_param.FSDPParam._init_sharded_param is original_main
    assert fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata is original_post
    with pytest.raises(AssertionError, match="post_forward_mesh_info"), initialization.meta_initialization():
        fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata(
            SimpleNamespace(post_forward_mesh_info=None), torch.empty(3, 5, device="meta")
        )
    assert fsdp_param.FSDPParam._init_sharded_param is original_main
    assert fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata is original_post
    assert fsdp_param._chunk_with_empty is original_helper


def test_rebinding_preserves_existing_decorator_closures_defaults_and_native_body(monkeypatch):
    original = fsdp_param.FSDPParam._init_sharded_param
    observations = []

    @wraps(original)
    def outer(*args, marker="kept", **kwargs):
        observations.append(marker)
        return original(*args, **kwargs)

    monkeypatch.setattr(fsdp_param.FSDPParam, "_init_sharded_param", outer)
    with initialization.meta_initialization():
        rebound = fsdp_param.FSDPParam._init_sharded_param
        assert rebound.__kwdefaults__ == {"marker": "kept"}
        assert rebound.__wrapped__ is not original
        # Native no_grad decorator remains between the outer wrapper and body.
        assert rebound.__wrapped__.__wrapped__ is not original.__wrapped__
        assert rebound.__wrapped__.__code__ is original.__code__
        assert rebound.__wrapped__.__wrapped__.__code__ is original.__wrapped__.__code__
    assert fsdp_param.FSDPParam._init_sharded_param is outer


def test_rebinding_executes_preserved_wrapper_and_no_grad_with_private_helper(monkeypatch):
    monkeypatch.setitem(globals(), "_chunk_with_empty", fsdp_param._chunk_with_empty)
    observations = []
    tensor = torch.empty(1, 3, device="meta")

    @torch.no_grad()
    def original(value=tensor):
        observations.append(torch.is_grad_enabled())
        return _chunk_with_empty(value, 8, 0)

    @wraps(original)
    def outer(*args, marker="kept", **kwargs):
        observations.append(marker)
        return original(*args, **kwargs)

    rebound = initialization._rebind_initializer(outer)
    with torch.enable_grad():
        result = rebound()
        assert torch.is_grad_enabled()
    assert observations == ["kept", False]
    assert isinstance(result, initialization._InitializationChunks)
    assert result._empty == {}
    assert globals()["_chunk_with_empty"] is fsdp_param._chunk_with_empty


def test_repeated_scopes_release_rebound_functions_and_restore_after_failures():
    mesh_init = DeviceMesh.__init__
    originals = tuple(
        getattr(fsdp_param.FSDPParam, name)
        for name in ("_init_sharded_param", "_init_sharded_post_forward_param_metadata")
    )
    refs = []
    for index in range(20):
        try:
            with initialization.meta_initialization():
                refs.append(weakref.ref(fsdp_param.FSDPParam._init_sharded_param))
                if index % 2:
                    raise ValueError("scope failure")
        except ValueError:
            pass
        assert DeviceMesh.__init__ is mesh_init
        assert (
            fsdp_param.FSDPParam._init_sharded_param,
            fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata,
        ) == originals
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_simulate_reuses_rank_maps_and_axis_subsets_during_fsdp_initialization(caplog):
    config = SimulationConfig(world_size=8, logical_rank=7, data_parallel_replicate_degree=2, seq_len=128)
    with caplog.at_level(logging.INFO, logger=initialization.__name__):
        runner.simulate(config)
    (count,) = re.findall(r"Meta FSDP initialization shared_rank_maps=(\d+)", caplog.text)
    assert int(count) > 0
    if has_axis_cache:
        ((hits, misses),) = re.findall(r"Meta FSDP axis subset cache hits=(\d+) misses=(\d+)", caplog.text)
        assert int(hits) > 0 and int(misses) > 0


@pytest.mark.parametrize("algorithm", ["Muon", "AdamW"])
def test_simulate_restores_before_capture_and_recovers_after_parallelize_failure(monkeypatch, algorithm):
    config = SimulationConfig(
        optimizer_name=algorithm,
        muon_ns_steps=1 if algorithm == "Muon" else None,
        world_size=8,
        logical_rank=7,
        expert_parallel_degree=2,
        seq_len=128,
    )
    with monkeypatch.context() as patches:
        patches.setattr(initialization, "meta_initialization", nullcontext)
        baseline = runner.simulate(config)
    native_helper = fsdp_param._chunk_with_empty
    methods = (fsdp_param.FSDPParam._init_sharded_param, fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata)
    helper = initialization._initialization_chunks
    failed_bindings = []

    def fail_inside_native_initialization(*args, **kwargs):
        # The runner has entered parallelize and a real FSDP initializer has
        # already generated its chunks. Fail there, not before the scope opens.
        result = helper(*args, **kwargs)
        assert isinstance(result, initialization._InitializationChunks)
        active = fsdp_param.FSDPParam._init_sharded_param
        assert active is not methods[0]
        assert fsdp_param._chunk_with_empty is native_helper
        failed_bindings.append(weakref.ref(active))
        raise RuntimeError("injected native parallelize failure")

    with monkeypatch.context() as patches:
        patches.setattr(initialization, "_initialization_chunks", fail_inside_native_initialization)
        with pytest.raises(RuntimeError, match="injected native parallelize failure"):
            runner.simulate(config)
    assert failed_bindings
    assert (
        fsdp_param.FSDPParam._init_sharded_param,
        fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata,
    ) == methods
    assert fsdp_param._chunk_with_empty is native_helper
    gc.collect()
    assert all(ref() is None for ref in failed_bindings)

    chunks_seen = []
    references = []

    def observed(*args, **kwargs):
        value = helper(*args, **kwargs)
        assert fsdp_param._chunk_with_empty is native_helper
        if isinstance(value, initialization._InitializationChunks):
            chunks_seen.append(len(value))
            references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(initialization, "_initialization_chunks", observed)
    warmup = runner.warmup_training_iteration

    def checked_warmup(*args, **kwargs):
        assert (
            fsdp_param.FSDPParam._init_sharded_param,
            fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata,
        ) == methods
        return warmup(*args, **kwargs)

    monkeypatch.setattr(runner, "warmup_training_iteration", checked_warmup)
    actual = runner.simulate(config)
    assert chunks_seen and max(chunks_seen) == 8
    assert (
        fsdp_param.FSDPParam._init_sharded_param,
        fsdp_param.FSDPParam._init_sharded_post_forward_param_metadata,
    ) == methods
    before = baseline.iteration.schedule.annotations
    after = actual.iteration.schedule.annotations
    assert after["parameter_layouts"] == before["parameter_layouts"]
    assert after["memory_plan"].to_summary_dict() == before["memory_plan"].to_summary_dict()

    def signature(graph, workload):
        # Capture IDs and tensor labels are session-local; compare full geometry
        # and graph topology by insertion order rather than raw identifiers.
        positions = {}
        for templates in (workload.step_templates, workload.captured_workload.step_templates):
            for phase, item in templates.items():
                for index, op_id in enumerate(item.nodes):
                    positions.setdefault(op_id, (phase, index))

        def tensors(values):
            return [{key: value for key, value in asdict(item).items() if key != "name"} for item in values]

        return [
            (
                node.op_type,
                node.annotations["raw_op_type"],
                tensors(node.inputs),
                tensors(node.outputs),
                [positions[item] for item in node.predecessors],
                [positions[item] for item in node.successors],
            )
            for node in graph.nodes.values()
        ]

    assert actual.step_templates.keys() == baseline.step_templates.keys()
    for key, template in baseline.step_templates.items():
        assert signature(actual.step_templates[key], actual) == signature(template, baseline)
    gc.collect()
    assert all(ref() is None for ref in references)
