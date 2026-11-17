"""Tests for whole-graph profiling, cost extraction, and rank alignment."""

import operator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import whole_graph_benchmark as benchmark_module
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.utils import (
    assign_stable_node_tags,
    node_id,
)
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.whole_graph_benchmark import (
    _align_costs,
    _collective_count_manifest,
    _cross_rank_semantic_keys,
    _extract_cann_node_costs,
    _host_op_names,
    _is_host_only_scalar_node,
    _is_non_collective_device_work,
    _match_fx_nodes_to_host_scopes,
    make_calibration_runner,
)


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"name": "aclnnMul", "args": {"Task Type": "AI_VECTOR_CORE"}}, True),
        ({"name": "KERNEL_MIX_AIC", "args": {"Task Type": "MIX_AIC"}}, True),
        ({"name": "MEMCPY_ASYNC", "args": {"Task Type": "MEMCPY_ASYNC"}}, True),
        ({"name": "SDMA_SQE", "args": {"Task Type": "SDMA_SQE"}}, True),
        ({"name": "future_kernel", "args": {"Task Type": "NEW_CORE"}}, True),
        ({"name": "EVENT_WAIT", "args": {"Task Type": "EVENT_WAIT"}}, False),
        ({"name": "notify", "args": {"Task Type": "NOTIFY_WAIT_SQE"}}, False),
        ({"name": "WRITE_VALUE_SQE", "args": {}}, False),
        ({"name": "hcom_alltoallv", "args": {}}, False),
        ({"name": "task", "args": {"Task Type": "COMMUNICATION"}}, False),
    ],
)
def test_non_collective_device_work_filters_only_control_tasks(event, expected):
    assert _is_non_collective_device_work(event) is expected


def test_local_scalar_from_cpu_tensor_is_host_only():
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty((), device="cpu")
    scalar = graph.call_function(torch.ops.aten._local_scalar_dense.default, (value,))

    assert _is_host_only_scalar_node(scalar)


def test_local_scalar_from_non_cpu_tensor_is_not_host_only():
    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty((), device="meta")
    scalar = graph.call_function(torch.ops.aten._local_scalar_dense.default, (value,))

    assert not _is_host_only_scalar_node(scalar)


def test_candidate_runner_clones_inputs_and_restores_mutated_buffers(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.register_buffer("running", torch.tensor(3.0))

    @dataclass
    class TracedResult:
        gm: object

    model = Model()
    original_gm, candidate_gm = object(), object()
    traced_result = TracedResult(original_gm)
    training_input = torch.tensor([1.0])
    seen = {}

    def fake_run_traced(candidate, *, module):
        assert candidate.gm is candidate_gm
        assert module is model

        def execute(value):
            seen["value"] = value
            value.add_(10)
            model.running.add_(4)
            return value * model.weight

        return execute

    monkeypatch.setattr(benchmark_module, "run_traced", fake_run_traced)
    runner = make_calibration_runner(
        None,
        {
            "module": model,
            "traced_result": traced_result,
            "args": (training_input,),
            "train_context": nullcontext,
        },
    )

    result = runner(candidate_gm)

    assert seen["value"] is not training_input
    assert torch.equal(training_input, torch.tensor([1.0]))
    assert torch.equal(model.running, torch.tensor(3.0))
    assert torch.equal(result, torch.tensor([22.0]))
    assert traced_result.gm is original_gm


def test_candidate_runner_restores_mutated_buffers_after_failure(monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running", torch.tensor(3.0))

    @dataclass
    class TracedResult:
        gm: object

    model = Model()

    def fake_run_traced(_candidate, *, module):
        assert module is model

        def execute(_value):
            model.running.add_(4)
            raise RuntimeError("candidate failed")

        return execute

    monkeypatch.setattr(benchmark_module, "run_traced", fake_run_traced)
    runner = make_calibration_runner(
        None,
        {
            "module": model,
            "traced_result": TracedResult(object()),
            "args": (torch.tensor([1.0]),),
            "train_context": nullcontext,
        },
    )

    with pytest.raises(RuntimeError, match="candidate failed"):
        runner(object())

    assert torch.equal(model.running, torch.tensor(3.0))


def test_host_op_name_uses_dispatcher_schema():
    gm = torch.fx.symbolic_trace(lambda x: torch.ops.aten.mul.Tensor(x, x))
    work = next(node for node in gm.graph.nodes if node.op == "call_function")
    assert _host_op_names(work) == ("aten::mul",)


def test_python_getitem_has_no_native_host_scope_expectation():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    item = graph.call_function(operator.getitem, (x, 0))
    graph.output(item)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)

    assert _host_op_names(item) == ()
    scopes, stats = _match_fx_nodes_to_host_scopes([], gm)
    assert scopes == {}
    assert stats["expected"] == stats["matched"] == 0


def _unpermute_grad_wrapper_target(value):
    return value


_unpermute_grad_wrapper_target._schema = SimpleNamespace(name="torchtitan_npu::npu_moe_token_unpermute_grad_unweighted")


def _graph_with_unpermute_grad_wrappers(count):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    before = graph.call_function(torch.ops.aten.neg.default, (x,))
    wrappers = []
    value = before
    for _ in range(count):
        value = graph.call_function(_unpermute_grad_wrapper_target, (value,))
        wrappers.append(value)
    after = graph.call_function(torch.ops.aten.relu.default, (value,))
    graph.output(after)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    return gm, before, wrappers, after


def _within_calibration_scope(events):
    begin = min(float(event["ts"]) for event in events) - 1
    end = max(float(event["ts"]) + float(event["dur"]) for event in events) + 1
    scope = {
        "ph": "X",
        "cat": "user_annotation",
        "name": benchmark_module._CALIBRATION_GRAPH_SCOPE,
        "pid": events[0]["pid"],
        "tid": events[0]["tid"],
        "ts": begin,
        "dur": end - begin,
    }
    return [scope, *events]


def test_host_order_match_uses_unpermute_grad_native_alias():
    gm, before, wrappers, after = _graph_with_unpermute_grad_wrappers(1)
    trace = _within_calibration_scope(
        [
            {"ph": "X", "cat": "cpu_op", "name": "aten::neg", "pid": 1, "tid": 10, "ts": 1, "dur": 1},
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "npu::npu_moe_token_unpermute_grad",
                "pid": 1,
                "tid": 10,
                "ts": 2,
                "dur": 1,
            },
            {"ph": "X", "cat": "cpu_op", "name": "aten::relu", "pid": 1, "tid": 10, "ts": 3, "dur": 1},
        ]
    )

    scopes, stats = _match_fx_nodes_to_host_scopes(trace, gm)

    assert stats["matched"] == stats["expected"] == 3
    assert scopes[node_id(before)]["ts"] == 1
    assert scopes[node_id(wrappers[0])]["ts"] == 2
    assert scopes[node_id(after)]["ts"] == 3


def test_host_order_match_does_not_shift_partial_aliases():
    gm, before, wrappers, after = _graph_with_unpermute_grad_wrappers(2)
    trace = _within_calibration_scope(
        [
            {"ph": "X", "cat": "cpu_op", "name": "aten::neg", "pid": 1, "tid": 10, "ts": 1, "dur": 1},
            {
                "ph": "X",
                "cat": "cpu_op",
                "name": "npu::npu_moe_token_unpermute_grad",
                "pid": 1,
                "tid": 10,
                "ts": 2,
                "dur": 1,
            },
            {"ph": "X", "cat": "cpu_op", "name": "aten::relu", "pid": 1, "tid": 10, "ts": 3, "dur": 1},
        ]
    )

    scopes, stats = _match_fx_nodes_to_host_scopes(trace, gm)

    assert stats["expected"] == 4
    assert stats["matched"] == 2
    assert set(scopes) == {node_id(before), node_id(after)}
    assert all(node_id(wrapper) not in scopes for wrapper in wrappers)


def test_host_order_match_uses_full_fx_sequence_and_native_scopes():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
    second = graph.call_function(torch.ops.aten.mul.Tensor, (first, x))
    graph.output(second)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    trace = _within_calibration_scope(
        [
            {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "pid": 1, "tid": 10, "ts": 1, "dur": 2},
            # An inner aclnn scope must not be treated as another FX call.
            {"ph": "X", "cat": "cpu_op", "name": "aclnnMul", "pid": 1, "tid": 10, "ts": 1.5, "dur": 1},
            {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "pid": 1, "tid": 10, "ts": 4, "dur": 2},
        ]
    )
    scopes, stats = _match_fx_nodes_to_host_scopes(trace, gm)
    assert stats["matched"] == stats["expected"] == 2
    assert scopes[node_id(first)]["ts"] == 1
    assert scopes[node_id(second)]["ts"] == 4


def test_repeated_profiles_replay_rng_and_release_outputs(monkeypatch):
    import weakref

    gm = torch.fx.symbolic_trace(lambda x: -x)
    refs, inputs_seen, profile_rounds = [], [], []
    npu_state = torch.tensor([123], dtype=torch.uint8)

    def set_npu_state(state):
        nonlocal npu_state
        npu_state = state.clone()

    fake_npu = SimpleNamespace(
        current_device=lambda: 0,
        get_rng_state=lambda: npu_state.clone(),
        set_rng_state=set_npu_state,
        synchronize=lambda: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(benchmark_module.torch_npu, "npu", fake_npu)

    @contextmanager
    def fork_rng(**kwargs):
        assert kwargs == {"devices": [0], "device_type": "npu"}
        cpu, npu = torch.random.get_rng_state().clone(), npu_state.clone()
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu)
            set_npu_state(npu)

    def candidate(graph):
        assert graph is gm
        assert all(ref() is None for ref in refs)
        inputs_seen.append((torch.rand(2), npu_state.clone()))
        set_npu_state(npu_state + 1)
        result = torch.ones(2)
        refs.append(weakref.ref(result))
        return result

    def local_profile(graph, runner, ids, calibration_round, debug):
        assert debug is False
        profile_rounds.append(calibration_round)
        return {}, runner(graph)

    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(benchmark_module, "_validate_collective_counts_across_ranks", lambda gm: None)
    monkeypatch.setattr(benchmark_module, "_local_profile_costs", local_profile)
    monkeypatch.setattr(benchmark_module, "_align_costs", lambda gm, costs, ids: costs)
    before = torch.random.get_rng_state().clone()
    for number in (1, 2):
        benchmark_module.profile_whole_graph_costs(gm, candidate, calibration_round=number)
        assert torch.equal(torch.random.get_rng_state(), before)
        assert npu_state.item() == 123
    assert profile_rounds == [1, 2]
    assert len(inputs_seen) == 8  # Each round: three warmups + one measurement.
    assert all(torch.equal(cpu, inputs_seen[0][0]) and torch.equal(npu, inputs_seen[0][1]) for cpu, npu in inputs_seen)
    assert all(ref() is None for ref in refs)


def test_whole_graph_profile_coordinates_local_failure(monkeypatch):
    gm = torch.fx.symbolic_trace(lambda x: -x)
    npu_state = torch.tensor([123], dtype=torch.uint8)
    fake_npu = SimpleNamespace(
        current_device=lambda: 0,
        get_rng_state=lambda: npu_state.clone(),
        set_rng_state=lambda _state: None,
        synchronize=lambda: None,
    )
    gathered_errors = []

    def gather(output, local_error):
        gathered_errors.append(local_error)
        output[:] = [None, local_error]

    def local_profile(*_args, **_kwargs):
        raise ValueError("bad trace")

    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(benchmark_module.torch_npu, "npu", fake_npu)
    monkeypatch.setattr(torch.random, "fork_rng", lambda **_kwargs: nullcontext())
    monkeypatch.setattr(
        benchmark_module,
        "_validate_collective_counts_across_ranks",
        lambda _gm: None,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_local_profile_costs",
        local_profile,
    )
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)

    result = benchmark_module.profile_whole_graph_costs(
        gm,
        lambda _gm: torch.ones(1),
    )

    assert result is None
    assert gathered_errors == ["ValueError: bad trace"]


@pytest.mark.parametrize("collective", [False, True])
def test_align_costs_uses_rank_minimum(monkeypatch, collective):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    if collective:
        work = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (x, "sum", "test_group"))
    else:
        work = graph.call_function(torch.ops.aten.neg.default, (x,))
    missing = graph.call_function(torch.ops.aten.neg.default, (work,))
    graph.output(missing)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    work_id, missing_id = node_id(work), node_id(missing)
    semantic_keys = _cross_rank_semantic_keys(gm, None)
    work_key, missing_key = semantic_keys[work_id], semantic_keys[missing_id]
    gathered = [{work_key: value, missing_key: 2.0} for value in [9.0, 3.0, 7.0, 5.0]]
    del gathered[-1][missing_key]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 4)

    def gather(output, local):
        output[:] = [local] * 4 if isinstance(local, list) else gathered

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    for local_value in [9.0, 3.0, 7.0, 5.0]:
        local = {work_id: local_value, missing_id: 2.0}
        aligned = _align_costs(gm, local)
        assert aligned == {work_id: 3.0}
        assert missing_id not in aligned  # Preserve unanimous first-round fallback.


def test_align_costs_without_distributed_keeps_local_cost(monkeypatch):
    gm = torch.fx.symbolic_trace(lambda x: -x)
    assign_stable_node_tags(gm)
    work = next(node for node in gm.graph.nodes if node.op == "call_function")
    local = {node_id(work): 0.25}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    assert _align_costs(gm, local) == local


def test_align_costs_batches_layers_with_shared_rank_order(monkeypatch):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    nodes = []
    # Deliberately reverse layer order; include both phases and chunks.
    for layer, backward, chunk in [(1, False, 0), (0, False, 0), (0, True, 1)]:
        node = graph.call_function(torch.ops.aten.neg.default, (x,))
        node.meta.update(
            chunked_region_fqn=f"layers.{layer}.moe",
            chunked_region_is_backward=backward,
            chunk_id=chunk,
        )
        nodes.append(node)
    graph.output(tuple(nodes))
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    keys = _cross_rank_semantic_keys(gm, None)
    ids = [node_id(node) for node in nodes]
    local = dict.fromkeys(ids, 9.0)
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, payload):
        calls.append(payload.copy())
        if isinstance(payload, list):
            # The peer has an extra layer absent from this rank.
            output[:] = [payload, ["layers.2.moe", "layers.0.moe", "layers.1.moe"]]
        elif len(calls) == 2:
            assert set(payload) == {keys[ids[1]], keys[ids[2]]}
            # Missing backward measurement must retain fallback.
            output[:] = [payload, {keys[ids[1]]: 3.0}]
        elif len(calls) == 3:
            assert set(payload) == {keys[ids[0]]}
            output[:] = [payload, {keys[ids[0]]: 5.0}]
        else:
            assert payload == {}  # Participate even without local nodes.
            output[:] = [payload, {("peer_only",): 1.0}]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    assert _align_costs(gm, local) == {ids[1]: 3.0, ids[0]: 5.0}
    assert len(calls) == 4
    assert calls[0] == ["layers.0.moe", "layers.1.moe"]


def test_collective_count_manifest_groups_fx_launches_by_type():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (x, "sum", "group_a"))
    second = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (first, "sum", "group_a"))
    third = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (second, "sum", "group_b"))
    graph.output(third)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)

    collective = benchmark_module._collective_type(first)
    assert _collective_count_manifest(gm) == ((0, collective, 2), (1, collective, 1))


def test_collective_count_validation_only_compares_group_type_counts(monkeypatch):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    work = graph.call_function(torch.ops._c10d_functional.all_reduce.default, (x, "sum", "group_a"))
    graph.output(work)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    manifest = _collective_count_manifest(gm)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, local: output.__setitem__(slice(None), [local, local])
    )
    benchmark_module._validate_collective_counts_across_ranks(gm)

    different = ((manifest[0][0], manifest[0][1], manifest[0][2] + 1),)
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda output, local: output.__setitem__(slice(None), [local, different]),
    )
    with pytest.raises(RuntimeError, match="collective counts differ"):
        benchmark_module._validate_collective_counts_across_ranks(gm)


def test_compute_semantic_key_ignores_unrelated_metadata_node():
    def make_graph(*, extra_metadata):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty((4, 8), device="meta")
        if extra_metadata:
            metadata = graph.call_function(operator.getitem, ((x, x), 0))
            metadata.meta["val"] = x.meta["val"]
        work = graph.call_function(torch.ops.aten.neg.default, (x,))
        work.meta["val"] = torch.empty((4, 8), device="meta")
        graph.output(work)
        gm = torch.fx.GraphModule({}, graph)
        assign_stable_node_tags(gm)
        stable_id = node_id(work)
        return gm, stable_id

    plain, plain_id = make_graph(extra_metadata=False)
    changed, changed_id = make_graph(extra_metadata=True)
    plain_key = _cross_rank_semantic_keys(plain, frozenset({plain_id}))[plain_id]
    changed_key = _cross_rank_semantic_keys(changed, frozenset({changed_id}))[changed_id]
    assert plain_id != changed_id
    assert plain_key == changed_key


def test_compute_semantic_key_numbers_duplicate_signatures():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.empty((4, 8), device="meta")
    first = graph.call_function(torch.ops.aten.neg.default, (x,))
    first.meta["val"] = torch.empty((4, 8), device="meta")
    second = graph.call_function(torch.ops.aten.neg.default, (first,))
    second.meta["val"] = torch.empty((4, 8), device="meta")
    graph.output(second)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    keys = _cross_rank_semantic_keys(gm, frozenset({node_id(first), node_id(second)}))

    assert keys[node_id(first)][:-1] == keys[node_id(second)][:-1]
    assert keys[node_id(first)][-1] == 0
    assert keys[node_id(second)][-1] == 1


def _profile_fixture(target=None):
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    node = graph.call_function(target or torch.ops.aten.mul.Tensor, (x, x))
    graph.output(node)
    gm = torch.fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    stable_id = node_id(node)
    host_scope = {
        "ph": "X",
        "name": _host_op_names(node)[0],
        "cat": "cpu_op",
        "pid": 1,
        "tid": 10,
        "ts": 0,
        "dur": 10,
    }
    events = [
        {"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "Python"}},
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}},
        {"ph": "M", "name": "process_name", "pid": 3, "args": {"name": "Communication"}},
        host_scope,
    ]
    return events, {stable_id: node}, {stable_id: host_scope}, stable_id


def _add_device_flow(events, flow_id, ts, dur, name="Mul", task_type="KERNEL_AIVEC", pid=2, finish_offset=0):
    events.extend(
        [
            {"ph": "s", "cat": "async_npu", "name": "torch_to_npu", "id": flow_id, "pid": 1, "tid": 10, "ts": 5},
            {
                "ph": "f",
                "cat": "async_npu",
                "name": "torch_to_npu",
                "id": flow_id,
                "pid": pid,
                "tid": 47,
                "ts": ts + finish_offset,
            },
            {"ph": "X", "name": name, "pid": pid, "tid": 47, "ts": ts, "dur": dur, "args": {"Task Type": task_type}},
        ]
    )


@pytest.mark.parametrize("host_flow_id", [1, 77])
@pytest.mark.parametrize("wrapped", [False, True])
def test_extract_compute_ignores_host_dequeue_and_flow_id_collisions(host_flow_id, wrapped):
    events, nodes, scopes, stable_id = _profile_fixture()
    _add_device_flow(events, 1, 30000, 100, name="aclnnMul_MulAiCore_Mul")
    # Actual 0826 failure: CPU Dequeue@aclnnMul lasts ~20ms while the device
    # Mul lasts only ~100us. Queue IDs may also collide with another flow kind.
    events.extend(
        [
            {
                "ph": "s",
                "cat": "async_task_queue",
                "name": "enqueue_to_dequeue",
                "id": host_flow_id,
                "pid": 1,
                "tid": 10,
                "ts": 5,
            },
            {
                "ph": "f",
                "cat": "async_task_queue",
                "name": "enqueue_to_dequeue",
                "id": host_flow_id,
                "pid": 1,
                "tid": 11,
                "ts": 1000,
            },
            {"ph": "X", "cat": "dequeue", "name": "Dequeue@aclnnMul", "pid": 1, "tid": 11, "ts": 1000, "dur": 20000},
        ]
    )
    trace = {"traceEvents": events} if wrapped else events
    assert _extract_cann_node_costs(trace, nodes, scopes) == {stable_id: pytest.approx(0.1)}


@pytest.mark.parametrize("finish_offset", [0, 0.01])
def test_extract_requires_device_process_for_exact_and_containment_matches(finish_offset):
    events, nodes, scopes, _ = _profile_fixture()
    # Even an async_npu flow cannot authorize treating an untyped host range
    # as a device task. Test exact lookup and containment fallback alike.
    _add_device_flow(events, 1, 1000, 20000, name="host_range", pid=1, finish_offset=finish_offset)
    assert _extract_cann_node_costs(events, nodes, scopes) == {}


def test_extract_compute_unions_kernels_excludes_gaps_and_control_waits():
    events, nodes, scopes, stable_id = _profile_fixture()
    _add_device_flow(events, 1, 100, 100)
    _add_device_flow(events, 2, 150, 100, name="future_kernel", task_type="NEW_CORE")
    _add_device_flow(events, 3, 300, 50, name="MEMCPY_ASYNC", task_type="SDMA_SQE", finish_offset=0.01)
    _add_device_flow(events, 4, 80, 20000, name="EVENT_WAIT", task_type="EVENT_WAIT")
    assert _extract_cann_node_costs(events, nodes, scopes) == {stable_id: pytest.approx(0.2)}


def test_extract_costs_from_native_host_scope():
    events, nodes, scopes, stable_id = _profile_fixture()
    _add_device_flow(events, 1, 100, 125)

    assert _extract_cann_node_costs(
        events,
        nodes,
        scopes,
    ) == {stable_id: pytest.approx(0.125)}


def test_extract_collective_keeps_full_hcom_interval():
    events, nodes, scopes, stable_id = _profile_fixture(torch.ops._c10d_functional.all_reduce.default)
    _add_device_flow(events, 1, 100, 2000, name="hcom_allReduce", task_type="COMMUNICATION", pid=3)
    _add_device_flow(events, 2, 100, 1500, name="NOTIFY_WAIT", task_type="NOTIFY_WAIT")
    assert _extract_cann_node_costs(events, nodes, scopes) == {stable_id: pytest.approx(2.0)}


@pytest.mark.parametrize("case", ["no_flow", "control_only", "missing_finish", "missing_metadata"])
def test_extract_missing_device_work_falls_back_instead_of_zero(case):
    events, nodes, scopes, _ = _profile_fixture()
    if case == "control_only":
        _add_device_flow(events, 1, 100, 20000, name="EVENT_WAIT", task_type="EVENT_WAIT")
    elif case == "missing_finish":
        _add_device_flow(events, 1, 100, 100)
        events = [e for e in events if e.get("ph") != "f"]
    elif case == "missing_metadata":
        _add_device_flow(events, 1, 100, 100)
        events = [e for e in events if e.get("ph") != "M"]
    assert _extract_cann_node_costs(events, nodes, scopes) == {}


def test_extract_proven_metadata_node_keeps_zero_cost():
    events, nodes, scopes, stable_id = _profile_fixture(torch.ops.aten.alias.default)
    assert _extract_cann_node_costs(events, nodes, scopes) == {stable_id: 0.0}
