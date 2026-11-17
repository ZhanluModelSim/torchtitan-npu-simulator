"""Tests for MoE auto-overlap cost collection and greedy scheduling."""

import gc
import operator
import weakref
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import npu_moe_auto_scheduler as scheduler_module
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.compute_benchmark import (
    benchmark_npu_compute_node,
    make_npu_do_bench,
)
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.npu_moe_auto_scheduler import (
    NpuMoeAutoOverlapScheduler,
    _Exchange,
    _exchange_label,
    _is_token_count_sync_copy,
    _is_zero_cost_schedulable_node,
)
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.utils import (
    assign_stable_node_tags,
    is_metadata_only_compute_node,
    node_id,
)


def test_operator_getitem_is_metadata_only():
    graph = torch.fx.Graph()
    values = graph.placeholder("values")
    node = graph.call_function(operator.getitem, (values, 0))

    assert is_metadata_only_compute_node(node)


@pytest.mark.parametrize("group_size", [4, 8])
def test_scheduler_requires_world_communication_groups(monkeypatch, group_size):
    gm, _ = _build_two_chunk_graph()
    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        align_across_ranks=False,
    )
    process_group = object()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: group_size if group is process_group else 8,
    )
    monkeypatch.setattr(
        scheduler_module,
        "resolve_collective_process_group",
        lambda _node: process_group,
    )

    if group_size == 8:
        scheduler._validate_world_communication_groups()
    else:
        with pytest.raises(RuntimeError, match="span WORLD"):
            scheduler._validate_world_communication_groups()


@pytest.mark.parametrize("profile_costs", [False, True])
def test_profile_costs_include_nodes_rejected_by_standalone_benchmark(profile_costs):
    gm, refs = _build_two_chunk_graph()
    assign_stable_node_tags(gm)
    # Integer inputs are deliberately excluded from random-input benchmarking.
    setup = refs["setup0"]
    setup.args[0].meta["val"] = torch.empty((4, 8), dtype=torch.int32, device="meta")
    measured = []
    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda node: measured.append(node) or 0.15625,
        collective_cost_fn=lambda node: 1.0,
        align_across_ranks=False,
        use_profile_compute_costs=profile_costs,
    )
    scheduler.run()
    assert (setup in measured) == profile_costs
    assert scheduler.costs_by_node_id[node_id(setup)] == (0.15625 if profile_costs else 0.0)


def test_generic_benchmark_cache_key_includes_copy_kwargs(monkeypatch):
    from torch._inductor.fx_passes import overlap_scheduling

    graph = torch.fx.Graph()
    value = graph.placeholder("value")
    value.meta["val"] = torch.empty((4, 8), device="meta")
    async_copy = graph.call_function(
        torch.ops.aten._to_copy.default,
        (value,),
        {"device": torch.device("cpu"), "non_blocking": True},
    )
    blocking_copy = graph.call_function(
        torch.ops.aten._to_copy.default,
        (value,),
        {"device": torch.device("cpu"), "non_blocking": False},
    )
    cache = {}
    calls = []
    monkeypatch.setattr(overlap_scheduling, "get_cached_node_time", cache.get)
    monkeypatch.setattr(
        overlap_scheduling,
        "set_cached_node_time",
        lambda key, value: cache.__setitem__(key, value),
    )

    first = benchmark_npu_compute_node(
        async_copy,
        lambda _callable: calls.append("async") or 1.0,
    )
    second = benchmark_npu_compute_node(
        blocking_copy,
        lambda _callable: calls.append("blocking") or 2.0,
    )

    assert first is not None and second is not None
    assert first[1] != second[1]
    assert calls == ["async", "blocking"]


def test_auto_scheduler_uses_profiler_for_compute_and_events_for_collective(
    monkeypatch,
):
    from torch._inductor.runtime.benchmarking import TorchProfilerBenchmarker

    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import collective_benchmark

    calls = []

    def fake_benchmark_gpu(self, benchmark_callable, **kwargs):
        calls.append((self, benchmark_callable(), kwargs))
        return 1.75

    monkeypatch.setattr(
        TorchProfilerBenchmarker,
        "benchmark_gpu",
        fake_benchmark_gpu,
    )
    event_calls = []

    def fake_event_benchmark(node, nruns=2, **kwargs):
        event_calls.append((node, nruns, kwargs))
        return 2.25, "event-key"

    monkeypatch.setattr(
        collective_benchmark,
        "benchmark_collective_with_npu_events",
        fake_event_benchmark,
    )
    gm, refs = _build_two_chunk_graph()
    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        align_across_ranks=False,
    )

    assert make_npu_do_bench()(lambda: "ran") == 1.75
    assert isinstance(calls[0][0], TorchProfilerBenchmarker)
    assert calls[0][1] == "ran"
    assert calls[0][2] == {
        "rep": 5,
        "estimation_iters": 2,
        "memory_warmup_iters": 1,
        "device_type": "npu",
    }

    assert scheduler._benchmark_collective(refs["launch0"]) == 2.25
    assert event_calls == [
        (
            refs["launch0"],
            5,
            {
                "uniform_dispatch_rows": None,
                "reverse_uniform_splits": False,
            },
        )
    ]
    assert len(calls) == 1


def _mark_region(node, *, chunk_id: int):
    node.meta.update(
        {
            "chunked_region_role": "body",
            "chunked_region_fqn": "layers.0.moe",
            "chunked_region_is_backward": False,
            "chunked_region_producer": "graph",
            "chunk_id": chunk_id,
        }
    )
    return node


def _build_two_chunk_graph(*, external_bridge=False, ready_filler=False):
    graph = torch.fx.Graph()
    x0 = graph.placeholder("x0")
    x1 = graph.placeholder("x1")
    weight = graph.placeholder("weight")
    for node, value in (
        (x0, torch.empty((4, 8), device="meta")),
        (x1, torch.empty((4, 8), device="meta")),
        (weight, torch.empty((8, 8), device="meta")),
    ):
        node.meta["val"] = value

    setup0 = _mark_region(graph.call_function(torch.clone, (x0,)), chunk_id=0)
    setup0.meta["val"] = torch.empty((4, 8), device="meta")
    launch0 = _mark_region(
        graph.call_function(
            torch.ops._c10d_functional.all_to_all_single.default,
            (setup0, [], [], "ep"),
        ),
        chunk_id=0,
    )
    launch0.meta["custom"] = {"EP_token_exchange": "dispatch"}
    launch0.meta["val"] = torch.empty((4, 8), device="meta")
    wait0 = _mark_region(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            (launch0,),
        ),
        chunk_id=0,
    )
    wait0.meta["val"] = torch.empty((4, 8), device="meta")
    compute0 = _mark_region(
        graph.call_function(torch.ops.aten.matmul.default, (wait0, weight)),
        chunk_id=0,
    )
    compute0.meta["val"] = torch.empty((4, 8), device="meta")

    if external_bridge:
        bridge = graph.call_function(torch.clone, (compute0,))
        bridge.meta["val"] = torch.empty((4, 8), device="meta")
        setup1_input = graph.call_function(torch.ops.aten.add.Tensor, (x1, bridge))
        setup1_input.meta["val"] = torch.empty((4, 8), device="meta")
    else:
        setup1_input = x1
    setup1 = _mark_region(graph.call_function(torch.clone, (setup1_input,)), chunk_id=1)
    setup1.meta["val"] = torch.empty((4, 8), device="meta")
    filler = None
    if ready_filler:
        filler = _mark_region(
            graph.call_function(torch.ops.aten.matmul.default, (x0, weight)),
            chunk_id=0,
        )
        filler.meta["val"] = torch.empty((4, 8), device="meta")
    launch1 = _mark_region(
        graph.call_function(
            torch.ops._c10d_functional.all_to_all_single.default,
            (setup1, [], [], "ep"),
        ),
        chunk_id=1,
    )
    launch1.meta["custom"] = {"EP_token_exchange": "dispatch"}
    launch1.meta["val"] = torch.empty((4, 8), device="meta")
    wait1 = _mark_region(
        graph.call_function(
            torch.ops._c10d_functional.wait_tensor.default,
            (launch1,),
        ),
        chunk_id=1,
    )
    wait1.meta["val"] = torch.empty((4, 8), device="meta")
    compute1 = _mark_region(
        graph.call_function(torch.ops.aten.matmul.default, (wait1, weight)),
        chunk_id=1,
    )
    compute1.meta["val"] = torch.empty((4, 8), device="meta")
    result = graph.call_function(torch.ops.aten.add.Tensor, (compute0, compute1))
    if filler is not None:
        result = graph.call_function(torch.ops.aten.add.Tensor, (result, filler))
    graph.output(result)
    gm = torch.fx.GraphModule({}, graph)
    return gm, {
        "setup0": setup0,
        "launch0": launch0,
        "wait0": wait0,
        "compute0": compute0,
        "setup1": setup1,
        "filler": filler,
        "launch1": launch1,
        "wait1": wait1,
        "compute1": compute1,
    }


def test_uniform_a2a_assumption_reuses_dispatch_rows_for_combine():
    gm, refs = _build_two_chunk_graph()
    refs["launch1"].meta["chunk_id"] = 0
    refs["launch1"].meta["custom"] = {"EP_token_exchange": "combine"}
    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        align_across_ranks=False,
    )
    exchanges = (
        _Exchange("dispatch", refs["launch0"], ()),
        _Exchange("combine", refs["launch1"], ()),
    )

    scheduler._set_uniform_a2a_assumptions(scheduler.regions[0], exchanges)

    assert scheduler._uniform_a2a_assumptions == {
        refs["launch0"]: (4, False),
        refs["launch1"]: (4, True),
    }


def test_uniform_a2a_assumption_accepts_multiple_dispatch_dtypes():
    gm, refs = _build_two_chunk_graph()
    refs["launch1"].meta["chunk_id"] = 0
    refs["setup1"].meta["val"] = torch.empty((4, 1), dtype=torch.float32, device="meta")
    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        align_across_ranks=False,
    )
    exchanges = (
        _Exchange("dispatch", refs["launch0"], ()),
        _Exchange("dispatch", refs["launch1"], ()),
    )

    scheduler._set_uniform_a2a_assumptions(scheduler.regions[0], exchanges)

    assert scheduler._uniform_a2a_assumptions == {
        refs["launch0"]: (4, False),
        refs["launch1"]: (4, False),
    }


def test_uniform_a2a_assumption_rejects_different_dispatch_rows():
    gm, refs = _build_two_chunk_graph()
    refs["launch1"].meta["chunk_id"] = 0
    refs["setup1"].meta["val"] = torch.empty((5, 1), dtype=torch.float32, device="meta")
    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        align_across_ranks=False,
    )
    exchanges = (
        _Exchange("dispatch", refs["launch0"], ()),
        _Exchange("dispatch", refs["launch1"], ()),
    )

    scheduler._set_uniform_a2a_assumptions(scheduler.regions[0], exchanges)

    assert scheduler._uniform_a2a_assumptions == {}


def test_permutation_assumptions_use_count_exchange_expert_dimension(monkeypatch):
    from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer import npu_moe_auto_scheduler as module

    gm, refs = _build_two_chunk_graph()
    monkeypatch.setattr(module, "PERMUTATION_TARGETS", {str(refs["setup0"].target)})
    graph = gm.graph
    with graph.inserting_before(refs["setup0"]):
        counts = graph.placeholder("counts")
        counts.meta["val"] = torch.empty((8, 2), dtype=torch.int64, device="meta")
        count = graph.call_function(torch.ops._c10d_functional.all_to_all_single.default, (counts, [], [], "ep"))
        count.meta["chunk_id"] = 0
    scheduler = NpuMoeAutoOverlapScheduler(gm, module_pattern="layers.*.moe", align_across_ranks=False)
    scheduler._set_uniform_a2a_assumptions(
        scheduler.regions[0],
        (
            _Exchange("count", count, ()),
            _Exchange("dispatch", refs["launch0"], ()),
        ),
    )
    assert scheduler._uniform_permutation_assumptions[refs["setup0"]] == (4, 16)


def test_ready_collective_launches_before_unrelated_compute():
    gm, refs = _build_two_chunk_graph(ready_filler=True)
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda _node: 8.0,
        collective_cost_fn=lambda _node: 10.0,
        align_across_ranks=False,
    ).run()

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    assert order[refs["launch0"]] < order[refs["launch1"]]
    assert order[refs["launch1"]] < order[refs["filler"]]


@pytest.mark.parametrize(
    "communication,preparation,filler,move",
    [
        (11.0, 2.0, 6.0, True),
        (10.0, 2.0, 6.0, False),
        (8.0, 2.0, 6.0, False),
        (1.0, 2.0, 0.5, False),
        (0.0, 2.0, 0.5, False),
    ],
)
def test_slack_filler_reserves_next_launch_preparation(communication, preparation, filler, move):
    gm, refs = _build_two_chunk_graph(ready_filler=True)
    costs = {refs["setup1"]: preparation, refs["filler"]: filler}
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda node: costs.get(node, 1.0),
        collective_cost_fn=lambda _node: communication,
        align_across_ranks=False,
    ).run()
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    assert order[refs["launch0"]] < order[refs["launch1"]]
    assert (order[refs["filler"]] < order[refs["setup1"]]) == move
    if not move:
        assert order[refs["launch1"]] < order[refs["filler"]]


@pytest.mark.parametrize("communication,move_smaller", [(12.0, False), (14.0, True)])
def test_slack_filler_moves_metadata_setup_and_rechecks_budget(communication, move_smaller):
    gm, refs = _build_two_chunk_graph(ready_filler=True)
    filler = refs["filler"]
    x, weight = filler.args
    with gm.graph.inserting_before(filler):
        transpose = _mark_region(gm.graph.call_function(torch.ops.aten.t.default, (weight,)), chunk_id=0)
    filler.args = (x, transpose)
    with gm.graph.inserting_after(filler):
        smaller = _mark_region(gm.graph.call_function(torch.ops.aten.matmul.default, (x, weight)), chunk_id=0)
        smaller.meta["val"] = filler.meta["val"]
    output = next(node for node in gm.graph.nodes if node.op == "output")
    with gm.graph.inserting_before(output):
        result = gm.graph.call_function(torch.ops.aten.add.Tensor, (output.args[0], smaller))
    output.args = (result,)
    costs = {refs["setup1"]: 2.0, filler: 6.0, smaller: 3.0}
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda node: costs.get(node, 1.0),
        collective_cost_fn=lambda _node: communication,
        align_across_ranks=False,
    ).run()
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    assert order[refs["launch0"]] < order[transpose] < order[filler] < order[refs["setup1"]]
    # After six units of filler, reserve twice the two-unit preparation cost.
    # Only the larger communication window has room for another three units.
    assert (order[smaller] < order[refs["setup1"]]) == move_smaller
    if not move_smaller:
        assert order[refs["launch1"]] < order[smaller]
    gm.graph.lint()


def test_slack_filler_reserves_all_launch_ancestors():
    gm, refs = _build_two_chunk_graph(ready_filler=True)
    setup = refs["setup1"]
    with gm.graph.inserting_before(setup):
        other = _mark_region(gm.graph.call_function(torch.clone, (setup.args[0],)), chunk_id=1)
        other.meta["val"] = setup.meta["val"]
    # Two independent compute branches both occupy the same compute stream.
    with gm.graph.inserting_before(refs["launch1"]):
        merged = _mark_region(gm.graph.call_function(torch.ops.aten.add.Tensor, (setup, other)), chunk_id=1)
        merged.meta["val"] = setup.meta["val"]
    refs["launch1"].replace_input_with(setup, merged)
    costs = {setup: 3.0, other: 3.0, merged: 1.0, refs["filler"]: 4.0}
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda node: costs.get(node, 1.0),
        collective_cost_fn=lambda _node: 10.0,
        align_across_ranks=False,
    ).run()
    order = {node: i for i, node in enumerate(gm.graph.nodes)}
    assert order[refs["launch1"]] < order[refs["filler"]]


def _build_count_and_data_a2a_graph(*, ready_filler=False):
    graph = torch.fx.Graph()
    x0 = graph.placeholder("x0")
    x1 = graph.placeholder("x1")
    weight = graph.placeholder("weight")
    for node, value in (
        (x0, torch.empty((4, 8), device="meta")),
        (x1, torch.empty((4, 8), device="meta")),
        (weight, torch.empty((8, 8), device="meta")),
    ):
        node.meta["val"] = value

    refs = {}
    for chunk_id, x in enumerate((x0, x1)):
        count = _mark_region(
            graph.call_function(
                torch.ops._c10d_functional.all_to_all_single.default,
                (x, [], [], "ep"),
            ),
            chunk_id=chunk_id,
        )
        count.meta["custom"] = {"EP_token_count_exchange": "dispatch"}
        count.meta["val"] = torch.empty((4, 8), device="meta")
        refs[chunk_id] = {"count": count}

    filler = None
    if ready_filler:
        filler = _mark_region(
            graph.call_function(torch.ops.aten.matmul.default, (x0, weight)),
            chunk_id=0,
        )
        filler.meta["val"] = torch.empty((4, 8), device="meta")

    outputs = []
    for chunk_id, x in enumerate((x0, x1)):
        count = refs[chunk_id]["count"]
        count_wait = _mark_region(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                (count,),
            ),
            chunk_id=chunk_id,
        )
        count_wait.meta["val"] = torch.empty((4, 8), device="meta")
        d2hs = []
        for copy_index in range(2):
            d2h = _mark_region(
                graph.call_function(
                    torch.ops.aten._to_copy.default,
                    (count_wait,),
                    {
                        "device": torch.device("cpu"),
                        "non_blocking": copy_index == 0,
                    },
                ),
                chunk_id=chunk_id,
            )
            d2h.meta["custom"] = {"EP_token_count_sync": "dispatch"}
            d2h.meta["val"] = torch.empty((4, 8), device="cpu")
            d2hs.append(d2h)
        setup = _mark_region(graph.call_function(torch.clone, (x,)), chunk_id=chunk_id)
        setup.meta["val"] = torch.empty((4, 8), device="meta")
        data = _mark_region(
            graph.call_function(
                torch.ops._c10d_functional.all_to_all_single.default,
                (setup, d2hs[0], d2hs[1], "ep"),
            ),
            chunk_id=chunk_id,
        )
        data.meta["custom"] = {"EP_token_exchange": "dispatch"}
        data.meta["val"] = torch.empty((4, 8), device="meta")
        data_wait = _mark_region(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                (data,),
            ),
            chunk_id=chunk_id,
        )
        data_wait.meta["val"] = torch.empty((4, 8), device="meta")
        compute = _mark_region(
            graph.call_function(
                torch.ops.aten.matmul.default,
                (data_wait, weight),
            ),
            chunk_id=chunk_id,
        )
        compute.meta["val"] = torch.empty((4, 8), device="meta")
        other_collective = _mark_region(
            graph.call_function(
                torch.ops._c10d_functional.all_reduce.default,
                (compute, "sum", "ep"),
            ),
            chunk_id=chunk_id,
        )
        other_collective.meta["val"] = torch.empty((4, 8), device="meta")
        other_wait = _mark_region(
            graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default,
                (other_collective,),
            ),
            chunk_id=chunk_id,
        )
        other_wait.meta["val"] = torch.empty((4, 8), device="meta")
        refs[chunk_id].update(
            {
                "count_wait": count_wait,
                "d2hs": tuple(d2hs),
                "setup": setup,
                "data": data,
                "data_wait": data_wait,
                "compute": compute,
                "other_collective": other_collective,
                "other_wait": other_wait,
            }
        )
        outputs.append(other_wait)

    result = graph.call_function(torch.ops.aten.add.Tensor, tuple(outputs))
    if filler is not None:
        result = graph.call_function(torch.ops.aten.add.Tensor, (result, filler))
    graph.output(result)
    refs["filler"] = filler
    return torch.fx.GraphModule({}, graph), refs


def test_count_protocol_advances_to_dispatch_before_ready_filler():
    gm, refs = _build_count_and_data_a2a_graph(ready_filler=True)
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda _node: 8.0,
        # Deliberately exaggerate count latency, matching the noisy standalone
        # profiler estimate seen on A3 hardware.
        collective_cost_fn=lambda _node: 10.0,
        align_across_ranks=False,
    ).run()

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    assert order[refs[0]["count"]] < order[refs[1]["count"]]
    assert order[refs[1]["count"]] < order[refs[0]["data"]]
    assert order[refs[0]["data"]] < order[refs["filler"]]


def test_count_d2h_protocol_releases_ancestor_map_without_cyclic_gc():
    gm, _ = _build_count_and_data_a2a_graph()
    nodes = tuple(node for node in gm.graph.nodes if "chunk_id" in node.meta)

    class AncestorMap(dict):
        """Allow a weak reference to observe map lifetime."""

    ancestors = AncestorMap()
    for node in gm.graph.nodes:
        dependencies = set()
        for dependency in node.all_input_nodes:
            dependencies.update(ancestors[dependency])
            dependencies.add(dependency)
        ancestors[node] = dependencies
    map_ref = weakref.ref(ancestors)
    scheduler = SimpleNamespace(original_order={node: index for index, node in enumerate(gm.graph.nodes)})
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        NpuMoeAutoOverlapScheduler._apply_token_count_d2h_protocol(scheduler, nodes, (), ancestors)
        del ancestors
        # No gc.collect(): training disables GC, and each region's map must
        # be released immediately instead of surviving until periodic GC.
        assert map_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


def test_greedy_scheduler_preserves_collective_order_before_wait():
    gm, refs = _build_two_chunk_graph()
    measured_compute = []
    measured_collective = []

    scheduler = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda node: measured_compute.append(node) or 8.0,
        collective_cost_fn=lambda node: measured_collective.append(node) or 10.0,
        align_across_ranks=False,
    )
    scheduler.run()

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    assert order[refs["launch0"]] < order[refs["launch1"]]
    assert order[refs["launch1"]] < order[refs["wait0"]]
    assert order[refs["wait0"]] < order[refs["compute0"]]
    assert order[refs["compute0"]] < order[refs["wait1"]]
    assert set(measured_compute) == {
        refs["setup0"],
        refs["compute0"],
        refs["setup1"],
        refs["compute1"],
    }
    assert measured_collective == [refs["launch0"], refs["launch1"]]


def test_scheduler_instance_is_one_shot_but_fresh_instance_can_reschedule():
    gm, _ = _build_two_chunk_graph()
    first = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda _node: 1.0,
        collective_cost_fn=lambda _node: 2.0,
        align_across_ranks=False,
    )
    first.run()
    with pytest.raises(RuntimeError, match="one-shot"):
        first.run()

    # Profile-guided rounds create a fresh state machine from the previous
    # round's graph rather than attempting to reset consumed ready sets.
    second = NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda _node: 1.5,
        collective_cost_fn=lambda _node: 2.5,
        align_across_ranks=False,
    )
    assert second.run() is gm


def test_scheduler_respects_dependencies_that_leave_and_reenter_region():
    gm, refs = _build_two_chunk_graph(external_bridge=True)
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda _node: 8.0,
        collective_cost_fn=lambda _node: 10.0,
        align_across_ranks=False,
    ).run()

    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    assert order[refs["compute0"]] < order[refs["launch1"]]


def test_count_a2a_d2h_and_light_compute_are_benchmarked():
    gm, refs = _build_count_and_data_a2a_graph()
    measured_compute = []
    measured_collective = []
    NpuMoeAutoOverlapScheduler(
        gm,
        module_pattern="layers.*.moe",
        compute_cost_fn=lambda node: measured_compute.append(node) or 4.0,
        collective_cost_fn=lambda node: measured_collective.append(node) or 3.0,
        align_across_ranks=False,
    ).run()

    assert [_exchange_label(node) for node in measured_collective] == [
        "count",
        "count",
        "dispatch",
        "dispatch",
    ]
    assert _exchange_label(refs[0]["other_collective"]) is None
    assert set(measured_compute) == {
        *refs[0]["d2hs"],
        refs[0]["setup"],
        refs[0]["compute"],
        *refs[1]["d2hs"],
        refs[1]["setup"],
        refs[1]["compute"],
    }
    assert _is_token_count_sync_copy(refs[0]["d2hs"][0])
    assert _is_zero_cost_schedulable_node(refs[0]["d2hs"][0])
    assert _is_zero_cost_schedulable_node(refs[0]["setup"])
    order = {node: index for index, node in enumerate(gm.graph.nodes)}
    # The four D2H copies retain the native overlap pass's one final blocking
    # synchronization point, while dispatch setup dependencies stay separated.
    assert [copy.kwargs["non_blocking"] for chunk_id in (0, 1) for copy in refs[chunk_id]["d2hs"]] == [
        True,
        True,
        True,
        False,
    ]
    copies = [copy for chunk_id in (0, 1) for copy in refs[chunk_id]["d2hs"]]
    assert all(order[left] < order[right] for left, right in pairwise(copies))
    # Chunk 1's count-wait dependency is not pulled in front of chunk 0's
    # copies.  The cross-chunk relation is only copy issue order.
    assert order[refs[0]["d2hs"][-1]] < order[refs[1]["count_wait"]]
    assert order[copies[-1]] < order[refs[0]["data"]]
    assert order[copies[-1]] < order[refs[1]["data"]]
    for chunk_id in (0, 1):
        assert order[refs[chunk_id]["count"]] < order[refs[chunk_id]["d2hs"][0]]
        assert order[refs[chunk_id]["d2hs"][-1]] < order[refs[chunk_id]["data"]]
