# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for CV-chunk scheduling and emitted dependencies."""

import json
import operator
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch_npu
from torch import fx
from torchtitan.experiments.graph_trainer import passes as graph_passes
from torchtitan.experiments.graph_trainer.registry import PASS_PIPELINE_REGISTRY, POST_INIT_HOOKS

from torchtitan_npu.extensions.experiment.cv_parallel import (
    batch_chunk,
    cv_parallel,
    schedule_calibration,
    schedule_search,
    schedule_simulation,
)
from torchtitan_npu.extensions.experiment.cv_parallel.cv_parallel import (
    dependency_only_schedule,
    event_frontiers,
    materialize,
)
from torchtitan_npu.extensions.experiment.cv_parallel.schedule_calibration import build_profile_schedule
from torchtitan_npu.extensions.experiment.cv_parallel.whole_graph_runtime_profile import (
    RankLocalProfileCosts,
    assign_stable_node_tags,
    node_id,
)


@pytest.mark.parametrize("mismatch", [False, True])
def test_calibration_checks_rank_signatures_after_local_search(two_chains, monkeypatch, mismatch):
    gm, _, _ = two_chains
    events = []
    plan = dependency_only_schedule(gm)
    prediction = {"free": {}, "scheduled": {}}

    def local_search(*args, **kwargs):
        events.append("search")
        return plan, {
            "changed_from_free": True,
            "predicted_coverage_improved": False,
            "prediction": prediction,
        }

    def gather(value):
        events.append("gather")
        other = {**value, "changed": False}
        if mismatch:
            other["signature"] = "different-collective-order"
        return [value, other]

    monkeypatch.setattr(schedule_calibration, "build_local_profile_schedule", local_search)
    monkeypatch.setattr(schedule_calibration, "gather_rank_values", gather)
    if mismatch:
        with pytest.raises(RuntimeError, match="Collective order differs"):
            schedule_calibration.build_profile_schedule(gm, None)
    else:
        result, stats = schedule_calibration.build_profile_schedule(gm, None)
        assert result is plan
        assert stats["changed_from_free_by_rank"] == [True, False]
        assert stats["prediction_by_rank"] == [prediction, prediction]
        assert stats["predicted_cv_improved_by_rank"] == [False, False]
    assert events == ["search", "gather"]


@pytest.fixture
def two_chains():
    g = fx.Graph()
    x = g.placeholder("x")
    w = g.placeholder("w")
    c0 = g.call_function(torch.ops.aten.mm.default, (x, w))
    c1 = g.call_function(torch.ops.aten.mm.default, (x, w))
    v0 = g.call_function(torch.ops.aten.sin.default, (c0,))
    v1 = g.call_function(torch.ops.aten.cos.default, (c1,))
    g.output((v0, v1))
    for n, chunk in ((c0, 0), (v0, 0), (c1, 1), (v1, 1)):
        n.meta.update(chunk_id=chunk, val=torch.empty(2, 2))
    gm = fx.GraphModule({}, g)
    assign_stable_node_tags(gm)
    costs = {node_id(n): 10.0 for n in (c0, c1, v0, v1)}
    resources = {node_id(n): r for n, r in ((c0, "aic"), (c1, "aic"), (v0, "aiv"), (v1, "aiv"))}
    profile = RankLocalProfileCosts(local_ms=costs, local_resources=resources)
    return gm, profile, (c0, c1, v0, v1)


def test_deepep_stream_barrier_preserves_chunk_lane_and_emits_cross_stream_waits():
    def fake_deepep_dispatch(value):
        return value

    fake_deepep_dispatch._schema = SimpleNamespace(name="deepep::dispatch", returns=())
    graph = fx.Graph()
    x = graph.placeholder("x")
    main_prefix = graph.call_function(torch.ops.aten.sin.default, (x,))
    auxiliary_prefix = graph.call_function(torch.ops.aten.cos.default, (x,))
    dispatch = graph.call_function(fake_deepep_dispatch, (auxiliary_prefix,))
    main_suffix = graph.call_function(torch.ops.aten.neg.default, (main_prefix,))
    auxiliary_suffix = graph.call_function(torch.ops.aten.neg.default, (auxiliary_prefix,))
    graph.output((main_suffix, auxiliary_suffix, dispatch))
    for node, lane in (
        (main_prefix, 0),
        (auxiliary_prefix, 1),
        (dispatch, 1),
        (main_suffix, 0),
        (auxiliary_suffix, 1),
    ):
        node.meta.update(chunk_id=lane, val=torch.empty(2, 2))
    gm = fx.GraphModule({}, graph)

    schedule = dependency_only_schedule(gm)
    waits = event_frontiers(schedule)

    assert schedule.lanes[dispatch] == 1
    assert schedule.dependencies[dispatch] >= {main_prefix, auxiliary_prefix}
    assert dispatch in schedule.dependencies[main_suffix]
    assert dispatch in schedule.dependencies[auxiliary_suffix]
    assert waits[dispatch] is main_prefix
    assert waits[main_suffix] is dispatch
    assert auxiliary_suffix not in waits


@pytest.mark.parametrize("join", ["consumer", "gate", "cross_back", "device_cross_back", "tensor_index"])
def test_metadata_waits_track_device_producers_and_preserve_outputs_and_gradients(join):
    graph = fx.Graph()
    x = graph.placeholder("x")
    reduced = graph.call_function(torch.ops.aten.max.dim, (x, 1))
    value = graph.call_function(operator.getitem, (reduced, 0))
    expanded = graph.call_function(torch.ops.aten.unsqueeze.default, (value, 1))
    view = graph.call_function(torch.ops.aten.permute.default, (expanded, [1, 0]))
    device_value = graph.call_function(torch.ops.aten.sin.default, (view,)) if join == "device_cross_back" else None
    if device_value is not None:
        device_value.meta["chunk_id"] = 1
    cross_back = (
        graph.call_function(torch.ops.aten.neg.default, (device_value if device_value is not None else view,))
        if join in {"cross_back", "device_cross_back"}
        else None
    )
    other = graph.call_function(torch.ops.aten.cos.default, (x,))
    if join == "tensor_index":
        other.meta["val"] = torch.empty(2, 2)
        indexed = graph.call_function(operator.getitem, (other, [1, 0]))
        indexed.meta["chunk_id"] = 1
        other_view = graph.call_function(torch.ops.aten.t.default, (indexed,))
    else:
        other_view = graph.call_function(torch.ops.aten.t.default, (other,))
    result = graph.call_function(torch.ops.aten.add.Tensor, (view, other_view))
    graph.output((result, cross_back) if cross_back is not None else (result,))
    for node in (value, expanded, view, other_view, result):
        node.meta["chunk_id"] = 1
    reduced.meta["val"] = (torch.empty(2), torch.empty(2, dtype=torch.int64))
    gm = fx.GraphModule({}, graph)
    schedule = dependency_only_schedule(gm)
    if join == "gate":
        # A scheduled gate on a metadata node must survive at the consumer.
        assert schedule.order.index(other) < schedule.order.index(view)
        schedule.dependencies[view].add(other)

    candidate, waits = materialize(gm, schedule)

    consumer = indexed if join == "tensor_index" else result
    assert waits[consumer] is other
    assert value not in waits and expanded not in waits and other_view not in waits
    if device_value is not None:
        assert waits[device_value] is reduced
        assert waits[cross_back] is device_value
    else:
        assert waits == {consumer: other}
    records = [n for n in candidate.graph.nodes if n.op == "call_method" and n.target == "record"]
    assert {n.args[3][0].name for n in records} == {n.name for n in waits.values()}
    assert len(records) == len(set(waits.values()))  # No output-only event; finish() joins the streams.
    entries = [n for n in candidate.graph.nodes if n.op == "call_method" and n.target == "enter" and n.args[2] >= 0]
    assert len(entries) == len(waits)

    inputs = torch.tensor([[0.2, 0.4], [0.8, 0.1]], dtype=torch.float64, requires_grad=True)
    reference = inputs.detach().clone().requires_grad_()
    expected_view = torch.stack((reference[0, 1], reference[1, 0])).unsqueeze(0)
    other_values = reference.cos()
    if join == "tensor_index":
        other_values = torch.stack((other_values[1], other_values[0]))
    expected = (expected_view + other_values.T,)
    if cross_back is not None:
        expected += (-expected_view.sin() if device_value is not None else -expected_view,)
    expected_grad = torch.autograd.grad(sum(v.sum() for v in expected), reference)
    for compiled in (gm, candidate):
        actual = compiled(inputs)
        gradients = torch.autograd.grad(sum(v.sum() for v in actual), inputs)
        for actual_value, expected_value in zip(actual + gradients, expected + expected_grad, strict=True):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def test_dual_chunk_entry_preserves_outputs_and_gradients(two_chains, monkeypatch, tmp_path):
    gm, profile, _ = two_chains
    x = torch.tensor([[0.2, 0.4], [0.6, 0.8]], requires_grad=True)
    w = torch.tensor([[0.1, 0.3], [0.5, 0.7]], requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_w = w.detach().clone().requires_grad_()
    product = reference_x @ reference_w
    expected = (product.sin(), product.cos())
    expected_grads = torch.autograd.grad(sum(value.sum() for value in expected), (reference_x, reference_w))

    context = {"module": gm, "traced_result": None, "args": (x, w), "train_context": nullcontext}
    chunk_contexts, feedback_calls = [], []

    def configure_chunks(passes, config, *, runtime_context):
        # The fixture is already chunked; check the lowering adapter's input.
        chunk_contexts.append(runtime_context)
        return passes

    def cpu_profile_feedback(graph, run_candidate, *, profile_root):
        # Isolate NPU capture; retain the real runner, planner and event emitter.
        feedback_calls.append(graph)
        assert graph is gm
        assert profile_root == tmp_path
        run_candidate.prepare()
        try:
            for actual, original in zip(run_candidate.calibration_args, (x, w), strict=True):
                torch.testing.assert_close(actual, original)
                assert actual.data_ptr() != original.data_ptr()
        finally:
            run_candidate.finalize()
        return materialize(graph, build_profile_schedule(graph, profile)[0])[0]

    monkeypatch.setenv("CV_PARALLEL_PROFILING_DIR", str(tmp_path))
    monkeypatch.setattr(cv_parallel, "construct_default_graph_passes", lambda *args, **kwargs: [])
    monkeypatch.setattr(cv_parallel, "configure_batch_chunk_passes", configure_chunks)
    monkeypatch.setattr(schedule_calibration, "calibrate_and_select_schedule", cpu_profile_feedback)
    config = SimpleNamespace(
        compile=SimpleNamespace(backend="aot_eager", ep_overlap=SimpleNamespace(enabled=True)),
        dump_folder=str(tmp_path),
    )
    token = batch_chunk._RUNTIME_CONTEXT.set(lambda: context)
    try:
        passes = PASS_PIPELINE_REGISTRY["cv_parallel"](None, config)
    finally:
        batch_chunk._RUNTIME_CONTEXT.reset(token)
    candidate = graph_passes.apply_graph_passes(gm, (x, w), passes)
    actual = candidate(x, w)
    actual_grads = torch.autograd.grad(sum(value.sum() for value in actual), (x, w))

    assert candidate is not gm
    assert len(chunk_contexts) == 1
    assert all(value is context for value in chunk_contexts)
    assert len(feedback_calls) == 1
    for result, reference in zip(actual + actual_grads, expected + expected_grads, strict=True):
        torch.testing.assert_close(result, reference)


@pytest.mark.parametrize(
    "outcome",
    [
        "prediction",
        "measurement_cv",
        "measurement_directional",
        "measurement_same_type",
        "measurement_fb",
        "accepted",
    ],
)
def test_scheduled_feedback_selects_expected_usable_graph(two_chains, outcome, monkeypatch, tmp_path):
    gm, profile, _ = two_chains
    scheduled = replace(dependency_only_schedule(gm), algorithm="profile_guided_cv")
    predicted_gain = outcome != "prediction"
    planning_stats = {
        "prediction_by_rank": [
            {
                "free": {
                    "cv_ms": 1.0,
                    "compute_union_ms": 10.0,
                    "cube_covered_by_vector_percent": 10.0,
                    "vector_covered_by_cube_percent": 10.0,
                    "cc_ms": 2.0,
                    "vv_ms": 2.0,
                    "span_ms": 40.0,
                },
                "scheduled": {
                    "cv_ms": 2.0 if predicted_gain else 1.0,
                    "compute_union_ms": 10.0,
                    "cube_covered_by_vector_percent": 20.0 if predicted_gain else 10.0,
                    "vector_covered_by_cube_percent": 20.0 if predicted_gain else 10.0,
                    "cc_ms": 1.0 if predicted_gain else 2.0,
                    "vv_ms": 1.0 if predicted_gain else 2.0,
                    "span_ms": 40.0,
                },
            }
        ],
        "window_refinement": {},
        "predicted_free_cv_ms": 1.0,
        "predicted_cv_ms": 2.0 if predicted_gain else 1.0,
        "predicted_free_cc_ms": 2.0,
        "predicted_cc_ms": 1.0 if predicted_gain else 2.0,
        "predicted_free_vv_ms": 0.0,
        "predicted_vv_ms": 0.0,
        "predicted_free_cube_covered_by_vector_percent": 10.0,
        "predicted_cube_covered_by_vector_percent": 20.0 if predicted_gain else 10.0,
        "predicted_free_vector_covered_by_cube_percent": 10.0,
        "predicted_vector_covered_by_cube_percent": 20.0 if predicted_gain else 10.0,
        "predicted_free_span_ms": 40.0,
        "predicted_span_ms": 40.0,
        "planning_wall_seconds": 0.01,
        "search_wall_seconds": 0.01,
        "num_waits_free": 1,
        "num_waits": 1,
        "num_waits_added": 0,
        "num_waits_removed": 0,
        "num_waits_retargeted": 0,
        "changed_from_free_by_rank": [predicted_gain],
        "predicted_coverage_improved_by_rank": [predicted_gain],
        "predicted_cv_improved_by_rank": [predicted_gain],
    }
    overlap = {
        "cc_max_ms": 0.0,
        "vv_max_ms": 0.0,
        "cv_mean_ms": 1.0,
        "cv_percent_of_compute": 10.0,
        "cube_mix_aiv_mean_ms": 0.0,
        "vector_mix_aic_mean_ms": 0.0,
        "cv_with_cube_mix_aiv_mean_ms": 1.0,
        "cv_with_cube_mix_aiv_percent_of_compute": 10.0,
    }
    profile_calls = []

    def fake_profile(*args, kernel_summary, **kwargs):
        profile_calls.append(kwargs["profile_name"])
        kernel_summary["time_ms"] = {"compute_union": 10.0, "cv_overlap": 1.0}
        return profile

    measurements = iter(
        [
            {"samples_ms": [10.0], "median_ms": 10.0},
            {
                "samples_ms": [11.0 if outcome == "measurement_fb" else 9.0],
                "median_ms": 11.0 if outcome == "measurement_fb" else 9.0,
            },
        ]
    )
    monkeypatch.setattr(torch.npu, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: SimpleNamespace(get_device_name=lambda value: "test-npu"),
    )
    monkeypatch.setattr(schedule_calibration, "_measure_schedule", lambda *args, **kwargs: next(measurements))
    monkeypatch.setattr(schedule_calibration, "profile_whole_graph_costs", fake_profile)
    monkeypatch.setattr(schedule_calibration, "_kernel_overlap", lambda summaries: overlap.copy())
    monkeypatch.setattr(
        schedule_calibration,
        "build_profile_schedule",
        lambda *args, **kwargs: (scheduled, planning_stats),
    )
    cv_passed = outcome not in {"prediction", "measurement_cv"}
    directional_passed = outcome != "measurement_directional"
    same_type_passed = outcome != "measurement_same_type"
    monkeypatch.setattr(
        schedule_calibration,
        "_validate_cv_overlap",
        lambda *args: [
            {
                "free_ms": 1.0,
                "scheduled_ms": 2.0 if cv_passed else 1.0,
                "delta_ms": 1.0 if cv_passed else 0.0,
                "free_percent_of_compute": 10.0,
                "scheduled_percent_of_compute": 20.0 if cv_passed else 10.0,
                "share_delta_percentage_points": 10.0 if cv_passed else 0.0,
                "time_passed": cv_passed,
                "share_passed": cv_passed,
                "cube_coverage_passed": directional_passed,
                "vector_coverage_passed": directional_passed,
                "cc_passed": same_type_passed,
                "vv_passed": same_type_passed,
                "cube_covered_by_vector_delta_percentage_points": 10.0 if directional_passed else 0.0,
                "vector_covered_by_cube_delta_percentage_points": 10.0 if directional_passed else 0.0,
                "passed": cv_passed and directional_passed and same_type_passed,
            }
        ],
    )

    selected_graph = schedule_calibration.calibrate_and_select_schedule(gm, object(), profile_root=tmp_path)

    report_text = (tmp_path / "schedule_feedback.json").read_text()
    assert report_text.endswith("\n")
    report = json.loads(report_text)
    predicted_fallback = outcome == "prediction"
    assert report["status"] == ("fallback" if predicted_fallback else "applied")
    assert report["selected"] == ("dependency_only" if predicted_fallback else "profile_guided_cv")
    assert report["fallback_reason"] == ("no_effective_predicted_cv_gain" if predicted_fallback else None)
    expected_warning = {
        "prediction": None,
        "measurement_cv": "measured_cv_not_improved",
        "measurement_directional": "measured_directional_coverage_not_improved",
        "measurement_same_type": "measured_same_type_overlap_increased",
        "measurement_fb": "forward_backward_regression",
        "accepted": None,
    }
    assert report.get("validation_warning") == expected_warning[outcome]
    if not predicted_fallback:
        assert report["validation_passed"] == (outcome == "accepted")
    expected_profile_calls = (
        ["dependency_only"] if outcome == "prediction" else ["dependency_only", "profile_guided_cv_validation"]
    )
    assert profile_calls == expected_profile_calls
    if outcome != "prediction":
        assert report["total_graph_replays"] == 12
    x = torch.tensor([[0.2, 0.4], [0.6, 0.8]])
    w = torch.tensor([[0.1, 0.3], [0.5, 0.7]])
    actual = selected_graph(x, w)
    expected_product = x @ w
    torch.testing.assert_close(actual[0], expected_product.sin())
    torch.testing.assert_close(actual[1], expected_product.cos())


@pytest.mark.parametrize(
    "regression,expected_field",
    [
        ("none", "passed"),
        ("cube_coverage", "cube_coverage_passed"),
        ("vector_coverage", "vector_coverage_passed"),
        ("cc", "cc_passed"),
        ("vv", "vv_passed"),
    ],
)
def test_cv_validation_requires_directional_coverage_without_same_type_growth(regression, expected_field):
    free = {
        "time_ms": {
            "cv_overlap": 2.0,
            "compute_union": 10.0,
            "cube_busy": 8.0,
            "vector_busy": 8.0,
            "cc_overlap": 2.0,
            "vv_overlap": 2.0,
        }
    }
    scheduled = {
        "time_ms": {
            "cv_overlap": 3.0,
            "compute_union": 10.0,
            "cube_busy": 8.0,
            "vector_busy": 8.0,
            "cc_overlap": 1.0,
            "vv_overlap": 1.0,
        }
    }
    if regression == "cube_coverage":
        scheduled["time_ms"]["cube_busy"] = 13.0
    elif regression == "vector_coverage":
        scheduled["time_ms"]["vector_busy"] = 13.0
    elif regression == "cc":
        scheduled["time_ms"]["cc_overlap"] = 3.0
    elif regression == "vv":
        scheduled["time_ms"]["vv_overlap"] = 3.0

    result = schedule_calibration._validate_cv_overlap([free], [scheduled])[0]

    assert result[expected_field] == (regression == "none")
    assert result["passed"] == (regression == "none")


def test_cv_candidate_generation_enumerates_all_pairs_before_budgeting():
    planner = schedule_search._WindowPlanner.__new__(schedule_search._WindowPlanner)
    planner.simulation = schedule_simulation.ScheduleSimulation.__new__(schedule_simulation.ScheduleSimulation)
    partners = [(2.0 * i, 2.0 * i + 1.0, "aic", i + 1, 0) for i in range(300)]
    planner.simulation.absolute = [[(1000.0, 1001.0, "aiv", 0, 0)], partners]
    planner.simulation.positions = {0: 1, **{i + 1: i + 1 for i in range(300)}}
    planner.simulation.clocks = [[1, 0], *[[0, i + 1] for i in range(300)]]
    planner.simulation.timelines = [
        schedule_simulation._timelines(planner.simulation.absolute[0]),
        schedule_simulation._timelines(planner.simulation.absolute[1]),
    ]
    planner.simulation.stats = Counter()

    candidates = planner.candidates()

    assert len(candidates) == 300
    assert planner.simulation.stats["num_cv_candidates"] == 300


def test_reorder_change_bounds_preserve_window_and_full_dag_scores(rms_reorder_window):
    gm, profile, (_, moved_node, _, _) = rms_reorder_window
    planner = schedule_search._WindowPlanner(gm, profile, {})
    moved = planner.simulation.nodes.index(moved_node)
    lane, first, size = next(
        window
        for window in planner.simulation.windows
        if moved in planner.simulation.lanes[window[0]][window[1] : window[1] + window[2]]
    )
    sequence = planner.simulation.lanes[lane][first : first + size]
    records = planner._reorder_proposal_records(sequence, moved)
    members = frozenset(sequence)
    upper_rank = max(planner.simulation.rank[i] for i in sequence)

    for candidate in records:
        proposal = candidate.sequence
        if candidate.left > candidate.right:
            continue
        assert planner.simulation._window_score(proposal) == planner.simulation._window_score(
            proposal,
            lane=lane,
            first=first,
            changed_bounds=(candidate.left, candidate.right),
        )
        planner.simulation.reorder_trial_cache.clear()
        expected = planner.simulation._reorder_trial(lane, first, proposal)
        planner.simulation.reorder_trial_cache.clear()
        actual = planner.simulation._reorder_trial(
            lane,
            first,
            proposal,
            changed_bounds=(candidate.left, candidate.right),
            window_members=members,
            window_upper_rank=upper_rank,
            cache_result=False,
        )
        assert actual == expected


def test_unsafe_view_mutation_keeps_shared_storage_barrier():
    graph = fx.Graph()
    x = graph.placeholder("x")
    cube = graph.call_function(torch.ops.aten.mm.default, (x, x))
    view = graph.call_function(torch.ops.aten._unsafe_view.default, (x, [2, 2]))
    update = graph.call_function(torch.ops.aten.add_.Tensor, (view, view))
    after = graph.call_function(torch.ops.aten.mul.Tensor, (x, x))
    graph.output((cube, update, after))
    for node, lane in ((cube, 1), (view, 0), (update, 0), (after, 1)):
        node.meta.update(chunk_id=lane, val=torch.empty(2, 2))
    gm = fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    costs = {node_id(cube): 10.0, node_id(update): 1.0, node_id(after): 1.0}
    resources = {node_id(cube): "aic", node_id(update): "aiv", node_id(after): "aiv"}
    profile = RankLocalProfileCosts(local_ms=costs, local_resources=resources)

    schedule = build_profile_schedule(gm, profile)[0]
    candidate, _ = materialize(gm, schedule)
    x = torch.tensor([[0.2, 0.3], [0.5, 0.7]])
    expected_cube = x @ x
    expected = x * 2
    actual = candidate(x)

    assert cube in schedule.dependencies[update]
    assert schedule.start_ms[update] >= schedule.end_ms[cube]
    assert update in schedule.dependencies[after]
    assert schedule.start_ms[after] >= schedule.end_ms[update]
    torch.testing.assert_close(actual[0], expected_cube)
    torch.testing.assert_close(actual[1], expected)
    torch.testing.assert_close(actual[2], expected.square())
    torch.testing.assert_close(x, expected)


def test_batch_metadata_remap_changes_direct_and_derived_consumers(monkeypatch):
    graph = fx.Graph()
    x = graph.placeholder("x")
    full = graph.placeholder("full")
    local0 = graph.placeholder("local0")
    local1 = graph.placeholder("local1")
    size = graph.call_function(torch.ops.aten.sym_numel.default, (full,))
    indices = graph.call_function(torch.ops.aten.arange.default, (size,))
    a = graph.call_function(torch.ops.aten.add.Tensor, (x, full))
    b = graph.call_function(torch.ops.aten.sub.Tensor, (x, full))
    y0 = graph.call_function(torch.ops.aten.add.Tensor, (a, indices))
    y1 = graph.call_function(torch.ops.aten.add.Tensor, (b, indices))
    graph.output((y0, y1))
    gm = fx.GraphModule({}, graph)
    for n in (x, local0, local1, a, b, y0, y1):
        n.meta["val"] = torch.empty(2, device="meta")
    for n in (full, indices):
        n.meta["val"] = torch.empty(3, device="meta")
    size.meta["val"] = 3
    for chunk, nodes in enumerate(((a, y0), (b, y1))):
        for n in nodes:
            n.meta["chunk_id"] = chunk
    values = [
        torch.tensor([10.0, 20.0]),
        torch.tensor([50.0, 60.0, 70.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([3.0, 4.0]),
    ]
    for value, chunk in zip(values[1:], (-1, 0, 1), strict=True):
        value._npu_chunk_cv_metadata_pair = ("plan.gather", chunk)
    monkeypatch.setattr(batch_chunk, "_runtime_flat_inputs", lambda _: values)

    candidate = batch_chunk.remap_prebuilt_batch_metadata_pass(gm, runtime_context={})
    actual = candidate(*values)

    torch.testing.assert_close(actual[0], torch.tensor([11.0, 23.0]))
    torch.testing.assert_close(actual[1], torch.tensor([7.0, 17.0]))


def test_prebuilt_metadata_is_not_split_into_two_chunks_again(monkeypatch):
    graph = fx.Graph()
    full = graph.placeholder("full")
    local0 = graph.placeholder("local0")
    local1 = graph.placeholder("local1")
    size = graph.call_function(operator.floordiv, (2, 2))
    size.meta["val"] = 1
    split = graph.call_function(torch.ops.aten.split_with_sizes.default, (full, [size, size], 0))
    split.meta["chunked_region_role"] = "split_boundary"
    first = graph.call_function(operator.getitem, (split, 0))
    second = graph.call_function(operator.getitem, (split, 1))
    first.meta["chunk_id"] = 0
    second.meta["chunk_id"] = 1
    graph.output((first, second))
    gm = fx.GraphModule({}, graph)
    values = [torch.tensor([100, 200]), torch.tensor([10]), torch.tensor([20])]
    for node, value, chunk in zip((full, local0, local1), values, (-1, 0, 1), strict=True):
        node.meta["val"] = value
        value._npu_chunk_cv_metadata_pair = ("plan.residual", chunk)
    monkeypatch.setattr(batch_chunk, "_runtime_flat_inputs", lambda _: values)

    candidate = batch_chunk.remap_prebuilt_batch_metadata_pass(gm, runtime_context={})
    actual = candidate(*values)

    torch.testing.assert_close(actual[0], torch.tensor([10]))
    torch.testing.assert_close(actual[1], torch.tensor([20]))


def test_collective_wait_retains_cross_lane_input_event(two_chains):
    gm, profile, (c0, c1, v0, _) = two_chains
    with gm.graph.inserting_before(v0):
        wait = gm.graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (c1,))
    wait.meta.update(chunk_id=0, val=torch.empty(2, 2))
    v0.args = (wait,)
    assign_stable_node_tags(gm)
    # Retagging adds only the new node; rebuild explicit costs by node name.
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    costs = {node_id(n): 0.0 if n is wait else 10.0 for n in nodes}
    resources = {node_id(n): "aic" if n in (c0, c1) else "aiv" for n in nodes}
    profile = RankLocalProfileCosts(local_ms=costs, local_resources=resources)

    schedule = build_profile_schedule(gm, profile)[0]
    frontiers = event_frontiers(schedule)

    assert c1 in schedule.dependencies[wait]
    assert schedule.start_ms[wait] >= schedule.end_ms[c1]
    # Either order may join at an earlier node on wait's lane. The input
    # completion must still be protected by an emitted direct/transitive event.
    assert any(
        source is c1 and schedule.lanes[target] == 0 and schedule.order.index(target) <= schedule.order.index(wait)
        for target, source in frontiers.items()
    )


@pytest.fixture
def rms_reorder_window():
    graph = fx.Graph()
    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    c0 = graph.call_function(torch.ops.aten.mm.default, (x, x))
    c0b = graph.call_function(torch.ops.aten.mm.default, (x, x))
    norm = graph.call_function(torch.ops.npu.npu_rms_norm.default, (c0, weight, 1e-6))
    value = graph.call_function(operator.getitem, (norm, 0))
    v0 = graph.call_function(torch.ops.aten.sin.default, (value,))
    c1 = graph.call_function(torch.ops.aten.mm.default, (x, x))
    v1 = graph.call_function(torch.ops.aten.cos.default, (c1,))
    graph.output((c0b, v0, v1))
    for node, lane in ((c0, 0), (c0b, 0), (norm, 0), (value, 0), (v0, 0), (c1, 1), (v1, 1)):
        node.meta.update(chunk_id=lane, val=torch.empty(2, 2))
    norm.meta["val"] = (torch.empty(2, 2), torch.empty(2, 1))
    gm = fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    costs = {node_id(n): t for n, t in ((c0, 10), (c0b, 10), (norm, 1), (v0, 19), (c1, 20), (v1, 10))}
    resources = {
        node_id(n): r for n, r in ((c0, "aic"), (c0b, "aic"), (norm, "aiv"), (v0, "aiv"), (c1, "aic"), (v1, "aiv"))
    }
    profile = RankLocalProfileCosts(local_ms=costs, local_resources=resources)
    return gm, profile, (c0b, norm, value, v0)


class _CpuRmsInterpreter(fx.Interpreter):
    """Replace only the NPU kernel with differentiable CPU RMSNorm."""

    def call_function(self, target, args, kwargs):
        if target == torch.ops.npu.npu_rms_norm.default:
            x, weight, eps = args
            return torch.nn.functional.rms_norm(x, weight.shape, weight, eps), (x.square().mean(-1, True) + eps).rsqrt()
        return super().call_function(target, args, kwargs)


def test_small_regions_use_exhaustive_search_without_budget(rms_reorder_window):
    gm, profile, _ = rms_reorder_window

    _, stats = build_profile_schedule(gm, profile)

    refinement = stats["window_refinement"]
    assert stats["search_mode"] == "exhaustive_region_greedy"
    assert stats["search_budget"] is None
    assert refinement["num_search_regions"] >= 1
    assert refinement["num_passes"] >= refinement["num_search_regions"]
    assert refinement.get("num_bounded_search_regions", 0) == 0


def test_large_region_reports_adaptive_search_budget(rms_reorder_window, monkeypatch):
    gm, profile, _ = rms_reorder_window
    monkeypatch.setattr(schedule_search, "_EXHAUSTIVE_REGION_NODE_LIMIT", 1)

    _, stats = build_profile_schedule(gm, profile)

    assert stats["search_mode"] == "adaptive_bounded_region_greedy"
    assert stats["search_budget"] == {
        "exhaustive_region_node_limit": 1,
        "bounded_region_max_passes": 8,
        "reorder_moved_nodes_per_kind": 16,
        "reorder_destinations_per_node": 24,
        "cv_candidates_per_pass": 2048,
        "cv_sources_per_candidate": 32,
    }
    assert stats["window_refinement"]["num_bounded_search_regions"] >= 1


def test_implied_wait_gates_are_pruned_with_one_refresh(two_chains):
    gm, profile, (c0, c1, v0, v1) = two_chains
    planner = schedule_search._WindowPlanner(gm, profile, {})
    indices = {node: planner.simulation.nodes.index(node) for node in (c0, c1, v0, v1)}
    redundant = {
        indices[v0]: indices[c1],
        indices[v1]: indices[c0],
    }
    for target, source in redundant.items():
        planner.simulation.base_parents[target].add(source)
    planner.simulation.gates = redundant.copy()
    planner.simulation.refresh()
    refreshes = planner.simulation.stats["num_full_refreshes"]

    planner.prune_gates()

    assert planner.simulation.gates == {}
    assert planner.simulation.stats["num_gate_implied_prunes"] == 2
    assert planner.simulation.stats["num_full_refreshes"] == refreshes + 1
    assert {entry["reason"] for entry in planner.simulation.pruned_gates} == {"implied_dependency"}


@pytest.mark.parametrize("quantizer", ["li_fp8", "kv_mxfp8"])
def test_quantized_vector_chain_can_reorder_before_independent_cube(rms_reorder_window, quantizer):
    gm, profile, (delayed_cube, quant, value, vector) = rms_reorder_window
    if quantizer == "li_fp8":
        quant.target = torch.ops.npu.npu_dynamic_quant.default
        quant.args = (quant.args[0],)
        quant.kwargs = {"dst_type": torch_npu.float8_e4m3fn, "quant_mode": "pertoken"}
    else:
        # Optional TorchAO-NPU registers the production custom-op schema;
        # this CPU check evaluates the schedule, not the quantization kernel.
        pytest.importorskip("torchao_npu")
        from torchao_npu.quantization.quant_primitives.mx import mx_last_dim_fake_quantize  # noqa: F401

        quant.target = torch.ops.torchao_npu.mx_last_dim_fake_quantize.default
        quant.args = (quant.args[0], torch_npu.float8_e4m3fn, 64, "rint", 0, 0.0)
        quant.meta["val"] = torch.empty(2, 2)
        value.replace_all_uses_with(quant)
        gm.graph.erase_node(value)
    gm.recompile()

    schedule, stats = build_profile_schedule(gm, profile)

    assert schedule.order.index(quant) < schedule.order.index(vector) < schedule.order.index(delayed_cube)
    assert max(schedule.end_ms.values()) == 40
    assert stats["predicted_cv_ms"] == 30
    assert not cv_parallel._is_stream_barrier(quant)
    assert quant.args[0] in schedule.dependencies[quant]
    if quantizer == "li_fp8":
        assert quant in schedule.dependencies[value]
        assert value in schedule.dependencies[vector]
    else:
        assert quant in schedule.dependencies[vector]


def test_optimization_summary_attributes_reorder_and_wait_benefits(rms_reorder_window):
    gm, profile, _ = rms_reorder_window

    _, stats = build_profile_schedule(gm, profile)

    summary = stats["optimization_summary"]
    assert summary["reorder"]["reordered_nodes"] == stats["num_reordered_nodes"]
    assert summary["reorder"]["actions"] == len(stats["reorder_changes"])
    assert summary["wait_event"]["waits_added"] == stats["num_waits_added"]
    assert summary["wait_event"]["waits_removed"] == stats["num_waits_removed"]
    assert summary["wait_event"]["waits_retargeted"] == stats["num_waits_retargeted"]
    for field, total in summary["total_estimated_benefit"].items():
        attributed = summary["reorder"]["estimated_benefit"][field] + summary["wait_event"]["estimated_benefit"][field]
        assert attributed == pytest.approx(total)


def test_mix_aic_resource_fallback_keeps_a_schedulable_phase(two_chains):
    gm, profile, (_, _, vector, _) = two_chains
    profile.local_resources[node_id(vector)] = "mix_aic"

    planner = schedule_search._WindowPlanner(gm, profile, {})

    index = planner.simulation.nodes.index(vector)
    assert planner.simulation.phases[index] == [(0.0, 10.0, "mix_aic")]


@pytest.mark.parametrize(
    "cross_lane_consumer,untagged_cube",
    [(False, False), (True, False), (False, True)],
    ids=["independent", "round_trip_dependency", "untagged_cube"],
)
def test_npu_reorder_exposes_vector_chain_and_preserves_outputs_and_gradients(
    rms_reorder_window, cross_lane_consumer, untagged_cube, monkeypatch
):
    gm, profile, (c0b, norm, value, v0) = rms_reorder_window
    if untagged_cube:
        c0b.meta.pop("chunk_id")
    if cross_lane_consumer:
        consumer = next(n for n in gm.graph.nodes if n.target == torch.ops.aten.cos.default)
        consumer.args = (v0,)
        c0b.args = (consumer, c0b.args[0])
        consumer.append(c0b)

    schedule, stats = build_profile_schedule(gm, profile)
    candidate, _ = materialize(gm, schedule)

    def uncached_coverage(simulation, index, start):
        return schedule_simulation._coverage(
            ((start + a, start + b, kind) for a, b, kind in simulation.phases[index]),
            simulation.timelines[1 - simulation.lane_of[index]],
        )

    with monkeypatch.context() as patch:
        patch.setattr(schedule_simulation.ScheduleSimulation, "_window_coverage", uncached_coverage)
        timelines = schedule_simulation._timelines
        patch.setattr(schedule_simulation, "_timelines", lambda phases, **kwargs: timelines(phases, cache=False))
        patch.setattr(schedule_simulation, "_SHIFTED_PHASE_CACHE_LIMIT", 0)
        patch.setattr(schedule_simulation, "_TRIAL_CACHE_LIMIT", 0)
        patch.setattr(schedule_search, "_REORDER_TEMPLATE_CACHE_NODE_LIMIT", 0)
        uncached, uncached_stats = build_profile_schedule(gm, profile)

    assert schedule == uncached
    assert stats["reorder_changes"] == uncached_stats["reorder_changes"]
    assert stats["predicted_cv_ms"] == uncached_stats["predicted_cv_ms"]

    assert schedule.order.index(v0) < schedule.order.index(c0b)
    assert max(schedule.end_ms.values()) == (50 if cross_lane_consumer else 40)
    assert stats["predicted_cv_ms"] == (20 if cross_lane_consumer else 30)
    assert stats["predicted_cc_ms"] <= stats["predicted_free_cc_ms"]
    assert stats["predicted_vv_ms"] <= stats["predicted_free_vv_ms"]
    assert stats["predicted_cube_covered_by_vector_percent"] > stats["predicted_free_cube_covered_by_vector_percent"]
    assert stats["predicted_vector_covered_by_cube_percent"] > stats["predicted_free_vector_covered_by_cube_percent"]
    assert "num_vv_edits" not in stats["window_refinement"]
    assert norm in schedule.dependencies[value]
    assert value in schedule.dependencies[v0]
    if cross_lane_consumer:
        assert v0 in schedule.dependencies[consumer]
        assert schedule.start_ms[consumer] >= schedule.end_ms[v0]
        assert consumer in schedule.dependencies[c0b]
        assert schedule.start_ms[c0b] >= schedule.end_ms[consumer]
    x = torch.tensor([[0.2, 0.1], [0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    weight = torch.tensor([0.7, 1.3], dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    product = reference_x @ reference_x
    normalized = product / (product.square().mean(-1, True) + 1e-6).sqrt() * reference_weight
    last = (normalized.sin() if cross_lane_consumer else product).cos()
    expected = (last @ reference_x if cross_lane_consumer else product, normalized.sin(), last)
    actual = _CpuRmsInterpreter(candidate).run(x, weight)
    actual_grads = torch.autograd.grad(sum(v.sum() for v in actual), (x, weight))
    expected_grads = torch.autograd.grad(sum(v.sum() for v in expected), (reference_x, reference_weight))
    for a, b in zip(actual + actual_grads, expected + expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("constraint", ["unknown_custom", "mutation", "rng", "collective"])
def test_npu_reorder_retains_window_and_effect_boundaries(rms_reorder_window, constraint):
    gm, profile, (c0b, norm, _value, v0) = rms_reorder_window
    with gm.graph.inserting_before(norm):
        if constraint == "unknown_custom":
            boundary = gm.graph.call_function(torch.ops.npu.npu_silu.default, (c0b,))
        elif constraint == "mutation":
            boundary = gm.graph.call_function(torch.ops.aten.add_.Tensor, (c0b, c0b))
        elif constraint == "rng":
            boundary = gm.graph.call_function(torch.ops.aten.rand_like.default, (c0b,))
        else:
            boundary = gm.graph.call_function(torch.ops._c10d_functional.all_reduce.default, (c0b, "sum", "0"))
    boundary.meta.update(chunk_id=0, val=torch.empty(2, 2))
    assign_stable_node_tags(gm)

    schedule, _ = build_profile_schedule(gm, profile)

    assert schedule.order.index(c0b) < schedule.order.index(boundary) < schedule.order.index(norm)
    assert schedule.order.index(norm) < schedule.order.index(v0)


@pytest.mark.parametrize(
    "shared_qk,chunk_only_metadata,quantized",
    [(False, False, False), (True, True, False), (True, False, True)],
    ids=["separate_qk", "shared_qk_extra_roles", "fp8_li"],
)
def test_chunk_config_prebuilds_metadata_with_full_qk_contract(
    dsv4, shared_qk, chunk_only_metadata, quantized, monkeypatch
):
    from torchtitan.config.override import apply_overrides
    from torchtitan.models.common.attention import VarlenMetadata

    from torchtitan_npu.models.common.metadata_extension import MetadataExtension
    from torchtitan_npu.models.deepseek_v4.config_registry import (
        graph_trainer_deepseek_v4_debugmodel,
    )
    from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
    from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc

    li_calls = []

    def record_li_metadata(*args, **kwargs):
        li_calls.append((args, kwargs))
        # Encode the opaque kernel kind and local token extent, not LI math.
        mode = args[4] if len(args) == 5 else 0
        return torch.tensor([mode, kwargs["cu_seqlens_q"][-1].item()], dtype=torch.int32)

    monkeypatch.setattr(ascendc, "lightning_indexer_metadata", record_li_metadata)
    if chunk_only_metadata:
        # The vendor may omit a field for the full layout but return it for
        # each local layout. Only full-layout roles belong in the FX inputs.
        def optional_smla_metadata(*args, cu_seqlens_q, **kwargs):
            return None if cu_seqlens_q.numel() == 3 else torch.tensor([7], dtype=torch.int32)

        monkeypatch.setattr(ascendc, "sparse_flash_mla_metadata", optional_smla_metadata)

    config = graph_trainer_deepseek_v4_debugmodel()
    config.override.imports.extend(
        (
            "torchtitan_npu.override.deepseek_v4.sparse_attn.asc_li_metadata",
            "torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_metadata.asc_metadata",
        )
    )
    config.model_spec.model.metadata_extension = MetadataExtension.Config(
        window_size=128,
        num_heads=16,
        head_dim=512,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=512,
    )
    cu_seq = torch.tensor([0, 128, 256], dtype=torch.int32)
    varlen = VarlenMetadata(
        cu_seq_q=cu_seq,
        cu_seq_k=cu_seq if shared_qk else cu_seq.clone(),
        max_q=128,
        max_k=128,
    )

    if quantized:
        pytest.importorskip("torchao_npu")
        from torchtitan_npu.extensions.adaptor.torchao import converter as torchao_converter
        from torchtitan_npu.config.configs import QuantizationExtensionConfig

        monkeypatch.setattr(torchao_converter, "_npu_quantized_module_cache", {})
        monkeypatch.setattr(dsv4.cann_ops, "quant_lightning_indexer_metadata", record_li_metadata, raising=False)
        config.model_spec = torchao_converter.apply_quantization_converter(
            config.model_spec,
            QuantizationExtensionConfig(enable_quantized_training=True, recipe="all_block_fp8", li_quantization="fp8"),
            model_compile_enabled=False,
        )
    apply_overrides(config.override, config)
    # Bypass weight construction and distributed Trainer initialization only;
    # retain config selection, the post-init hook and the model metadata path.
    model = SimpleNamespace(
        _lightning_indexer_metadata=config.model_spec.model.lightning_indexer_metadata.build(),
        _metadata_extension=config.model_spec.model.metadata_extension.build(),
        compress_ratios=(1, 4, 128),
        get_attention_masks=lambda **kwargs: varlen,
    )
    trainer = SimpleNamespace(
        model_parts=[model],
        _make_fx_forward_backward_step=lambda *args: None,
        _prepare_trace_inputs=lambda *args: None,
    )
    POST_INIT_HOOKS["cv_parallel"](trainer)
    _, _, kwargs = DeepSeekV4Model.build_attention_masks(model, None, None, {})
    metadata = kwargs["attention_masks"]

    assert metadata.varlen.cu_seq_q.tolist() == [0, 128, 256]
    assert (metadata.varlen.cu_seq_k is metadata.varlen.cu_seq_q) == shared_qk
    assert metadata.batch_chunk_tensors is not None
    assert metadata.varlen.cu_seq_q._npu_chunk_cv_metadata_pair == ("varlen.cu_seq_q", -1)
    assert len(li_calls) == 3
    for (_, call), q_bounds, k_bounds in zip(
        li_calls, ([0, 128, 256], [0, 128], [0, 128]), ([0, 32, 64], [0, 32], [0, 32]), strict=True
    ):
        assert call["cu_seqlens_q"].tolist() == q_bounds
        assert call["cu_seqlens_k"].tolist() == k_bounds
        assert (call["layout_q"], call["layout_k"], call["mask_mode"], call["cmp_ratio"]) == ("TND", "TND", 3, 4)
    assert metadata.plans[4].li_metadata is metadata.asc_plans[4].li_metadata
    assert metadata.plans[4].li_metadata.tolist() == [int(quantized), 256]
    for chunk_id, tensors in enumerate(metadata.batch_chunk_tensors):
        fields = {value._npu_chunk_cv_metadata_pair: value for value in tensors}
        assert fields["varlen.cu_seq_q", chunk_id].tolist() == [0, 128]
        assert (("varlen.cu_seq_k", chunk_id) in fields) != shared_qk
        if not shared_qk:
            assert fields["varlen.cu_seq_k", chunk_id].tolist() == [0, 128]
        for ratio in (1, 4, 128):
            assert ((f"asc_plans.{ratio}.smla_metadata", chunk_id) in fields) != chunk_only_metadata
            assert (f"asc_plans.{ratio}.smla_grad_metadata", chunk_id) in fields
        assert ("asc_plans.4.li_metadata", chunk_id) in fields
        assert fields["asc_plans.4.li_metadata", chunk_id].tolist() == [int(quantized), 128]

    if quantized:
        consumer = SimpleNamespace(_torchao_npu_module_swap_config=SimpleNamespace(cmp_ratio=4), index_topk=512)
        prepared = torchao_converter._prepare_quant_lightning_indexer_inputs(
            consumer, torch.empty(2, 128, 8, 128), torch.empty(2, 32, 128), torch.empty(2, 128, 8), metadata
        )
        assert prepared.metadata is metadata.plans[4].li_metadata
        assert prepared.metadata._npu_chunk_cv_metadata_pair == ("asc_plans.4.li_metadata", -1)
        assert prepared.cu_seqlens_q.tolist() == [0, 128, 256]
        assert prepared.cu_seqlens_k.tolist() == [0, 32, 64]
