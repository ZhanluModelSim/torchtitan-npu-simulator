# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU checks for CV-chunk CANN mapping and profile summaries."""

import csv
import json
from types import SimpleNamespace

import pytest
import torch
from torch import fx

from torchtitan_npu.extensions.experiment.cv_parallel.cann_kernel_metrics import CannKernelMetrics
from torchtitan_npu.extensions.experiment.cv_parallel.cann_launches import associate_cann_launches
from torchtitan_npu.extensions.experiment.cv_parallel.profile_overlap import SCOPE, summarize_profile
from torchtitan_npu.extensions.experiment.cv_parallel.whole_graph_runtime_profile import (
    _collective_count_manifest,
    _extract_cann_node_measurements,
    _match_fx_nodes_to_host_scopes,
    assign_stable_node_tags,
    collective_order_keys,
    node_id,
)
from torchtitan_npu.ops.ascendc.moe_token_permute import npu_moe_token_permute  # noqa: F401


def test_cann_connection_recovers_launch_without_torch_to_npu_flow():
    scope = {"ph": "X", "cat": "cpu_op", "pid": 1, "tid": 10, "ts": 0, "dur": 20}
    device_task = {
        "ph": "X",
        "name": "custom_kernel",
        "pid": 3,
        "tid": 30,
        "ts": 100,
        "dur": 5,
        "args": {"connection_id": 99},
    }
    events = [
        {"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "CANN"}},
        {"ph": "M", "name": "process_name", "pid": 3, "args": {"name": "Ascend Hardware"}},
        scope,
        {
            "ph": "X",
            "cat": "enqueue",
            "pid": 1,
            "tid": 10,
            "ts": 5,
            "dur": 2,
            "args": {"correlation_id": 7},
        },
        {
            "ph": "X",
            "cat": "dequeue",
            "pid": 1,
            "tid": 20,
            "ts": 30,
            "dur": 20,
            "args": {"correlation_id": 7},
        },
        {
            "ph": "X",
            "pid": 2,
            "tid": 20,
            "ts": 35,
            "dur": 5,
            "args": {"level": "acl", "connection_id": 99},
        },
        device_task,
    ]

    assert associate_cann_launches(events, {"node": scope}) == {"node": [device_task]}


def test_collective_identity_and_counts_survive_execution_reordering():
    graph = fx.Graph()
    x = graph.placeholder("x")
    reduce = torch.ops._c10d_functional.all_reduce.default
    first = graph.call_function(reduce, (x, "sum", "local_group_a"))
    second = graph.call_function(reduce, (x,), {"reduce_op": "sum", "group_name": "local_group_b"})
    repeated = graph.call_function(reduce, (x, "sum", "local_group_a"))
    gather = graph.call_function(torch.ops._c10d_functional.all_gather_into_tensor.default, (x, 2, "local_group_a"))
    wait = graph.call_function(torch.ops._c10d_functional.wait_tensor.default, (first,))
    graph.output((wait, second, repeated, gather))
    gm = fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    first.prepend(second)

    keys = collective_order_keys(gm)
    manifest = _collective_count_manifest(gm)

    assert keys == {
        first: (0, "_c10d_functional::all_reduce", 0),
        second: (1, "_c10d_functional::all_reduce", 0),
        repeated: (0, "_c10d_functional::all_reduce", 1),
        gather: (0, "_c10d_functional::all_gather_into_tensor", 0),
    }
    assert manifest == (
        (0, "_c10d_functional::all_gather_into_tensor", 1),
        (0, "_c10d_functional::all_reduce", 2),
        (1, "_c10d_functional::all_reduce", 1),
    )


def test_deepep_stages_share_ordered_communication_group():
    graph = fx.Graph()
    x = graph.placeholder("x")
    stages = []
    for schema_name in (
        "deepep::dispatch",
        "deepep::combine",
        "deepep::dispatch_backward",
        "deepep::combine_backward",
    ):

        def stage(value):
            return value

        stage.__name__ = schema_name.removeprefix("deepep::")
        stage._schema = SimpleNamespace(name=schema_name, arguments=())
        stages.append(graph.call_function(stage, (x,)))
    graph.output(tuple(stages))
    gm = fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)

    keys = collective_order_keys(gm)
    manifest = _collective_count_manifest(gm)

    assert list(keys.values()) == [
        (0, "deepep::dispatch", 0),
        (0, "deepep::combine", 0),
        (0, "deepep::dispatch_backward", 0),
        (0, "deepep::combine_backward", 0),
    ]
    assert manifest == (
        (0, "deepep::combine", 1),
        (0, "deepep::combine_backward", 1),
        (0, "deepep::dispatch", 1),
        (0, "deepep::dispatch_backward", 1),
    )


def test_native_profiler_step_preserves_fx_host_matching():
    graph = fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_function(torch.ops.aten.sin.default, (x,))
    second = graph.call_function(torch.ops.aten.sin.default, (first,))
    graph.output(second)
    gm = fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    common = {"cat": "cpu_op", "ph": "X", "pid": 1, "tid": 1}
    first_scope = dict(common, name="aten::sin", ts=10, dur=20)
    second_scope = dict(common, name="aten::sin", ts=40, dur=20)
    trace = {
        "traceEvents": [
            dict(common, name="ProfilerStep#0", ts=0, dur=100),
            first_scope,
            dict(common, name="aclnnSin", ts=12, dur=5),
            second_scope,
        ]
    }

    matched, stats = _match_fx_nodes_to_host_scopes(trace, gm)

    assert matched == {node_id(first): first_scope, node_id(second): second_scope}
    assert stats["matched"] == 2
    assert stats["missing"] == 0


@pytest.mark.parametrize("wrapped", [False, True], ids=["native_scope_only", "outer_wrapper_scope"])
def test_permute_native_scope_maps_device_cost_without_shifting_later_calls(wrapped, tmp_path):
    graph = fx.Graph()
    x = graph.placeholder("x")
    indices = graph.placeholder("indices")
    permute = graph.call_function(torch.ops.torchtitan_npu.npu_moe_token_permute.default, (x, indices))
    sine = graph.call_function(torch.ops.aten.sin.default, (x,))
    graph.output((permute, sine))
    gm = fx.GraphModule({}, graph)
    assign_stable_node_tags(gm)
    host = {"cat": "cpu_op", "ph": "X", "pid": 1, "tid": 1}
    scopes = [
        dict(host, name="npu::npu_moe_token_permute", ts=10, dur=10),
        dict(host, name="aten::sin", ts=30, dur=10),
    ]
    if wrapped:
        scopes.insert(0, dict(host, name="torchtitan_npu::npu_moe_token_permute", ts=9, dur=12))
    flow = {"cat": "async_npu", "name": "torch_to_npu"}
    trace = {
        "traceEvents": scopes
        + [{"ph": "M", "name": "process_name", "pid": 2, "args": {"name": "Ascend Hardware"}}]
        + [
            event
            for index, (ts, duration) in enumerate(((12, 280), (32, 20)))
            for event in (
                dict(flow, id=index, ph="s", pid=1, tid=1, ts=ts),
                dict(flow, id=index, ph="f", pid=2, tid=7, ts=1000 + index * 1000),
                {
                    "ph": "X",
                    "name": "kernel",
                    "pid": 2,
                    "tid": 7,
                    "ts": 1000 + index * 1000,
                    "dur": duration,
                    "args": {"Task Type": "MIX_AIV" if index == 0 else "AI_VECTOR_CORE", "Physic Stream Id": 7},
                },
            )
        ]
    }
    with (tmp_path / "op_summary_0.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Op Name",
                "Task Start Time(us)",
                "Task Duration(us)",
                "Stream Id",
                "aicore_time(us)",
                "aiv_time(us)",
                "aic_mac_time(us)",
                "aiv_vec_time(us)",
            ]
        )
        # Same launch name/time on another stream must not supply its counters.
        writer.writerow(["kernel", 1000, 280, 8, 140, 140, 250, 5])
        writer.writerow(["kernel", 1000, 280, 7, 140, 140, 5, 250])
    observations = {}

    matched, stats = _match_fx_nodes_to_host_scopes(trace, gm)
    costs, resources = _extract_cann_node_measurements(
        trace,
        {node_id(n): n for n in (permute, sine)},
        matched,
        kernel_metrics=CannKernelMetrics(tmp_path),
        kernel_observations=observations,
    )

    assert stats["missing"] == 0
    assert costs == {node_id(permute): 0.28, node_id(sine): 0.02}
    assert resources == {node_id(permute): "mix_aiv", node_id(sine): "aiv"}
    assert observations[node_id(permute)] == [(0.0, 0.28, "(2, 7)", "mix_aiv")]


@pytest.mark.parametrize("with_counters", [False, True], ids=["unclassified_mix", "classified_mix"])
def test_profile_summary_counts_only_pure_cv_overlap(tmp_path, with_counters):
    output = tmp_path / "ASCEND_PROFILER_OUTPUT"
    output.mkdir()
    (output / "trace_view.json").write_text(
        json.dumps({"traceEvents": [{"ph": "X", "name": SCOPE, "ts": 500, "dur": 3000}]})
    )
    with (output / "kernel_details.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Name",
                "Device_id",
                "Stream ID",
                "Accelerator Core",
                "Start Time(us)",
                "Duration(us)",
                "aic_time(us)",
                "aiv_time(us)",
            ]
        )
        writer.writerow(["cube", 0, 0, "AI_CORE", 1000, 2000, "", ""])
        # Vector/MIX overlap on the same stream must not count as pure CV.
        writer.writerow(["vector", 0, 1, "AI_VECTOR_CORE", 1500, 1250, "", ""])
        writer.writerow(["vector", 0, 0, "AI_VECTOR_CORE", 3000, 500, "", ""])
        writer.writerow(["mix", 0, 1, "MIX_AIV", 2500, 500, *([5, 450] if with_counters else ["", ""])])
        writer.writerow(["mix", 0, 1, "MIX_AIC", 3000, 500, *([450, 5] if with_counters else ["", ""])])
    profile_link = tmp_path / "training_profile"
    profile_link.symlink_to(output, target_is_directory=True)

    summary = summarize_profile(profile_link)

    assert summary["time_ms"]["cv_overlap"] == 1.0
    assert summary["time_ms"]["compute_union"] == 2.5
    assert summary["cv_percent_of_compute"] == 40.0
    assert summary["kernel_counts"] == {"cube": 1, "vector": 2, "mix": 2}
    assert summary["mixed_kernel_counts"] == ({"mix_aiv": 1, "mix_aic": 1} if with_counters else {"mix": 2})
    assert summary["time_ms"]["cube_mix_aiv_overlap"] == (0.25 if with_counters else 0.0)
    assert summary["time_ms"]["vector_mix_aic_overlap"] == (0.5 if with_counters else 0.0)
    assert summary["time_ms"]["mixed_overlap"] == 1.0
    assert summary["time_ms"]["mix_busy"] == 1.0
    assert summary["time_ms"]["forward_backward_scope"] == 3.0
    assert summary["time_ms"]["overlap"] == 2.0
    assert summary["time_ms"]["single_stream"] == 0.5
    assert summary["time_ms"]["kernel_duration_sum"] == 4.75
    assert summary["stream_busy_ms"] == {"0": 2.5, "1": 2.0}
    assert summary["overlap_percent"] == 80.0
    assert summary["cube_covered_by_vector_percent"] == 50.0
