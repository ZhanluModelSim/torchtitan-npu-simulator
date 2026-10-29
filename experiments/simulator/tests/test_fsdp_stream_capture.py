# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native FSDP event waits survive assembly into the final workload DAG."""

import inspect

import pytest
from torchtitan_simulator.capture.dispatch_capture import get_active_capture
from torchtitan_simulator.compat.device import MetaDeviceModule
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


@pytest.mark.parametrize(
    "config,num_mtp", [("deepseek_v4_smoketest", 1), ("kimi_k3_smoketest", 0)], ids=["dsv4", "kimi"]
)
def test_native_hsdp_partial_backward_wait_delays_later_matmul(monkeypatch, config, num_mtp):
    observed = []
    original = MetaDeviceModule.Stream.wait_event

    def wait_event(stream, event):
        frame = inspect.currentframe().f_back
        try:
            # The native partial-group path waits for post-reduce accumulation
            # before returning to autograd (e.g. a subsequent loss chunk).
            if frame.f_code.co_name == "post_backward" and "_fsdp_param_group" in frame.f_code.co_filename:
                group = frame.f_locals["self"]
                if getattr(group, "_post_reduce_event", None) is event:
                    capture = get_active_capture()
                    observed.append((event._frontier, capture.stream_id(stream)))
        finally:
            del frame
        return original(stream, event)

    monkeypatch.setattr(MetaDeviceModule.Stream, "wait_event", wait_event)
    workload = simulate(
        SimulationConfig(config=config, world_size=4, data_parallel_replicate_degree=2, num_mtp_modules=num_mtp)
    )
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    captured = workload.captured_workload or workload
    source_nodes = {
        node.op_id: node for template in captured.step_templates.values() for node in template.nodes.values()
    }

    def visible_frontier(frontier):
        result, pending = set(), list(frontier)
        seen = set()
        while pending:
            source = pending.pop()
            if source in seen:
                continue
            seen.add(source)
            if source in nodes:
                result.add(source)
            else:
                pending.extend(source_nodes[source].predecessors)
        return result

    for node in nodes.values():
        for source in node.annotations.get("stream_predecessors", []) + node.annotations.get("event_predecessors", []):
            assert source in node.predecessors

    def starts(delayed_id=None):
        result, finishes = {}, {}
        for node in sorted(nodes.values(), key=lambda item: item.seq_idx):
            result[node.op_id] = max((finishes[pred] for pred in node.predecessors), default=0)
            cost = 0 if node.op_type == "stream_wait" else (1_000_000 if node.op_id == delayed_id else 1)
            finishes[node.op_id] = result[node.op_id] + cost
        return result

    baseline = starts()
    checked = 0
    for frontier, stream_id in observed:
        reductions = [source for source in visible_frontier(frontier) if nodes[source].op_type == "allreduce"]
        if not reductions:
            continue
        reduction = reductions[-1]
        waits = [
            node
            for node in nodes.values()
            if node.op_type == "stream_wait"
            and node.annotations["stream_id"] == stream_id
            and reduction in node.predecessors
        ]
        assert waits
        wait = min(waits, key=lambda node: node.seq_idx)
        consumers = [
            node
            for node in nodes.values()
            if node.op_type == "matmul"
            and node.annotations.get("stream_id") == stream_id
            and node.seq_idx > wait.seq_idx
        ]
        if consumers:
            consumer = min(consumers, key=lambda node: node.seq_idx)
            assert starts(reduction)[consumer.op_id] > baseline[consumer.op_id] + 900_000
            checked += 1
    assert checked > 0, "Native HSDP post-reduce wait must gate a later loss-chunk matmul"


def test_kimi_fsdp_allgather_storage_and_completion_gate_attention_weight_use():
    workload = simulate(SimulationConfig(config="kimi_k3_smoketest", world_size=2, seq_len=16))
    annotations = workload.iteration.schedule.annotations
    memory = annotations["memory_plan"]
    projection = next(
        event
        for event in memory.raw_events
        if event.module_path.endswith("layers.0.attention.q_proj") and event.raw_op_type == "aten.mm.default"
    )
    assert any(
        projection.inputs[1].storage_key in event.storage_keys
        for event in memory.fsdp_residency_events
        if event.action == "alloc" and event.module_fqn == "layers.0"
    )
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    gathers = [
        event
        for event in annotations["communication_events"]
        if event["fsdp_module_fqn"] == "layers.0"
        and event["comp_type"] == "F"
        and nodes[event["op_id"]].op_type == "allgather"
    ]
    assert gathers
    gather = min(gathers, key=lambda event: event["seq_idx"])["op_id"]

    def compute_start(delay):
        finishes = {}
        for node in sorted(nodes.values(), key=lambda item: item.seq_idx):
            start = max((finishes[parent] for parent in node.predecessors), default=0)
            duration = 0 if node.op_type == "stream_wait" else (1_000_000 if delay == node.op_id else 1)
            finishes[node.op_id] = start + duration
            if node.op_id == projection.op_id:
                return start
        raise AssertionError("Attention projection disappeared from the workload graph")

    assert compute_start(gather) > compute_start(None) + 900_000
