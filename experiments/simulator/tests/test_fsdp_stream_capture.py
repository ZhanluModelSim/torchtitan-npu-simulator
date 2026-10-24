# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native FSDP event waits survive assembly into the final workload DAG."""

import inspect

from torchtitan_simulator.capture.dispatch_capture import get_active_capture
from torchtitan_simulator.compat.device import MetaDeviceModule
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


def test_native_hsdp_partial_backward_wait_delays_later_matmul(monkeypatch):
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
    workload = simulate(SimulationConfig(world_size=4, data_parallel_replicate_degree=2, num_mtp_modules=1))
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    sources = {node.op_id: node.op_id for node in nodes.values()}
    for node in nodes.values():
        for source in node.annotations.get("stream_predecessors", []) + node.annotations.get("event_predecessors", []):
            assert sources[source] in node.predecessors

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
        reductions = [sources[source] for source in frontier if nodes[sources[source]].op_type == "allreduce"]
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
