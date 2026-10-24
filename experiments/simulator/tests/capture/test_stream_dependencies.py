# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device waits constrain DAG schedules, including tensor-independent work."""

import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.compat.device import MetaDeviceModule


def _start_times(nodes, delayed_id=None):
    starts, finishes = {}, {}
    for node in sorted(nodes.values(), key=lambda node: node.seq_idx):
        starts[node.op_id] = max((finishes[pred] for pred in node.predecessors), default=0)
        duration = 0 if node.op_type == "stream_wait" else (1000 if node.op_id == delayed_id else 1)
        finishes[node.op_id] = starts[node.op_id] + duration
    return starts


def test_event_wait_delays_later_compute_but_preserves_independent_stream(monkeypatch):
    device = MetaDeviceModule()
    monkeypatch.setattr(torch, "meta", device, raising=False)
    reduction_stream, unrelated_stream = device.Stream(), device.Stream()
    x = torch.empty(4, 4, device="meta")
    capture = OpDispatchCapture(record_memory=True)
    with capture:
        with device.stream(reduction_stream):
            rs = capture.record_synthetic_op("comm.reduce_scatter", [x], [x])
            ar = capture.record_synthetic_op("comm.allreduce", [x], [x])
            event = reduction_stream.record_event()
        device.current_stream().wait_event(event)
        # A host-only metadata op between wait and compute must not swallow it.
        y = torch.empty(4, 4, device="meta")
        y.view(2, 8)
        y @ y
        with device.stream(unrelated_stream):
            z = torch.empty(4, 4, device="meta")
            z.sin()
    nodes = capture.build_nodes()
    consumer = next(node for node in nodes.values() if node.op_type == "matmul")
    independent = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.sin.default")
    baseline = _start_times(nodes)
    for delayed in (rs, ar):
        shifted = _start_times(nodes, delayed)
        assert shifted[consumer.op_id] > baseline[consumer.op_id] + 900
        assert shifted[independent.op_id] == baseline[independent.op_id]
    wait = next(node for node in nodes.values() if node.op_type == "stream_wait")
    assert ar in wait.predecessors
    assert wait.annotations["zero_duration"]
    assert not wait.annotations.get("cost_unknown")
    assert (wait.flops, wait.peak_mem, wait.param_mem, wait.comm_bytes) == (0, 0, 0, 0)
    assert wait.inputs == wait.outputs == []
    assert wait.op_id not in {event.op_id for event in capture.memory_events()}


def test_event_snapshot_and_rerecord_do_not_wait_for_future_source_work(monkeypatch):
    device = MetaDeviceModule()
    monkeypatch.setattr(torch, "meta", device, raising=False)
    source, destination = device.Stream(), device.Stream()
    x = torch.empty(4, device="meta")
    capture = OpDispatchCapture()
    with capture:
        with device.stream(source):
            first = capture.record_synthetic_op("comm.allreduce", [x], [x])
            event = source.record_event()
            later = capture.record_synthetic_op("comm.allreduce", [x], [x])
        destination.wait_event(event)
        first_wait_id = capture._events[-1].op_id
        source.record_event(event)
        destination.wait_event(event)
        second_wait_id = capture._events[-1].op_id
    nodes = capture.build_nodes()
    assert first in nodes[first_wait_id].predecessors
    assert later not in nodes[first_wait_id].predecessors
    assert later in nodes[second_wait_id].predecessors
    assert first_wait_id in nodes[second_wait_id].predecessors


def test_wait_stream_orders_tensor_independent_submissions_and_keeps_default_stream(monkeypatch):
    device = MetaDeviceModule()
    monkeypatch.setattr(torch, "meta", device, raising=False)
    source = device.Stream()
    capture = OpDispatchCapture()
    default = device.current_stream()
    with capture:
        with device.stream(source):
            torch.empty(4, device="meta").sin()
        default.wait_stream(source)
        torch.empty(4, device="meta").cos()
        torch.empty(4, device="meta").exp()
    assert device.current_stream() is default
    nodes = capture.build_nodes()
    source_op = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.sin.default")
    last_op = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.exp.default")
    assert _start_times(nodes, source_op.op_id)[last_op.op_id] > _start_times(nodes)[last_op.op_id] + 900


def test_old_event_does_not_reference_previous_capture_nodes(monkeypatch):
    device = MetaDeviceModule()
    monkeypatch.setattr(torch, "meta", device, raising=False)
    with OpDispatchCapture() as previous:
        torch.empty(4, device="meta").sin()
        event = device.current_stream().record_event()
    with OpDispatchCapture() as capture:
        event.wait()
        torch.empty(4, device="meta").cos()
    nodes = capture.build_nodes()
    old_ids = set(previous.build_nodes())
    assert all(not old_ids.intersection(node.predecessors) for node in nodes.values())


def test_metadata_alias_does_not_serialize_an_unwaited_stream(monkeypatch):
    device = MetaDeviceModule()
    monkeypatch.setattr(torch, "meta", device, raising=False)
    source = device.Stream()
    capture = OpDispatchCapture()
    with capture:
        with device.stream(source):
            x = torch.empty(4, device="meta").sin()
        x.view(2, 2)
        torch.empty(4, device="meta").cos()
    nodes = capture.build_nodes()
    source_op = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.sin.default")
    independent = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.cos.default")
    assert _start_times(nodes, source_op.op_id)[independent.op_id] == _start_times(nodes)[independent.op_id]
