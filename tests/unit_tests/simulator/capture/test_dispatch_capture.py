# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.module_path import ModulePathTracker


def test_capture_records_ops_on_meta_tensors():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(4, 8, device="meta")
        b = torch.randn(8, 16, device="meta")
        c = a @ b
        c.sum()
    nodes = capture.build_nodes()
    assert len(nodes) >= 3  # randn, randn, matmul, sum (at least)


def test_capture_builds_predecessor_successor_edges():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(4, 8, device="meta")
        b = a.relu()
        b.sum()
    nodes = capture.build_nodes()
    relu_nodes = [n for n in nodes.values() if "relu" in n.annotations["raw_op_type"]]
    assert len(relu_nodes) == 1
    relu_node = relu_nodes[0]
    assert len(relu_node.predecessors) == 1
    producer = nodes[relu_node.predecessors[0]]
    assert relu_node.op_id in producer.successors


def test_capture_deduplicates_consecutive_identical_ops():
    capture = OpDispatchCapture()
    with capture:
        x = torch.zeros(4, device="meta")
        for _ in range(5):
            x = x.relu()
    nodes = capture.build_nodes()
    relu_nodes = [n for n in nodes.values() if "relu" in n.annotations["raw_op_type"]]
    assert len(relu_nodes) == 1
    assert relu_nodes[0].annotations["repeat_count"] == 5


def test_capture_tags_module_path_when_tracker_supplied():
    model = nn.Sequential(nn.Linear(4, 8, device="meta"), nn.ReLU())
    tracker = ModulePathTracker(model)
    capture = OpDispatchCapture(module_path_tracker=tracker)
    with tracker, capture:
        model(torch.randn(2, 4, device="meta"))
    nodes = capture.build_nodes()
    tagged = [n for n in nodes.values() if "module_path" in n.annotations]
    assert tagged, "expected at least one op tagged with a module_path"
    assert any("0" in n.annotations["module_path"] for n in tagged)  # Sequential child "0" (Linear)


def test_unknown_op_type_is_flagged_in_annotations():
    capture = OpDispatchCapture()
    with capture:
        # aten.arange.default has no entry in OP_MAPPING -> canonical "unknown"
        torch.arange(4, device="meta")
    nodes = capture.build_nodes()
    unknown_nodes = [n for n in nodes.values() if n.op_type == "unknown"]
    assert unknown_nodes
    assert all(n.annotations.get("cost_unknown") for n in unknown_nodes)


def test_phase_provider_tags_every_node_and_defaults_to_forward():
    capture_no_provider = OpDispatchCapture()
    with capture_no_provider:
        torch.randn(2, 2, device="meta")
    default_nodes = capture_no_provider.build_nodes()
    assert all(n.annotations["phase"] == "forward" for n in default_nodes.values())

    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        torch.randn(2, 2, device="meta")
        phase_box["value"] = "backward"
        torch.randn(2, 2, device="meta")
    nodes = capture.build_nodes()
    phases = sorted({n.annotations["phase"] for n in nodes.values()})
    assert phases == ["backward", "forward"]


def test_record_synthetic_op_creates_a_node_with_given_raw_op_type():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(2, 4, device="meta")
        b = torch.empty(2, 4, device="meta")
        capture.record_synthetic_op("triton.hc_pre_bmm_forward", inputs=[a], outputs=[b])
    nodes = capture.build_nodes()
    synthetic = [n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.hc_pre_bmm_forward"]
    assert len(synthetic) == 1
    assert synthetic[0].op_type == "unknown"  # not in OP_MAPPING -- expected, display_op_label handles it
    assert [o.shape for o in synthetic[0].outputs] == [(2, 4)]


def test_capture_keeps_uncollapsed_memory_events_for_liveness():
    capture = OpDispatchCapture()
    with capture:
        x = torch.zeros(4, device="meta")
        for _ in range(3):
            x = x.relu()
    nodes = capture.build_nodes()
    relu_nodes = [n for n in nodes.values() if "relu" in n.annotations["raw_op_type"]]
    relu_memory_events = [e for e in capture.memory_events() if "relu" in e.raw_op_type]
    assert len(relu_nodes) == 1
    assert relu_nodes[0].annotations["repeat_count"] == 3
    assert len(relu_memory_events) == 3
    assert len({e.seq_idx for e in relu_memory_events}) == 3


def test_capture_returns_stable_id_for_same_live_tensor():
    capture = OpDispatchCapture()
    tensor = torch.zeros(4, device="meta")

    assert capture.tensor_id(tensor) == capture.tensor_id(tensor)


def test_record_synthetic_op_wires_producer_consumer_edges():
    capture = OpDispatchCapture()
    with capture:
        a = torch.randn(2, 4, device="meta")
        mid = torch.empty(2, 4, device="meta")
        capture.record_synthetic_op("triton.step_one", inputs=[a], outputs=[mid])
        out = mid.relu()  # a REAL dispatched op consuming the synthetic op's output
    nodes = capture.build_nodes()
    synthetic_node = next(n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.step_one")
    relu_node = next(n for n in nodes.values() if "relu" in n.annotations["raw_op_type"])
    assert relu_node.predecessors == [synthetic_node.op_id]
    assert synthetic_node.op_id in relu_node.predecessors
    assert relu_node.op_id in synthetic_node.successors


def test_record_synthetic_op_respects_phase_provider():
    phase_box = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase_box["value"])
    with capture:
        a = torch.randn(2, device="meta")
        b = torch.empty(2, device="meta")
        capture.record_synthetic_op("triton.fwd_step", inputs=[a], outputs=[b])
        phase_box["value"] = "backward"
        c = torch.empty(2, device="meta")
        capture.record_synthetic_op("triton.bwd_step", inputs=[b], outputs=[c])
    nodes = capture.build_nodes()
    fwd_node = next(n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.fwd_step")
    bwd_node = next(n for n in nodes.values() if n.annotations.get("raw_op_type") == "triton.bwd_step")
    assert fwd_node.annotations["phase"] == "forward"
    assert bwd_node.annotations["phase"] == "backward"


def test_get_active_capture_returns_none_outside_context():
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    assert get_active_capture() is None


def test_get_active_capture_returns_the_entered_instance():
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    capture = OpDispatchCapture()
    with capture:
        assert get_active_capture() is capture
    assert get_active_capture() is None
