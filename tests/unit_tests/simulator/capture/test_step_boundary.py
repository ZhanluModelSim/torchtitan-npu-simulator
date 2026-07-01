# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.step_boundary import StepBoundaryTracker, build_step_graphs
from torchtitan_npu.simulator.ir.op_node import OpNode


def _node(op_id: str, phase: str) -> OpNode:
    return OpNode(
        op_id=op_id, op_type="x", inputs=[], outputs=[], attrs={},
        predecessors=[], successors=[], annotations={"phase": phase},
    )


def test_build_step_graphs_buckets_by_phase():
    nodes = {
        "f1": _node("f1", "forward"),
        "b1": _node("b1", "backward"),
        "o1": _node("o1", "optimizer"),
    }
    graphs = build_step_graphs(nodes)
    assert set(graphs.keys()) == {"forward", "backward", "optimizer"}
    assert graphs["forward"].step_type == "forward"
    assert "f1" in graphs["forward"].nodes
    assert "b1" in graphs["backward"].nodes
    assert "o1" in graphs["optimizer"].nodes


def test_build_step_graphs_defaults_missing_phase_to_forward():
    node_without_phase = OpNode(
        op_id="x", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[], annotations={},
    )
    graphs = build_step_graphs({"x": node_without_phase})
    assert "x" in graphs["forward"].nodes


def test_build_step_graphs_skips_empty_phases():
    nodes = {"f1": _node("f1", "forward")}
    graphs = build_step_graphs(nodes)
    assert "backward" not in graphs
    assert "optimizer" not in graphs


def test_step_boundary_tracker_flips_phase_on_backward_call():
    tracker = StepBoundaryTracker()
    with tracker:
        assert tracker.current_phase == "forward"
        x = torch.randn(4, device="meta", requires_grad=True)
        x.sum().backward()
        assert tracker.current_phase == "backward"


def test_step_boundary_tracker_restores_original_backward_on_exit():
    original = torch.Tensor.backward
    tracker = StepBoundaryTracker()
    with tracker:
        pass
    assert torch.Tensor.backward is original


def test_step_boundary_tracker_mark_sets_phase_explicitly():
    tracker = StepBoundaryTracker()
    with tracker:
        tracker.mark("optimizer")
        assert tracker.current_phase == "optimizer"


def test_step_boundary_tracker_integrates_with_dispatch_capture():
    tracker = StepBoundaryTracker()
    capture = OpDispatchCapture(phase_provider=lambda: tracker.current_phase)
    with tracker, capture:
        x = torch.randn(4, device="meta", requires_grad=True)
        y = x.relu()
        y.sum().backward()
    nodes = capture.build_nodes()
    graphs = build_step_graphs(nodes)
    assert "forward" in graphs
    assert "backward" in graphs
    relu_nodes = [n for n in graphs["forward"].nodes.values() if "relu" in n.annotations["raw_op_type"]]
    assert relu_nodes, "relu should have been captured during the forward phase"
