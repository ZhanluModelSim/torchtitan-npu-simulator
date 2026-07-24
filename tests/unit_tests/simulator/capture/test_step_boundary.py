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


def test_build_step_graphs_buckets_by_stage_and_comp_type():
    nodes = {
        "f1": _node("f1", "forward"),
        "b1": _node("b1", "backward"),
        "o1": _node("o1", "optimizer"),
    }
    graphs = build_step_graphs(nodes)
    assert set(graphs.keys()) == {"s-1_F", "s-1_B", "s-1_OPTIMIZER"}
    assert graphs["s-1_F"].step_type == "F"
    assert "f1" in graphs["s-1_F"].nodes
    assert "b1" in graphs["s-1_B"].nodes
    assert "o1" in graphs["s-1_OPTIMIZER"].nodes


def test_build_step_graphs_defaults_missing_phase_to_forward():
    node_without_phase = OpNode(
        op_id="x", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[], annotations={},
    )
    graphs = build_step_graphs({"x": node_without_phase})
    assert "x" in graphs["s-1_F"].nodes


def test_build_step_graphs_skips_empty_phases():
    nodes = {"f1": _node("f1", "forward")}
    graphs = build_step_graphs(nodes)
    assert "s-1_B" not in graphs
    assert "s-1_OPTIMIZER" not in graphs


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
    assert "s-1_F" in graphs
    assert "s-1_B" in graphs
    relu_nodes = [
        node
        for node in graphs["s-1_F"].nodes.values()
        if "relu" in node.annotations["raw_op_type"]
    ]
    assert relu_nodes, "relu should have been captured during the forward phase"
