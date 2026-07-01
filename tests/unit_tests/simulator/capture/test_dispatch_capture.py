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
