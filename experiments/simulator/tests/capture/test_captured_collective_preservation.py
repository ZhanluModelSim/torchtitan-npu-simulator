# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
from torchtitan_simulator.capture.communication_ownership import CapturedCollectivePreservationPlugin
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.step_graph import StepGraph


def node(op_id, predecessors=(), **kwargs):
    return OpNode(
        op_id=op_id,
        op_type="allgather",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=list(predecessors),
        successors=[],
        **kwargs,
    )


def test_unconsumed_launch_survives_with_actual_ancestry_and_wire_bytes():
    source = node(1, seq_idx=10)
    marker = node(2, [1], seq_idx=11)
    launch = node(3, [2], seq_idx=12, comm_bytes=128)
    graph = StepGraph("s0_B", "B", {1: source, 2: marker, 3: launch})
    templates = {"s0_B": graph}
    events = [SimpleNamespace(op_id=1), SimpleNamespace(op_id=3)]
    plugin = CapturedCollectivePreservationPlugin(templates, events)
    # Normalization moves the first launch, removes its marker, and loses
    # the final launch because no completed compute transition follows it.
    graph.nodes = {-1: node(-1, seq_idx=10, annotations={"source_op_id": 1})}
    plugin.apply(step_templates=templates, specs=[], comm_events=events)
    restored = graph.nodes[3]
    assert restored.predecessors == [-1]
    assert restored.seq_idx == 12
    assert restored.comm_bytes == 128
    assert restored.successors == []
    assert graph.nodes[-1].successors == [3]
    assert graph.comm_volume == 128
    assert graph.is_acyclic
    # Re-applying cannot duplicate a launch.
    plugin.apply(step_templates=templates, specs=[], comm_events=events)
    assert set(graph.nodes) == {-1, 3}


def test_missing_capture_node_is_an_error_instead_of_synthetic_communication():
    plugin = CapturedCollectivePreservationPlugin({}, [SimpleNamespace(op_id=7)])
    with pytest.raises(ValueError, match="no source node"):
        plugin.apply(step_templates={}, specs=[], comm_events=[])
