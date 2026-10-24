# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ownership changes accounting, never the captured operator DAG."""

from torchtitan_simulator.capture.comm_events import CommEvent
from torchtitan_simulator.capture.communication_ownership import normalize_communication_ownership
from torchtitan_simulator.capture.schedule_assemblers import ActionSpec
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.ir.step_graph import StepGraph


def _node(op_id, raw, predecessors=(), **annotations):
    return OpNode(
        op_id=op_id,
        op_type="stream_wait" if raw == "simulator.stream_wait" else "unknown",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=list(predecessors),
        successors=[],
        annotations={"raw_op_type": raw, **annotations},
    )


def _event(op_id, primitive, *, transition="", gradient_reduction=False):
    return CommEvent(
        event_id=f"comm-{op_id}",
        comm_primitive=primitive,
        group_name="fsdp",
        world_size=2,
        tensor_shape=(16,),
        dtype="float32",
        volume_bytes=64,
        op_id=op_id,
        fsdp_transition_id=transition,
        gradient_reduction=gradient_reduction,
    )


def _residency(action, transition, *, source="state"):
    return ActionSpec(
        action_type=action,
        stage=0,
        mb_idx=0,
        seq_idx=0,
        order_key=(0, 0, 0),
        annotations={
            "fsdp_transition_id": transition,
            "fsdp_schedule_source": source,
            "shard_world_size": 2,
            "parent_compute_instance_id": "forward",
        },
    )


def test_ownership_keeps_every_native_node_and_dependency_including_unused_prefetch():
    nodes = {
        1: _node(1, "c10d._allgather_base_.default"),
        2: _node(2, "aten.view.default", [1]),
        3: _node(3, "simulator.stream_wait", [1], control_node=True, zero_duration=True),
        4: _node(4, "aten.mm.default", [2, 3]),
        5: _node(5, "c10d._allgather_base_.default", [1]),
        6: _node(6, "simulator.stream_wait", [5], control_node=True, zero_duration=True),
        7: _node(7, "c10d._allgather_base_.default", [6]),
    }
    # An unconsumed launch is real communication, not a missing transition to
    # reconstruct. Repeated ownership assignment must preserve its identity.
    nodes[5].comm_bytes = 64
    for op_id, node in nodes.items():
        for predecessor in node.predecessors:
            nodes[predecessor].successors.append(op_id)
    graph = StepGraph("s0_F", "F", nodes)
    templates = {"s0_F": graph}
    snapshot = {i: (node, list(node.predecessors), list(node.successors)) for i, node in nodes.items()}
    events = [
        _event(1, "allgather", transition="loaded"),
        _event(5, "allgather", transition="unused"),
        _event(7, "allgather", transition="future"),
    ]
    compute = ActionSpec("COMPUTE", 0, 0, 0, (0, 0, 0), comp_type="F", template_ref="s0_F")
    specs = [compute, _residency("UNSHARD", "loaded"), _residency("RESHARD", "loaded")]
    for _ in range(2):
        result = normalize_communication_ownership(step_templates=templates, specs=specs, comm_events=events)
        assert result.specs == [compute]
        assert result.internal_fsdp_transitions == {"loaded"}
        assert set(templates) == {"s0_F"}
        assert templates["s0_F"] is graph
        assert set(graph.nodes) == set(snapshot)
        for op_id, (original, predecessors, successors) in snapshot.items():
            assert graph.nodes[op_id] is original
            assert original.predecessors == predecessors
            assert original.successors == successors
        assert graph.nodes[5].comm_bytes == 64


def test_ownership_deduplicates_gradient_actions_without_treating_cp_as_gradient():
    rs = _node(1, "c10d._reduce_scatter_base_.default", gradient_reduction=True)
    ar = _node(2, "c10d.allreduce_.default", [1], gradient_reduction=True)
    cp = _node(3, "c10d.allreduce_.default", gradient_reduction=False)
    templates = {"s0_B": StepGraph("s0_B", "B", {1: rs, 2: ar, 3: cp})}
    events = [_event(1, "reduce_scatter", gradient_reduction=True), _event(2, "allreduce", gradient_reduction=True)]
    reductions = [ActionSpec("REDUCE_GRAD", 0, 0, 0, (0, 0, 0), comm_op_id=i) for i in (1, 2)]
    result = normalize_communication_ownership(step_templates=templates, specs=reductions, comm_events=events)
    assert result.specs == []
    assert result.internal_gradient_reductions == 2
    assert cp.annotations["gradient_reduction"] is False
    assert ar.predecessors == [1]


def test_ownership_preserves_standalone_unshard_and_external_prefetch_actions():
    nodes = {1: _node(1, "c10d._allgather_base_.default")}
    templates = {"s0_UNSHARD": StepGraph("s0_UNSHARD", "UNSHARD", nodes)}
    standalone = _residency("UNSHARD", "standalone")
    prefetch = _residency("UNSHARD", "prefetch", source="intent")
    result = normalize_communication_ownership(
        step_templates=templates,
        specs=[standalone, prefetch],
        comm_events=[_event(1, "allgather", transition="standalone")],
    )
    assert result.specs == [standalone, prefetch]
    assert result.internal_fsdp_transitions == set()
    assert result.external_fsdp_transitions == {"prefetch"}
    assert standalone.annotations["communication_owner"] == "L2_STANDALONE"
    assert prefetch.annotations["communication_owner"] == "L2_PREFETCH"
    assert templates["s0_UNSHARD"].nodes == nodes
