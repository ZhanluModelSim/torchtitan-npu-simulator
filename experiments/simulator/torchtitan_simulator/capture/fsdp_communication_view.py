# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Project FSDP bookkeeping out of the consumer graph, retaining capture facts."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from typing import TYPE_CHECKING

from torchtitan_simulator.capture.op_mapping import communication_primitive

if TYPE_CHECKING:
    from torchtitan_simulator.ir.op_node import OpNode
    from torchtitan_simulator.ir.schedule_plan import ScheduleAction
    from torchtitan_simulator.ir.workload_graph import WorkloadGraph


_FSDP_SCOPES = {
    "fsdp_unshard_scaffold",
    "fsdp_unshard_wait_scaffold",
    "fsdp_reshard_scaffold",
    "fsdp_post_backward_scaffold",
    "fsdp_state_pre_forward_scaffold",
    "fsdp_state_pre_backward_scaffold",
    "fsdp_state_post_forward_scaffold",
    "fsdp_finalize_backward_scaffold",
    "replicate_unshard_scaffold",
    "replicate_unshard_wait_scaffold",
    "replicate_reshard_scaffold",
}


def is_fsdp_bookkeeping(node: OpNode) -> bool:
    """Use observer-owned execution scopes, never a model's wrapper-class FQN."""
    return (
        bool(_FSDP_SCOPES.intersection(node.annotations.get("framework_scopes", ())))
        and communication_primitive(node.annotations.get("raw_op_type", "")) is None
        and node.op_type != "stream_wait"
    )


def _frontiers(nodes, hidden, *, reverse=False):
    links = {key: node.successors if reverse else node.predecessors for key, node in nodes.items()}
    pending = {key: len(set(parents)) for key, parents in links.items()}
    followers = {key: node.predecessors if reverse else node.successors for key, node in nodes.items()}
    ready = deque(key for key, count in pending.items() if count == 0)
    result = {}
    while ready:
        key = ready.popleft()
        result[key] = set().union(*(result[parent] for parent in links[key])) if key in hidden else {key}
        for child in set(followers[key]):
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if len(result) != len(nodes):
        raise ValueError("Cannot project an FSDP view with missing references or cyclic dependencies")
    return result


def project_fsdp_communication_view(workload: WorkloadGraph) -> WorkloadGraph:
    """Return a graph view; source scheduling and all memory observations stay intact.

    Singular display anchors use the communication node when contracting a
    branching buffer chain. Complete dependencies remain on the projected L0
    graph, and original anchors remain available on ``captured_workload``.
    """
    if workload.captured_workload is not None:
        return workload
    source = {key: node for template in workload.step_templates.values() for key, node in template.nodes.items()}
    hidden = {key for key, node in source.items() if is_fsdp_bookkeeping(node)}
    before = _frontiers(source, hidden)
    after = _frontiers(source, hidden, reverse=True)

    def expand(values, frontiers):
        return sorted(set().union(*(frontiers[value] for value in values))) if values else []

    def anchor(value, frontiers, fallback=0):
        if not value or value not in hidden:
            return value
        choices = frontiers[value]
        return next(iter(choices)) if len(choices) == 1 else fallback

    def boundaries(item, fallback=0):
        return replace(
            item,
            src_exit_op=anchor(item.src_exit_op, before, fallback),
            dst_entry_op=anchor(item.dst_entry_op, after, fallback),
        )

    nodes = {}
    for key, node in source.items():
        if key in hidden:
            continue
        annotations = dict(node.annotations)
        for name in ("stream_predecessors", "event_predecessors"):
            if name in annotations:
                annotations[name] = expand(annotations[name], before)
        nodes[key] = replace(
            node,
            predecessors=expand(node.predecessors, before),
            successors=[],
            attrs=dict(node.attrs),
            annotations=annotations,
        )
    for key, node in nodes.items():
        for parent in node.predecessors:
            nodes[parent].successors.append(key)
    templates = {
        key: replace(
            template,
            nodes={op_id: nodes[op_id] for op_id in template.nodes if op_id in nodes},
            entry_nodes=[],
            exit_nodes=[],
            annotations=dict(template.annotations),
        )
        for key, template in workload.step_templates.items()
    }

    def action_view(action: ScheduleAction) -> ScheduleAction:
        return replace(
            action,
            comm=boundaries(action.comm, action.comm_op_id) if action.comm is not None else None,
            sub_actions=[action_view(child) for child in action.sub_actions]
            if action.sub_actions is not None
            else None,
            annotations=dict(action.annotations),
        )

    annotations = dict(workload.iteration.schedule.annotations)
    annotations["native_dispatch_op_ids"] = [key for key in annotations["native_dispatch_op_ids"] if key not in hidden]
    annotations["communication_events"] = [
        {
            **event,
            "src_exit_op": anchor(event["src_exit_op"], before, event["op_id"]),
            "dst_entry_op": anchor(event["dst_entry_op"], after, event["op_id"]),
        }
        for event in annotations["communication_events"]
    ]
    annotations["fsdp_communication_view"] = {
        "hidden_op_ids": sorted(hidden),
        "num_hidden_ops": len(hidden),
        "hidden_raw_op_counts": dict(Counter(source[key].annotations["raw_op_type"] for key in hidden)),
        "timing_assumption": "FSDP packing, copying and casting have no modeled runtime in this view",
    }
    annotations["cost_unknown_counts"] = dict(
        Counter(node.annotations["raw_op_type"] for node in nodes.values() if node.annotations.get("cost_unknown"))
    )
    plan = workload.schedule_plan
    if plan is None:
        raise ValueError("An FSDP communication view requires a captured SchedulePlan")
    # Communication descriptors are the authoritative fallback for branched
    # slot anchors; they retain the same primitive, shape, bytes and ranks.
    slot_communication = {
        action.comm.slot_id: action.comm_op_id
        for action in plan.action_map.values()
        if action.comm is not None and action.comm.slot_id
    }
    projected_plan = replace(
        plan,
        step_templates=templates,
        actions=[action_view(action) for action in plan.actions],
        data_slots={key: boundaries(slot, slot_communication.get(key, 0)) for key, slot in plan.data_slots.items()},
        annotations=dict(plan.annotations),
    )

    def data_pass_view(data_pass):
        return replace(data_pass, slots=[boundaries(slot) for slot in data_pass.slots])

    schedule = replace(
        workload.iteration.schedule,
        step_templates=templates,
        data_passes=[data_pass_view(data_pass) for data_pass in workload.iteration.schedule.data_passes],
        annotations=annotations,
    )
    return replace(
        workload,
        step_templates=templates,
        iteration=replace(workload.iteration, schedule=schedule),
        schedule_plan=projected_plan,
        cross_iter_passes=[data_pass_view(data_pass) for data_pass in workload.cross_iter_passes],
        captured_workload=workload,
    )
