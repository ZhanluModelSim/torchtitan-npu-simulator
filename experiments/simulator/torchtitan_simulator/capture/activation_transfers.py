# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model activation copies on the consumer graph without changing capture facts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from torchtitan_simulator.capture.fsdp_communication_view import _frontiers, is_fsdp_bookkeeping
from torchtitan_simulator.capture.op_mapping import communication_primitive
from torchtitan_simulator.ir.op_node import OpNode
from torchtitan_simulator.memory.layer_scope import activation_layer_scope

if TYPE_CHECKING:
    from torchtitan_simulator.ir.workload_graph import WorkloadGraph
    from torchtitan_simulator.memory.records import MemoryPlan


@dataclass(frozen=True)
class ExecutionBlock:
    """An observed compute block; boundaries are DAG frontiers, not serial edges."""

    layer: str
    node_ids: tuple[int, ...]
    start_seq: int
    end_seq: int


@dataclass(frozen=True)
class ActivationTransferPlan:
    layer: str
    storage_bytes: tuple[tuple[str, int], ...]
    producer_ids: tuple[int, ...]
    ready_seq: int
    forward_release_seq: int
    forward: ExecutionBlock
    forward_window: ExecutionBlock | None
    forward_wait: ExecutionBlock | None
    backward: ExecutionBlock
    backward_window: ExecutionBlock | None
    backward_start_seq: int


def _blocks(nodes: dict[int, OpNode], phase: str) -> list[ExecutionBlock]:
    groups: list[tuple[str, list[OpNode]]] = []
    for node in sorted(nodes.values(), key=lambda item: item.seq_idx):
        expected_comp = "F" if phase == "forward" else "B"
        if node.annotations.get("comp_type") != expected_comp or is_fsdp_bookkeeping(node):
            continue
        raw = node.annotations.get("raw_op_type", "")
        if node.op_type == "stream_wait" or communication_primitive(raw) is not None:
            continue
        path = node.annotations.get("module_path", "")
        if not path:
            continue
        layer = activation_layer_scope(path)
        # Unattributed autograd bridges belong to the surrounding non-layer block.
        if not groups or groups[-1][0] != layer:
            groups.append((layer, []))
        groups[-1][1].append(node)
    return [
        ExecutionBlock(layer, tuple(n.op_id for n in members), members[0].seq_idx, members[-1].seq_idx)
        for layer, members in groups
    ]


def build_activation_transfer_plan(workload: WorkloadGraph) -> list[ActivationTransferPlan]:
    memory: MemoryPlan = workload.iteration.schedule.annotations["memory_plan"]
    if not memory.offload_ac_saved_tensors:
        return []
    source = workload.captured_workload or workload
    nodes = {key: node for step in source.step_templates.values() for key, node in step.nodes.items()}
    forward = _blocks(nodes, "forward")
    backward = _blocks(nodes, "backward")
    lifetimes = {item.tensor_id: item for item in memory.tensor_lifetimes}
    payloads: dict[str, dict[str, int]] = {}
    producers: dict[str, set[int]] = {}
    for record in memory._activation_prefetch_records():
        scope = record.checkpoint_id.partition(":")[2] or record.checkpoint_id
        layer = activation_layer_scope(scope)
        if not layer:
            raise ValueError(f"Activation offload candidate has no layer: {record.checkpoint_id}")
        lifetime = lifetimes[record.tensor_id]
        if lifetime.num_bytes == 0:
            continue
        storage = lifetime.storage_key or record.tensor_id
        sizes = payloads.setdefault(layer, {})
        sizes[storage] = max(sizes.get(storage, 0), lifetime.num_bytes)
        if lifetime.producer_op > 0 and lifetime.producer_op not in nodes:
            raise ValueError(f"Activation payload producer is absent from the captured graph: {lifetime.producer_op}")
        if lifetime.producer_op in nodes:
            producers.setdefault(layer, set()).add(lifetime.producer_op)
    plans = []
    for layer, sizes in payloads.items():
        if not sum(sizes.values()):
            continue
        fmatches = [i for i, block in enumerate(forward) if block.layer == layer]
        bmatches = [i for i, block in enumerate(backward) if block.layer == layer]
        if not fmatches or not bmatches:
            raise ValueError(f"Activation offload requires forward and backward boundaries for {layer}")
        fi, bi = fmatches[0], bmatches[0]
        if len(fmatches) != 1 or len(bmatches) != 1:
            raise ValueError(f"Repeated layer invocation is not supported for activation offload: {layer}")
        fblock, bblock = forward[fi], backward[bi]
        ready_seq = max([fblock.end_seq, *(nodes[key].seq_idx for key in producers.get(layer, ()))])
        for event in memory.raw_events:
            if event.comp_type == "F" and event.seq_idx > ready_seq and sizes.keys() & set(event.mutated_storage_keys):
                raise ValueError(f"Activation offload storage is mutated after its layer copy: {layer}")
        fend = fmatches[-1]
        window_index = next((i for i in range(fend + 1, len(forward)) if forward[i].end_seq >= ready_seq), None)
        window = forward[window_index] if window_index is not None else None
        wait = forward[window_index + 1] if window_index is not None and window_index + 1 < len(forward) else None
        phase_end = max(node.seq_idx for node in nodes.values() if node.annotations.get("comp_type") == "F")
        if ready_seq > fblock.end_seq:
            # Late unscoped head producers cannot meet the original neighbor
            # window. Drain at the F barrier and retain their buffers to F end.
            window = wait = None
        plans.append(
            ActivationTransferPlan(
                layer,
                tuple(sorted(sizes.items())),
                tuple(sorted(producers.get(layer, ()))),
                ready_seq,
                window.end_seq if window is not None else phase_end,
                fblock,
                window,
                wait,
                bblock,
                backward[bi - 1] if bi else None,
                backward[bi - 1].start_seq
                if bi
                else min(node.seq_idx for node in nodes.values() if node.annotations.get("comp_type") == "B"),
            )
        )
    return sorted(plans, key=lambda plan: plan.forward.start_seq)


def insert_activation_transfers(workload: WorkloadGraph) -> WorkloadGraph:
    """Insert modeled copies after FSDP projection; phase barriers drain copies."""
    plans = build_activation_transfer_plan(workload)
    if not plans:
        return workload
    annotations = workload.iteration.schedule.annotations
    if "activation_transfer_plan" in annotations:
        raise ValueError("Activation transfers have already been inserted")
    source = workload.captured_workload or workload
    raw = {key: node for step in source.step_templates.values() for key, node in step.nodes.items()}
    visible = {key: node for step in workload.step_templates.values() for key, node in step.nodes.items()}
    hidden = raw.keys() - visible.keys()
    before = _frontiers(raw, hidden)
    after = _frontiers(raw, hidden, reverse=True)
    nodes = {
        key: replace(
            node,
            predecessors=list(node.predecessors),
            successors=[],
            attrs=dict(node.attrs),
            annotations=dict(node.annotations),
        )
        for key, node in visible.items()
    }
    templates = {key: dict(step.nodes) for key, step in workload.step_templates.items()}
    owner = {node_id: key for key, step in workload.step_templates.items() for node_id in step.nodes}
    stream_base = max((node.annotations.get("stream_id", 0) for node in raw.values()), default=0) + 1
    next_id = max(raw) + 1
    previous_copy: dict[str, int] = {}
    modeled_ids: list[int] = []
    boundaries: dict[tuple[tuple[int, ...], bool], set[int]] = {}

    def boundary(block: ExecutionBlock, *, entry: bool) -> set[int]:
        cache_key = (block.node_ids, entry)
        if cache_key not in boundaries:
            selected = set(block.node_ids)
            links = "predecessors" if entry else "successors"
            # A -> collective/wait -> B still makes A an ancestor of B.
            # Contract all nonselected nodes before identifying block frontiers.
            contracted = _frontiers(raw, raw.keys() - selected, reverse=not entry)
            frontier = {key for key in selected if not any(contracted[parent] for parent in getattr(raw[key], links))}
            mapping = after if entry else before
            boundaries[cache_key] = set().union(*(mapping[key] for key in frontier))
        return boundaries[cache_key]

    def add_node(
        plan: ActivationTransferPlan, direction: str, parents: set[int], block: ExecutionBlock, *, wait: bool = False
    ) -> int:
        nonlocal next_id
        op_id = next_id
        next_id += 1
        name = "stream_wait" if wait else f"activation_{direction}"
        phase = "forward" if direction == "d2h" else "backward"
        anchor = next(iter(boundary(block, entry=True)))
        key = owner[anchor]
        node = OpNode(
            op_id,
            name,
            [],
            [],
            {}
            if wait
            else {
                "activation_transfer": {
                    "direction": direction,
                    "bytes": sum(size for _, size in plan.storage_bytes),
                    "layer": plan.layer,
                }
            },
            sorted(parents),
            [],
            annotations={
                "raw_op_type": f"simulator.{name}",
                "module_path": plan.layer,
                "phase": phase,
                "execution_kind": phase,
                "comp_type": "F" if direction == "d2h" else "B",
                "stream_id": stream_base + (direction == "h2d") + (2 if wait else 0),
                "modeling_source": "activation_offload_neighbor_v1",
            },
            seq_idx=block.start_seq if wait or direction == "h2d" else block.end_seq,
        )
        nodes[op_id] = node
        templates[key][op_id] = node
        owner[op_id] = key
        modeled_ids.append(op_id)
        return op_id

    for direction in ("d2h", "h2d"):
        ordered = sorted(plans, key=lambda item: item.ready_seq if direction == "d2h" else item.backward.start_seq)
        for plan in ordered:
            block = plan.forward if direction == "d2h" else plan.backward
            window = plan.backward_window
            if direction == "d2h":
                parents = boundary(block, entry=False).copy()
                for producer in plan.producer_ids:
                    parents.update(before[producer])
            elif window is None:
                parents = set()
            else:
                entries = boundary(window, entry=True)
                parents = {
                    parent
                    for entry in entries
                    for parent in visible[entry].predecessors
                    if owner[parent] == owner[entry]
                }
            if direction in previous_copy:
                parents.add(previous_copy[direction])
            copy_id = add_node(plan, direction, parents, block)
            previous_copy[direction] = copy_id
            target = plan.forward_wait if direction == "d2h" else plan.backward
            if target is not None:
                wait_id = add_node(plan, direction, {copy_id}, target, wait=True)
                for entry in boundary(target, entry=True):
                    nodes[entry].predecessors = sorted(set(nodes[entry].predecessors) | {wait_id})
    for node in nodes.values():
        for parent in node.predecessors:
            nodes[parent].successors.append(node.op_id)
    steps = {
        key: replace(
            step,
            nodes={node_id: nodes[node_id] for node_id in templates[key]},
            entry_nodes=[],
            exit_nodes=[],
            annotations=dict(step.annotations),
        )
        for key, step in workload.step_templates.items()
    }
    from torchtitan_simulator.memory.activation_transfer_budget import apply_activation_transfer_budget

    updated = dict(annotations)
    updated["memory_plan"] = apply_activation_transfer_budget(annotations["memory_plan"], plans)
    updated["memory_summary"] = updated["memory_plan"].to_summary_dict()
    updated["activation_transfer_plan"] = {
        "policy": "neighbor_v1_phase_barrier",
        "modeled_op_ids": modeled_ids,
        "layers": [plan.layer for plan in plans],
        "phase_drain_layers": [plan.layer for plan in plans if plan.ready_seq > plan.forward.end_seq],
    }
    logger = logging.getLogger(__name__)
    logger.info(
        "Activation offload: %d transfers, %d bytes per direction; independent F/B phase barriers",
        2 * len(plans),
        sum(size for plan in plans for _, size in plan.storage_bytes),
    )
    late_layers = updated["activation_transfer_plan"]["phase_drain_layers"]
    if late_layers:
        logger.warning(
            "Activation offload late_payload_phase_drain for %s: whole-layer copies wait for late "
            "producers and drain at F end; conservative overlap and potentially more than two resident layers",
            ", ".join(late_layers),
        )
    if workload.schedule_plan is None:
        raise ValueError("Activation transfers require a SchedulePlan")
    return replace(
        workload,
        step_templates=steps,
        schedule_plan=replace(workload.schedule_plan, step_templates=steps),
        iteration=replace(
            workload.iteration, schedule=replace(workload.iteration.schedule, step_templates=steps, annotations=updated)
        ),
    )
