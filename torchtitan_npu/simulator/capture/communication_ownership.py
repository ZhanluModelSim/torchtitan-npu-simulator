# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Normalize communication ownership before L1 templates are exported.

The pipeline runtime may execute stage-local communication as standalone
schedule actions.  Capture therefore sees fragments such as ``s0_UNSHARD``
even though the downstream simulator treats one F/B/I/W StepGraph as the
complete stage execution unit.  This module resolves that abstraction gap:

* PP point-to-point operators remain L2-owned and are extracted from compute
  templates into immutable communication fragments.
* FSDP collectives observed inside F/B/I/W become part of immutable L1
  template variants.
* Explicit FSDP prefetch actions remain in L2 because their issue position is
  outside the owning compute chunk.
* Gradient collectives already captured by a compute template are not emitted
  again as L2 ``REDUCE_GRAD`` actions.

The plugins consume semantic capture records only.  They do not inspect the
pipeline schedule class or branch on schedule names.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.schedule_assemblers import ActionSpec
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph


_COMPUTE_TYPES = {"F", "B", "I", "W", "F_RECOMPUTE"}
_STAGE_TEMPLATE_TYPES = _COMPUTE_TYPES | {"OPTIMIZER"}
_FSDP_PREFETCH_LAUNCH_OP = "FSDP_PREFETCH_LAUNCH"
_FSDP_POST_BACKWARD_SYNC_OP = "FSDP_POST_BACKWARD_SYNC"
_P2P_ACTION_BY_DIRECTION = {
    "forward_send": "SEND_F",
    "forward_recv": "RECV_F",
    "backward_send": "SEND_B",
    "backward_recv": "RECV_B",
}


@dataclass(slots=True)
class CommunicationOwnershipResult:
    specs: list[ActionSpec]
    internal_fsdp_transitions: set[str]
    external_fsdp_transitions: set[str]
    generated_templates: list[str]
    internal_gradient_reductions: int
    external_gradient_reductions: int
    removed_noop_gradient_intents: int
    stage_owned_collectives: int


class CommunicationOwnershipPlugin(Protocol):
    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        ...


def _iter_specs(specs: Iterable[ActionSpec]) -> Iterable[ActionSpec]:
    for spec in specs:
        yield spec
        if spec.sub_actions:
            yield from _iter_specs(spec.sub_actions)


def _clone_node(
    node: OpNode,
    *,
    op_id: int | None = None,
    predecessors: list[int] | None = None,
    annotations: dict | None = None,
) -> OpNode:
    return replace(
        node,
        op_id=node.op_id if op_id is None else op_id,
        inputs=list(node.inputs),
        outputs=list(node.outputs),
        attrs=dict(node.attrs),
        predecessors=(
            list(node.predecessors)
            if predecessors is None
            else list(predecessors)
        ),
        successors=[],
        annotations=(
            dict(node.annotations)
            if annotations is None
            else dict(annotations)
        ),
    )


def _rebuild_successors(nodes: dict[int, OpNode]) -> None:
    for node in nodes.values():
        node.successors = []
    for op_id, node in nodes.items():
        for predecessor in node.predecessors:
            if predecessor in nodes:
                nodes[predecessor].successors.append(op_id)


def _refresh_graph_topology(graph: StepGraph) -> None:
    _rebuild_successors(graph.nodes)
    graph.entry_nodes = [
        op_id
        for op_id, node in graph.nodes.items()
        if not any(predecessor in graph.nodes for predecessor in node.predecessors)
    ]
    graph.exit_nodes = [
        op_id
        for op_id, node in graph.nodes.items()
        if not any(successor in graph.nodes for successor in node.successors)
    ]
    in_degree = {
        op_id: sum(
            predecessor in graph.nodes for predecessor in node.predecessors
        )
        for op_id, node in graph.nodes.items()
    }
    ready = [
        op_id for op_id, degree in in_degree.items() if degree == 0
    ]
    visited = 0
    while ready:
        op_id = ready.pop()
        visited += 1
        for successor in graph.nodes[op_id].successors:
            if successor not in in_degree:
                continue
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                ready.append(successor)
    graph.is_acyclic = visited == len(graph.nodes)
    graph.comm_volume = sum(
        node.comm_bytes * int(node.annotations.get("repeat_count", 1))
        for node in graph.nodes.values()
    )


def _has_path(graph: StepGraph, start_op_id: int, target_op_id: int) -> bool:
    """Return whether adding ``target -> start`` would create a cycle."""
    pending = [start_op_id]
    visited: set[int] = set()
    while pending:
        op_id = pending.pop()
        if op_id == target_op_id:
            return True
        if op_id in visited:
            continue
        visited.add(op_id)
        node = graph.nodes.get(op_id)
        if node is not None:
            pending.extend(node.successors)
    return False


def _copy_without_nodes(
    template: StepGraph,
    removed_op_ids: set[int],
    *,
    annotations: dict | None = None,
) -> StepGraph:
    nodes = {
        op_id: _clone_node(
            node,
            predecessors=[
                predecessor
                for predecessor in node.predecessors
                if predecessor not in removed_op_ids
            ],
        )
        for op_id, node in template.nodes.items()
        if op_id not in removed_op_ids
    }
    _rebuild_successors(nodes)
    return StepGraph(
        step_id=template.step_id,
        step_type=template.step_type,
        nodes=nodes,
        tensor_lifetimes=dict(template.tensor_lifetimes),
        total_flops=template.total_flops,
        peak_active_mem=template.peak_active_mem,
        param_mem=template.param_mem,
        comm_volume=template.comm_volume,
        device_placement=dict(template.device_placement),
        annotations=(
            dict(template.annotations)
            if annotations is None
            else dict(annotations)
        ),
        fused_regions=list(template.fused_regions),
        internal_data_passes=list(template.internal_data_passes),
    )


def _find_node(
    step_templates: dict[str, StepGraph],
    op_id: int,
) -> tuple[str, OpNode] | None:
    if not op_id:
        return None
    for template_id, template in step_templates.items():
        node = template.nodes.get(op_id)
        if node is not None:
            return template_id, node
    return None


def _find_comm_node(
    step_templates: dict[str, StepGraph],
    event: CommEvent,
) -> tuple[str, OpNode] | None:
    if not event.op_id:
        return None
    found = _find_node(step_templates, event.op_id)
    if found is None:
        return None
    _template_id, node = found
    if node.annotations.get("raw_op_type") != f"comm.{event.comm_primitive}":
        return None
    return found


class PipelineP2POwnershipPlugin:
    """Extract cross-stage P2P operators from stage compute templates."""

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        located: dict[int, tuple[str, OpNode, CommEvent]] = {}
        for event in comm_events:
            if event.p2p_direction not in _P2P_ACTION_BY_DIRECTION:
                continue
            found = _find_comm_node(step_templates, event)
            if found is None:
                continue
            template_id, node = found
            located[event.op_id] = (template_id, node, event)
        if not located:
            return specs

        removed = set(located)
        for template_id, template in list(step_templates.items()):
            if template.step_type not in _COMPUTE_TYPES:
                continue
            if not removed.intersection(template.nodes):
                continue
            annotations = dict(template.annotations)
            annotations["communication_ownership_normalized"] = True
            step_templates[template_id] = _copy_without_nodes(
                template,
                removed,
                annotations=annotations,
            )

        fragment_nodes: dict[str, dict[int, OpNode]] = defaultdict(dict)
        for op_id, (_source_template, node, event) in located.items():
            action_type = _P2P_ACTION_BY_DIRECTION[event.p2p_direction]
            stage = int(event.p2p_stage)
            fragment_id = f"s{stage}_PP_{action_type}"
            annotations = dict(node.annotations)
            annotations.update({
                "communication_owner": "L2_PIPELINE",
                "source_op_id": op_id,
                "pp_action_type": action_type,
            })
            fragment_nodes[fragment_id][op_id] = _clone_node(
                node,
                predecessors=[],
                annotations=annotations,
            )

        for fragment_id, nodes in fragment_nodes.items():
            _rebuild_successors(nodes)
            step_templates[fragment_id] = StepGraph(
                step_id=fragment_id,
                step_type=fragment_id.split("_PP_", 1)[1],
                nodes=nodes,
                annotations={
                    "template_kind": "l2_communication_fragment",
                    "communication_owner": "L2_PIPELINE",
                },
            )
        return specs


@dataclass(frozen=True, slots=True)
class _OwnedCollective:
    spec: ActionSpec
    event: CommEvent
    source_template: str
    source_node: OpNode
    owner_compute_instance_id: str

    @property
    def signature(self) -> tuple:
        return (
            str(self.spec.annotations.get("fsdp_group_id", "")),
            str(self.spec.annotations.get("fsdp_module_fqn", "")),
            str(self.spec.annotations.get("fsdp_prefetch_source_fqn", "")),
            str(self.spec.annotations.get("fsdp_prefetch_type", "")),
            self.event.comm_primitive,
            tuple(self.event.tensor_shape),
            self.event.dtype,
            int(self.event.world_size),
            tuple(tuple(group) for group in self.event.comm_ranks),
            self.owner_compute_instance_id
            != str(
                self.spec.annotations.get(
                    "parent_compute_instance_id",
                    "",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class _FSDPGroupRegion:
    group_id: str
    module_fqn: str
    wait_seq_idx: int
    release_seq_idx: int
    entry_op_ids: tuple[int, ...]
    exit_op_ids: tuple[int, ...]
    external_predecessors: tuple[int, ...]
    param_group_index: int = -1
    num_param_groups: int = -1


@dataclass(frozen=True, slots=True)
class _FSDPPrefetchAnchor:
    predecessor_op_ids: tuple[int, ...]
    source_entry_op_ids: tuple[int, ...]
    seq_idx: int
    filtered_post_backward_allreduce_op_ids: tuple[int, ...] = ()


def _fsdp_region_for_collective(
    *,
    regions_by_group: dict[str, list[_FSDPGroupRegion]],
    wait_seq_idxs_by_group: dict[str, tuple[int, ...]],
    group_id: str,
    collective_seq_idx: int,
) -> _FSDPGroupRegion | None:
    """Return the first parameter residency region following an all-gather.

    An FSDP group can be entered more than once within a single captured
    template (for example, ``tok_embeddings`` is reused by an MTP module).
    Resolving that group solely by its id loses this occurrence information and
    can attach a later invocation's external predecessors to an earlier
    all-gather.  The L0 sequence number gives the missing occurrence key:
    depending on the capture boundary, an all-gather is either inside its
    matching region or immediately before its ``unshard_wait`` marker.

    The sequence arrays are built once per template variant, so this lookup is
    O(log R) for R lifetimes of one group rather than a graph traversal.
    """
    regions = regions_by_group.get(group_id)
    if not regions:
        return None
    wait_seq_idxs = wait_seq_idxs_by_group[group_id]
    index = bisect_left(wait_seq_idxs, collective_seq_idx)
    if index:
        preceding = regions[index - 1]
        if collective_seq_idx < preceding.release_seq_idx:
            return preceding
    if index == len(regions):
        return None
    return regions[index]


def _fsdp_prefetch_source_regions(
    *,
    source_regions: list[_FSDPGroupRegion],
    source_wait_seq_idxs: tuple[int, ...],
    source_invocation_starts: tuple[int, ...],
    target_region: _FSDPGroupRegion | None,
    target_collective_seq_idx: int,
) -> list[_FSDPGroupRegion]:
    """Select the source module invocation immediately before a prefetch.

    A module can have multiple parameter groups, which must stay together, but
    it can also be invoked multiple times in one template.  Restrict the
    anchor to the latest invocation that precedes the target residency region;
    otherwise a later reuse contributes backward-in-time predecessors.
    """
    if not source_regions:
        return []
    cutoff = (
        target_region.wait_seq_idx
        if target_region is not None
        else target_collective_seq_idx
    )
    end = bisect_left(source_wait_seq_idxs, cutoff)
    if end == 0:
        return []

    start_index = bisect_left(source_invocation_starts, end) - 1
    start = (
        source_invocation_starts[start_index]
        if start_index >= 0
        else end - 1
    )
    next_start = (
        source_invocation_starts[start_index + 1]
        if start_index + 1 < len(source_invocation_starts)
        else len(source_regions)
    )
    return source_regions[start : min(end, next_start)]


def _is_hsdp_post_backward_allreduce(node: OpNode | None) -> bool:
    """Identify the data-replica all-reduce issued after an HSDP RS.

    This is intentionally narrower than matching every all-reduce: tensor- or
    expert-parallel all-reduces may carry a real compute dependency, while the
    HSDP data-replica result is only consumed by later gradient handling.
    """
    if node is None or node.annotations.get("raw_op_type") != "comm.allreduce":
        return False
    parallel_dim = str(
        node.annotations.get("comm_dim")
        or node.annotations.get("group_name")
        or ""
    )
    return parallel_dim == "dp_replicate"


def _fsdp_prefetch_anchor(
    *,
    template_id: str,
    target_group_id: str,
    target_module_fqn: str,
    target_region: _FSDPGroupRegion | None,
    target_collective_seq_idx: int,
    prefetch_source_fqn: str,
    regions_by_module: dict[str, list[_FSDPGroupRegion]],
    wait_seq_idxs_by_module: dict[str, tuple[int, ...]],
    invocation_starts_by_module: dict[str, tuple[int, ...]],
    comm_id_by_region: dict[_FSDPGroupRegion, int],
    nodes_by_id: Mapping[int, OpNode],
) -> _FSDPPrefetchAnchor:
    source_regions = _fsdp_prefetch_source_regions(
        source_regions=regions_by_module.get(prefetch_source_fqn, []),
        source_wait_seq_idxs=wait_seq_idxs_by_module.get(
            prefetch_source_fqn,
            (),
        ),
        source_invocation_starts=invocation_starts_by_module.get(
            prefetch_source_fqn,
            (),
        ),
        target_region=target_region,
        target_collective_seq_idx=target_collective_seq_idx,
    )
    if not source_regions:
        raise RuntimeError(
            "FSDP prefetch has no source parameter-group compute region: "
            f"template={template_id}, target_group={target_group_id}, "
            f"target_module={target_module_fqn!r}, "
            f"source_module={prefetch_source_fqn!r}"
        )

    external_predecessors = list(
        dict.fromkeys(
            predecessor
            for region in source_regions
            for predecessor in region.external_predecessors
        )
    )
    # HSDP's post-backward all-reduce makes the reduced gradient available to
    # the optimizer, not to the next module's backward compute.  It can be an
    # external predecessor of the source FSDP region in the captured graph,
    # but must not turn into a prefetch-launch dependency: doing so serializes
    # the next all-gather and compute behind the all-reduce.
    filtered_post_backward_allreduce_op_ids = tuple(
        predecessor
        for predecessor in external_predecessors
        if _is_hsdp_post_backward_allreduce(nodes_by_id.get(predecessor))
    )
    filtered_post_backward_allreduce_op_id_set = set(
        filtered_post_backward_allreduce_op_ids
    )
    predecessor_op_ids = [
        predecessor
        for predecessor in external_predecessors
        if predecessor not in filtered_post_backward_allreduce_op_id_set
    ]
    predecessor_op_ids.extend(
        comm_id
        for region in source_regions
        if (comm_id := comm_id_by_region.get(region)) is not None
    )
    source_entry_op_ids = tuple(
        dict.fromkeys(
            entry
            for region in source_regions
            for entry in region.entry_op_ids
        )
    )
    return _FSDPPrefetchAnchor(
        predecessor_op_ids=tuple(dict.fromkeys(predecessor_op_ids)),
        source_entry_op_ids=source_entry_op_ids,
        seq_idx=min(region.wait_seq_idx for region in source_regions),
        filtered_post_backward_allreduce_op_ids=(
            filtered_post_backward_allreduce_op_ids
        ),
    )


def _fsdp_group_regions(
    template: StepGraph,
    removed_op_ids: set[int],
) -> list[_FSDPGroupRegion]:
    waits = sorted(
        (
            node
            for node in template.nodes.values()
            if node.annotations.get("fsdp_marker") == "unshard_wait"
        ),
        key=lambda node: (node.seq_idx, node.op_id),
    )
    releases_by_group: dict[str, list[OpNode]] = defaultdict(list)
    for node in template.nodes.values():
        if node.annotations.get("fsdp_marker") != "reshard_release":
            continue
        releases_by_group[
            str(node.annotations.get("fsdp_group_id", ""))
        ].append(node)
    for releases in releases_by_group.values():
        releases.sort(key=lambda node: (node.seq_idx, node.op_id))

    regions: list[_FSDPGroupRegion] = []
    for wait_index, wait in enumerate(waits):
        group_id = str(wait.annotations.get("fsdp_group_id", ""))
        if not group_id:
            continue
        release = next(
            (
                candidate
                for candidate in releases_by_group.get(group_id, ())
                if candidate.seq_idx > wait.seq_idx
            ),
            None,
        )
        next_wait_seq = (
            waits[wait_index + 1].seq_idx
            if wait_index + 1 < len(waits)
            else max(
                (node.seq_idx for node in template.nodes.values()),
                default=wait.seq_idx,
            )
            + 1
        )
        release_seq_idx = (
            release.seq_idx if release is not None else next_wait_seq
        )
        body = {
            op_id
            for op_id, node in template.nodes.items()
            if op_id not in removed_op_ids
            and wait.seq_idx < node.seq_idx < release_seq_idx
        }
        if not body:
            continue
        entries = tuple(
            sorted(
                (
                    op_id
                    for op_id in body
                    if not any(
                        predecessor in body
                        for predecessor in template.nodes[
                            op_id
                        ].predecessors
                    )
                ),
                key=lambda op_id: (
                    template.nodes[op_id].seq_idx,
                    op_id,
                ),
            )
        )
        exits = tuple(
            sorted(
                (
                    op_id
                    for op_id in body
                    if not any(
                        successor in body
                        for successor in template.nodes[op_id].successors
                    )
                ),
                key=lambda op_id: (
                    template.nodes[op_id].seq_idx,
                    op_id,
                ),
            )
        )
        external_predecessors = tuple(
            dict.fromkeys(
                predecessor
                for entry in entries
                for predecessor in template.nodes[entry].predecessors
                if predecessor not in body
                and predecessor not in removed_op_ids
                and predecessor in template.nodes
            )
        )
        regions.append(
            _FSDPGroupRegion(
                group_id=group_id,
                module_fqn=str(
                    wait.annotations.get("fsdp_module_fqn", "")
                ),
                wait_seq_idx=wait.seq_idx,
                release_seq_idx=release_seq_idx,
                entry_op_ids=entries,
                exit_op_ids=exits,
                external_predecessors=external_predecessors,
                param_group_index=int(
                    wait.annotations.get("fsdp_param_group_index", -1)
                ),
                num_param_groups=int(
                    wait.annotations.get("fsdp_num_param_groups", -1)
                ),
            )
        )
    return regions


def _add_fsdp_backward_reduction_syncs(
    graph: StepGraph,
    regions: list[_FSDPGroupRegion],
    next_synthetic_id: int,
) -> tuple[int, int]:
    """Model FSDP2's one-module reduce-scatter backpressure."""
    if graph.step_type not in {"B", "W"}:
        return next_synthetic_id, 0

    ordered_regions = sorted(
        regions,
        key=lambda region: (region.wait_seq_idx, region.group_id),
    )
    module_blocks: list[list[_FSDPGroupRegion]] = []
    for region in ordered_regions:
        if (
            not module_blocks
            or module_blocks[-1][0].module_fqn != region.module_fqn
        ):
            module_blocks.append([region])
        else:
            module_blocks[-1].append(region)
    if len(module_blocks) < 2:
        return next_synthetic_id, 0

    barrier_regions: list[_FSDPGroupRegion] = []
    for block in module_blocks:
        explicit_barriers = [
            region
            for region in block
            if region.num_param_groups > 0
            and region.param_group_index == region.num_param_groups - 1
        ]
        barrier_regions.append(
            min(
                explicit_barriers or block,
                key=lambda region: (
                    region.release_seq_idx,
                    -region.wait_seq_idx,
                    region.group_id,
                ),
            )
        )

    reduce_scatters = sorted(
        (
            node
            for node in graph.nodes.values()
            if node.annotations.get("raw_op_type")
            == "comm.reduce_scatter"
        ),
        key=lambda node: (node.seq_idx, node.op_id),
    )
    reductions_by_block: list[list[OpNode]] = []
    for block_index, block in enumerate(module_blocks):
        start_seq = block[0].wait_seq_idx
        end_seq = (
            module_blocks[block_index + 1][0].wait_seq_idx
            if block_index + 1 < len(module_blocks)
            else float("inf")
        )
        reductions_by_block.append(
            [
                node
                for node in reduce_scatters
                if start_seq < node.seq_idx < end_seq
            ]
        )

    sync_count = 0
    for block_index in range(1, len(module_blocks)):
        prior_reductions = reductions_by_block[block_index - 1]
        if not prior_reductions:
            continue
        barrier_region = barrier_regions[block_index]
        predecessor_ids = list(
            dict.fromkeys(
                [
                    *barrier_region.exit_op_ids,
                    *(node.op_id for node in prior_reductions),
                ]
            )
        )
        if not predecessor_ids:
            continue

        sync_id = next_synthetic_id
        next_synthetic_id -= 1
        prior_module_fqn = module_blocks[block_index - 1][0].module_fqn
        graph.nodes[sync_id] = OpNode(
            op_id=sync_id,
            op_type=_FSDP_POST_BACKWARD_SYNC_OP,
            inputs=[],
            outputs=[],
            attrs={},
            predecessors=predecessor_ids,
            successors=[],
            flops=0,
            peak_mem=0,
            param_mem=0,
            comm_bytes=0,
            annotations={
                "raw_op_type": _FSDP_POST_BACKWARD_SYNC_OP,
                "control_op": True,
                "zero_cost": True,
                "fsdp_module_fqn": barrier_region.module_fqn,
                "fsdp_prior_module_fqn": prior_module_fqn,
                "fsdp_barrier_group_id": barrier_region.group_id,
                "fsdp_waited_reduce_scatter_op_ids": [
                    node.op_id for node in prior_reductions
                ],
                "fsdp_sync_scope": "prior_module_reduce_scatter",
            },
            seq_idx=barrier_region.release_seq_idx,
        )

        next_barrier_seq = (
            barrier_regions[block_index + 1].release_seq_idx
            if block_index + 1 < len(barrier_regions)
            else float("inf")
        )
        post_barrier_ids = {
            op_id
            for op_id, node in graph.nodes.items()
            if barrier_region.release_seq_idx
            < node.seq_idx
            < next_barrier_seq
        }
        gated_node_ids = {
            op_id
            for op_id in post_barrier_ids
            if not any(
                predecessor in post_barrier_ids
                for predecessor in graph.nodes[op_id].predecessors
            )
        }
        for gated_id in gated_node_ids:
            gated = graph.nodes.get(gated_id)
            if gated is not None and sync_id not in gated.predecessors:
                gated.predecessors.append(sync_id)
        sync_count += 1

    return next_synthetic_id, sync_count


class FSDPStageOwnershipPlugin:
    """Fold compute-local FSDP collectives into immutable L1 variants."""

    def __init__(self) -> None:
        self.internal_transitions: set[str] = set()
        self.external_transitions: set[str] = set()
        self.generated_templates: list[str] = []

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        flat_specs = list(_iter_specs(specs))
        compute_specs = [
            spec
            for spec in flat_specs
            if spec.action_type == "COMPUTE"
            and spec.comp_type in _COMPUTE_TYPES
        ]
        compute_by_instance = {
            str(spec.annotations.get("compute_instance_id", "")): spec
            for spec in compute_specs
            if spec.annotations.get("compute_instance_id")
        }
        compute_by_key: dict[tuple[int, int, str], list[ActionSpec]] = (
            defaultdict(list)
        )
        for compute_spec in compute_specs:
            compute_by_key[
                (
                    compute_spec.stage,
                    compute_spec.mb_idx,
                    compute_spec.comp_type,
                )
            ].append(compute_spec)
        unshards = [
            spec for spec in specs if spec.action_type == "UNSHARD"
        ]

        self.external_transitions = {
            str(spec.annotations.get("fsdp_transition_id", ""))
            for spec in unshards
            if spec.annotations.get("fsdp_schedule_source") == "intent"
            and spec.annotations.get("fsdp_transition_id")
        }
        internal_unshards = [
            spec
            for spec in unshards
            if spec.annotations.get("fsdp_schedule_source") == "state"
            and spec.annotations.get("fsdp_transition_id")
            and spec.annotations.get("parent_compute_instance_id")
        ]
        self.internal_transitions = {
            str(spec.annotations["fsdp_transition_id"])
            for spec in internal_unshards
        }

        if internal_unshards:
            self._build_template_variants(
                step_templates=step_templates,
                compute_specs=compute_specs,
                compute_by_instance=compute_by_instance,
                compute_by_key=compute_by_key,
                internal_unshards=internal_unshards,
                comm_events=comm_events,
            )

        retained: list[ActionSpec] = []
        for spec in specs:
            if spec.action_type not in {"UNSHARD", "RESHARD"}:
                retained.append(spec)
                continue
            transition_id = str(
                spec.annotations.get("fsdp_transition_id", "")
            )
            if transition_id in self.external_transitions:
                spec.annotations["communication_owner"] = "L2_PREFETCH"
                retained.append(spec)
                continue
            if transition_id in self.internal_transitions:
                continue
            if spec.is_noop or spec.annotations.get("fsdp_intent_noop"):
                continue
            retained.append(spec)
        return retained

    def _build_template_variants(
        self,
        *,
        step_templates: dict[str, StepGraph],
        compute_specs: list[ActionSpec],
        compute_by_instance: dict[str, ActionSpec],
        compute_by_key: dict[tuple[int, int, str], list[ActionSpec]],
        internal_unshards: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> None:
        source_nodes: dict[int, tuple[str, OpNode]] = {}
        for template_id, template in step_templates.items():
            for op_id, node in template.nodes.items():
                source_nodes[op_id] = (template_id, node)

        real_events = [
            event
            for event in comm_events
            if event.comm_primitive == "allgather"
            and event.fsdp_transition_id
        ]
        event_by_transition = {
            event.fsdp_transition_id: event for event in real_events
        }
        canonical_by_group_comp: dict[
            tuple[int, str, str], CommEvent
        ] = {}
        canonical_by_group: dict[tuple[int, str], CommEvent] = {}

        def is_valid_allgather(event: CommEvent | None) -> bool:
            return bool(
                event is not None
                and event.op_id
                and event.op_id in source_nodes
                and source_nodes[event.op_id][1].annotations.get(
                    "raw_op_type"
                ) == "comm.allgather"
            )

        for event in real_events:
            if not is_valid_allgather(event):
                continue
            stage = int(event.p2p_stage)
            canonical_by_group_comp.setdefault(
                (stage, event.fsdp_group_id, event.comp_type),
                event,
            )
            canonical_by_group.setdefault(
                (stage, event.fsdp_group_id),
                event,
            )

        owned_by_parent: dict[str, list[_OwnedCollective]] = defaultdict(list)
        for spec in sorted(internal_unshards, key=lambda item: item.order_key):
            transition_id = str(spec.annotations["fsdp_transition_id"])
            semantic_event = event_by_transition.get(transition_id)
            source_event = semantic_event
            group_id = str(spec.annotations.get("fsdp_group_id", ""))
            comp_type = str(spec.annotations.get("residency_comp_type", ""))
            if not is_valid_allgather(source_event):
                source_event = canonical_by_group_comp.get(
                    (spec.stage, group_id, comp_type)
                ) or canonical_by_group.get((spec.stage, group_id))
            shard_world_size = int(
                spec.annotations.get("shard_world_size", -1)
            )
            if not is_valid_allgather(source_event):
                if shard_world_size > 1:
                    raise RuntimeError(
                        "compute-local FSDP transition has no reusable "
                        f"all-gather template: transition={transition_id}, "
                        f"stage={spec.stage}, group={group_id}"
                    )
                continue
            event = semantic_event or source_event
            if not event.fsdp_module_fqn:
                event.fsdp_module_fqn = str(
                    spec.annotations.get("fsdp_module_fqn", "")
                )
            if not event.fsdp_prefetch_source_fqn:
                event.fsdp_prefetch_source_fqn = str(
                    spec.annotations.get("fsdp_prefetch_source_fqn", "")
                )
            if not event.fsdp_prefetch_type:
                event.fsdp_prefetch_type = str(
                    spec.annotations.get("fsdp_prefetch_type", "")
                )
            parent_id = str(
                spec.annotations.get("parent_compute_instance_id", "")
            )
            if parent_id not in compute_by_instance:
                raise RuntimeError(
                    "compute-local FSDP transition references missing "
                    f"compute instance {parent_id!r}"
                )
            parent_compute = compute_by_instance[parent_id]
            if parent_compute.annotations.get("non_pipeline_capture"):
                # Non-PP has no chunk hook to update _pp_context.comp_type.
                # The residency state is authoritative for whether an AG was
                # launched by forward or backward FSDP hooks.
                event = replace(
                    event,
                    comp_type=parent_compute.comp_type,
                    p2p_stage=parent_compute.stage,
                    p2p_mb_idx=parent_compute.mb_idx,
                )
            owner_id = parent_id
            prefetch_source_fqn = (
                event.fsdp_prefetch_source_fqn
                or str(
                    spec.annotations.get(
                        "fsdp_prefetch_source_fqn",
                        "",
                    )
                )
            )
            if prefetch_source_fqn and event.comp_type in _COMPUTE_TYPES:
                launch_candidates = compute_by_key.get(
                    (
                        int(event.p2p_stage),
                        int(event.p2p_mb_idx),
                        event.comp_type,
                    ),
                    [],
                )
                launch_owner = next(
                    (
                        candidate
                        for candidate in launch_candidates
                        if int(
                            candidate.annotations.get(
                                "capture_start_seq",
                                candidate.seq_idx,
                            )
                        )
                        <= event.seq_idx
                        <= int(
                            candidate.annotations.get(
                                "capture_end_seq",
                                candidate.seq_idx,
                            )
                        )
                    ),
                    (
                        launch_candidates[0]
                        if len(launch_candidates) == 1
                        else None
                    ),
                )
                if launch_owner is not None:
                    owner_id = str(
                        launch_owner.annotations.get(
                            "compute_instance_id",
                            "",
                        )
                    )
            source_template, source_node = source_nodes[source_event.op_id]
            owned_by_parent[owner_id].append(
                _OwnedCollective(
                    spec=spec,
                    event=event,
                    source_template=source_template,
                    source_node=source_node,
                    owner_compute_instance_id=owner_id,
                )
            )

        fsdp_op_ids = {
            event.op_id
            for event in real_events
            if event.op_id
            and event.op_id in source_nodes
            and source_nodes[event.op_id][1].annotations.get(
                "raw_op_type"
            ) == "comm.allgather"
        }
        marker_op_ids = {
            op_id
            for template in step_templates.values()
            for op_id, node in template.nodes.items()
            if node.annotations.get("fsdp_marker")
        }
        next_synthetic_id = min(
            (op_id for template in step_templates.values() for op_id in template.nodes),
            default=0,
        ) - 1

        specs_by_template: dict[str, list[ActionSpec]] = defaultdict(list)
        for spec in compute_specs:
            specs_by_template[spec.template_ref].append(spec)

        for base_id, instances in specs_by_template.items():
            base_template = step_templates.get(base_id)
            if base_template is None:
                if any(
                    owned_by_parent.get(
                        str(spec.annotations.get("compute_instance_id", ""))
                    )
                    for spec in instances
                ):
                    raise RuntimeError(
                        f"FSDP-owned communication has no L1 base template {base_id!r}"
                    )
                continue

            signature_by_instance: dict[str, tuple] = {}
            owned_by_signature: dict[
                tuple, list[_OwnedCollective]
            ] = {}
            for spec in instances:
                instance_id = str(
                    spec.annotations.get("compute_instance_id", "")
                )
                owned = owned_by_parent.get(instance_id, [])
                signature = tuple(item.signature for item in owned)
                signature_by_instance[instance_id] = signature
                owned_by_signature.setdefault(signature, owned)

            if not any(owned_by_signature):
                continue

            base_source_ids = set(base_template.nodes)
            signatures = list(owned_by_signature)
            primary_signature = max(
                signatures,
                key=lambda signature: (
                    sum(
                        item.source_node.op_id in base_source_ids
                        for item in owned_by_signature[signature]
                    ),
                    -signatures.index(signature),
                ),
            )
            ordered_signatures = [
                primary_signature,
                *(signature for signature in signatures if signature != primary_signature),
            ]

            normalized_templates: dict[tuple, str] = {}
            for variant_index, signature in enumerate(ordered_signatures):
                template_id = (
                    base_id
                    if variant_index == 0
                    else f"{base_id}__comm_v{variant_index}"
                )
                removed_op_ids = fsdp_op_ids | marker_op_ids
                regions = _fsdp_group_regions(
                    base_template,
                    removed_op_ids,
                )
                pure = _copy_without_nodes(
                    base_template,
                    removed_op_ids,
                    annotations={
                        **base_template.annotations,
                        "communication_ownership_normalized": True,
                    },
                )
                owned = owned_by_signature[signature]
                node_id_map: dict[int, int] = {}
                for item in owned:
                    original_id = item.source_node.op_id
                    if (
                        variant_index == 0
                        and original_id in base_source_ids
                        and original_id not in pure.nodes
                    ):
                        new_id = original_id
                    else:
                        new_id = next_synthetic_id
                        next_synthetic_id -= 1
                    node_id_map[original_id] = new_id

                regions_by_group: dict[
                    str, list[_FSDPGroupRegion]
                ] = defaultdict(list)
                regions_by_module: dict[
                    str, list[_FSDPGroupRegion]
                ] = defaultdict(list)
                for region in regions:
                    regions_by_group[region.group_id].append(region)
                    if region.module_fqn:
                        regions_by_module[region.module_fqn].append(region)
                for grouped_regions in regions_by_group.values():
                    grouped_regions.sort(
                        key=lambda region: (
                            region.wait_seq_idx,
                            region.release_seq_idx,
                            region.group_id,
                        )
                    )
                for module_regions in regions_by_module.values():
                    module_regions.sort(
                        key=lambda region: (
                            region.wait_seq_idx,
                            region.param_group_index,
                            region.group_id,
                        )
                    )
                wait_seq_idxs_by_module = {
                    module_fqn: tuple(
                        region.wait_seq_idx for region in module_regions
                    )
                    for module_fqn, module_regions in regions_by_module.items()
                }
                # Param-group zero begins an FSDP module invocation.  Older
                # traces lack this index, so each such region is its own
                # conservative invocation boundary.
                invocation_starts_by_module = {
                    module_fqn: tuple(
                        index
                        for index, region in enumerate(module_regions)
                        if region.param_group_index in {-1, 0}
                    )
                    for module_fqn, module_regions in regions_by_module.items()
                }
                wait_seq_idxs_by_group = {
                    group_id: tuple(
                        region.wait_seq_idx for region in grouped_regions
                    )
                    for group_id, grouped_regions in regions_by_group.items()
                }

                region_by_source_op_id: dict[
                    int, _FSDPGroupRegion | None
                ] = {}
                comm_id_by_region: dict[_FSDPGroupRegion, int] = {}
                for item in owned:
                    group_id = str(
                        item.spec.annotations.get("fsdp_group_id", "")
                    )
                    region = _fsdp_region_for_collective(
                        regions_by_group=regions_by_group,
                        wait_seq_idxs_by_group=wait_seq_idxs_by_group,
                        group_id=group_id,
                        collective_seq_idx=item.source_node.seq_idx,
                    )
                    region_by_source_op_id[item.source_node.op_id] = region
                    target_parent_id = str(
                        item.spec.annotations.get(
                            "parent_compute_instance_id",
                            "",
                        )
                    )
                    if item.owner_compute_instance_id != target_parent_id:
                        continue
                    # A later action may prefetch the same group again. Only
                    # this action's all-gather proves its source params ready.
                    if region is not None:
                        comm_id_by_region.setdefault(
                            region,
                            node_id_map[item.source_node.op_id],
                        )
                successor_links: dict[int, list[int]] = defaultdict(list)
                prefetch_source_entry_links: list[tuple[int, tuple[int, ...]]] = []
                residency_intervals: list[dict[str, object]] = []
                for item in owned:
                    source = item.source_node
                    new_id = node_id_map[source.op_id]
                    group_id = str(
                        item.spec.annotations.get("fsdp_group_id", "")
                    )
                    module_fqn = (
                        item.event.fsdp_module_fqn
                        or str(
                            item.spec.annotations.get(
                                "fsdp_module_fqn",
                                "",
                            )
                        )
                    )
                    prefetch_source_fqn = (
                        item.event.fsdp_prefetch_source_fqn
                        or str(
                            item.spec.annotations.get(
                                "fsdp_prefetch_source_fqn",
                                "",
                            )
                        )
                    )
                    prefetch_type = (
                        item.event.fsdp_prefetch_type
                        or str(
                            item.spec.annotations.get(
                                "fsdp_prefetch_type",
                                "",
                            )
                        )
                    )
                    region = region_by_source_op_id[source.op_id]
                    if region is None and module_fqn:
                        module_regions = regions_by_module.get(
                            module_fqn,
                            [],
                        )
                        if len(module_regions) == 1:
                            region = module_regions[0]
                    predecessors = [
                        (
                            node_id_map[predecessor]
                            if predecessor in node_id_map
                            else predecessor
                        )
                        for predecessor in source.predecessors
                        if predecessor in pure.nodes
                        or predecessor in node_id_map
                    ]
                    captured_successors = [
                        successor
                        for successor in source.successors
                        if successor in pure.nodes
                    ]
                    source_is_captured = (
                        item.source_template == base_id
                        and source.op_id in base_source_ids
                    )
                    target_parent_id = str(
                        item.spec.annotations.get(
                            "parent_compute_instance_id",
                            "",
                        )
                    )
                    cross_action_prefetch = (
                        item.owner_compute_instance_id
                        != target_parent_id
                    )
                    target_entries: tuple[int, ...] = tuple(
                        captured_successors
                    )
                    if cross_action_prefetch:
                        placement = "cross_action_prefetch"
                    elif prefetch_source_fqn:
                        placement = "layer_prefetch"
                    elif captured_successors:
                        placement = "captured"
                    else:
                        placement = "layer_jit"
                    if not captured_successors:
                        if cross_action_prefetch:
                            target_entries = ()
                        elif region is None:
                            if source_is_captured and not prefetch_source_fqn:
                                residency_intervals.append({
                                    "fsdp_group_id": group_id,
                                    "fsdp_module_fqn": module_fqn,
                                    "allgather_op_id": new_id,
                                    "target_entry_op_ids": [],
                                    "release_after_op_ids": [],
                                    "fsdp_prefetch_source_fqn": (
                                        prefetch_source_fqn
                                    ),
                                    "fsdp_prefetch_type": prefetch_type,
                                    "ownership_placement": placement,
                                })
                                annotations = dict(source.annotations)
                                annotations.update({
                                    "communication_owner": "L1_STAGE",
                                    "source_op_id": source.op_id,
                                    "fsdp_group_id": group_id,
                                    "fsdp_module_fqn": module_fqn,
                                    "fsdp_prefetch_source_fqn": (
                                        prefetch_source_fqn
                                    ),
                                    "fsdp_prefetch_type": prefetch_type,
                                    "ownership_placement": placement,
                                })
                                pure.nodes[new_id] = _clone_node(
                                    source,
                                    op_id=new_id,
                                    predecessors=predecessors,
                                    annotations=annotations,
                                )
                                continue
                            raise RuntimeError(
                                "compute-local FSDP all-gather has no "
                                "parameter-group compute region: "
                                f"template={base_id}, group={group_id}, "
                                f"module={module_fqn!r}"
                            )
                        if not cross_action_prefetch and region is not None:
                            target_entries = region.entry_op_ids
                        if (
                            not cross_action_prefetch
                            and not target_entries
                        ):
                            raise RuntimeError(
                                "compute-local FSDP parameter group has no "
                                "compute entry: "
                                f"template={base_id}, group={group_id}, "
                                f"module={module_fqn!r}"
                            )
                        if not cross_action_prefetch and not prefetch_source_fqn:
                            predecessors.extend(
                                region.external_predecessors
                            )
                            if not predecessors:
                                previous_regions = [
                                    candidate
                                    for candidate in regions
                                    if candidate.release_seq_idx
                                    <= region.wait_seq_idx
                                ]
                                if previous_regions:
                                    previous = max(
                                        previous_regions,
                                        key=lambda candidate: (
                                            candidate.release_seq_idx,
                                            candidate.wait_seq_idx,
                                        ),
                                    )
                                    predecessors.extend(
                                        previous.exit_op_ids
                                    )
                    prefetch_launch_op_id: int | None = None
                    if prefetch_source_fqn:
                        anchor = _fsdp_prefetch_anchor(
                            template_id=base_id,
                            target_group_id=group_id,
                            target_module_fqn=module_fqn,
                            target_region=region,
                            target_collective_seq_idx=source.seq_idx,
                            prefetch_source_fqn=prefetch_source_fqn,
                            regions_by_module=regions_by_module,
                            wait_seq_idxs_by_module=wait_seq_idxs_by_module,
                            invocation_starts_by_module=(
                                invocation_starts_by_module
                            ),
                            comm_id_by_region=comm_id_by_region,
                            nodes_by_id=base_template.nodes,
                        )
                        prefetch_launch_op_id = next_synthetic_id
                        next_synthetic_id -= 1
                        pure.nodes[prefetch_launch_op_id] = OpNode(
                            op_id=prefetch_launch_op_id,
                            op_type=_FSDP_PREFETCH_LAUNCH_OP,
                            inputs=[],
                            outputs=[],
                            attrs={},
                            predecessors=list(anchor.predecessor_op_ids),
                            successors=[],
                            flops=0,
                            peak_mem=0,
                            param_mem=0,
                            comm_bytes=0,
                            annotations={
                                "raw_op_type": _FSDP_PREFETCH_LAUNCH_OP,
                                "control_op": True,
                                "zero_cost": True,
                                "fsdp_group_id": group_id,
                                "fsdp_module_fqn": module_fqn,
                                "fsdp_prefetch_source_fqn": (
                                    prefetch_source_fqn
                                ),
                                "fsdp_prefetch_type": prefetch_type,
                                "ownership_placement": placement,
                                "fsdp_target_compute_instance_id": (
                                    target_parent_id
                                ),
                                "fsdp_prefetch_filtered_post_backward_allreduce_op_ids": (
                                    list(
                                        anchor.filtered_post_backward_allreduce_op_ids
                                    )
                                ),
                            },
                            seq_idx=anchor.seq_idx,
                        )
                        prefetch_source_entry_links.append(
                            (
                                prefetch_launch_op_id,
                                anchor.source_entry_op_ids,
                            )
                        )
                        predecessors.append(prefetch_launch_op_id)
                    predecessors = list(dict.fromkeys(predecessors))
                    annotations = dict(source.annotations)
                    annotations.update({
                        "communication_owner": "L1_STAGE",
                        "source_op_id": source.op_id,
                        "fsdp_group_id": group_id,
                        "fsdp_module_fqn": module_fqn,
                        "fsdp_prefetch_source_fqn": (
                            prefetch_source_fqn
                        ),
                        "fsdp_prefetch_type": prefetch_type,
                        "ownership_placement": placement,
                        "fsdp_target_compute_instance_id": (
                            target_parent_id
                        ),
                        "fsdp_prefetch_launch_op_id": (
                            prefetch_launch_op_id
                        ),
                    })
                    pure.nodes[new_id] = _clone_node(
                        source,
                        op_id=new_id,
                        predecessors=predecessors,
                        annotations=annotations,
                    )
                    successor_links[new_id].extend(target_entries)
                    residency_intervals.append({
                        "fsdp_group_id": group_id,
                        "fsdp_module_fqn": module_fqn,
                        "allgather_op_id": new_id,
                        "target_entry_op_ids": list(target_entries),
                        "release_after_op_ids": (
                            list(region.exit_op_ids)
                            if region is not None
                            else []
                        ),
                        "fsdp_prefetch_source_fqn": (
                            prefetch_source_fqn
                        ),
                        "fsdp_prefetch_type": prefetch_type,
                        "ownership_placement": placement,
                        "fsdp_target_compute_instance_id": (
                            target_parent_id
                        ),
                        "prefetch_launch_op_id": prefetch_launch_op_id,
                    })

                for comm_id, successors in successor_links.items():
                    for successor in successors:
                        if comm_id not in pure.nodes[successor].predecessors:
                            pure.nodes[successor].predecessors.append(comm_id)

                _refresh_graph_topology(pure)
                for launch_op_id, source_entries in prefetch_source_entry_links:
                    skipped_entries: list[int] = []
                    for source_entry_id in source_entries:
                        # Some TP/FSDP layouts place the source all-gather
                        # after a source-region entry. Gating that entry on a
                        # launch which already waits for the all-gather closes
                        # launch -> entry -> all-gather -> launch.
                        if _has_path(pure, source_entry_id, launch_op_id):
                            skipped_entries.append(source_entry_id)
                            continue
                        source_entry = pure.nodes.get(source_entry_id)
                        if (
                            source_entry is not None
                            and launch_op_id not in source_entry.predecessors
                        ):
                            source_entry.predecessors.append(launch_op_id)
                            # Keep reachability current for later cycle checks
                            # without rebuilding the full graph per prefetch
                            # source entry. The final topology refresh below
                            # remains the canonical rebuild.
                            pure.nodes[launch_op_id].successors.append(
                                source_entry_id
                            )
                    if skipped_entries:
                        pure.nodes[launch_op_id].annotations[
                            "fsdp_prefetch_source_gate_skipped_entries"
                        ] = skipped_entries

                (
                    next_synthetic_id,
                    backward_sync_count,
                ) = _add_fsdp_backward_reduction_syncs(
                    pure,
                    regions,
                    next_synthetic_id,
                )

                _refresh_graph_topology(pure)
                if not pure.is_acyclic:
                    raise RuntimeError(
                        "communication ownership produced a cyclic L1 "
                        f"template {template_id!r}"
                    )
                pure.step_id = template_id
                pure.annotations.update({
                    "template_kind": "stage_compute",
                    "internal_fsdp_collectives": len(owned),
                    "fsdp_communication_signature": [
                        list(item.signature) for item in owned
                    ],
                    "fsdp_residency_intervals": residency_intervals,
                    "fsdp_post_backward_syncs": backward_sync_count,
                })
                step_templates[template_id] = pure
                normalized_templates[signature] = template_id
                if template_id != base_id:
                    self.generated_templates.append(template_id)

            for spec in instances:
                instance_id = str(
                    spec.annotations.get("compute_instance_id", "")
                )
                signature = signature_by_instance[instance_id]
                spec.template_ref = normalized_templates[signature]
                spec.annotations["internal_fsdp_collectives"] = len(signature)
                spec.annotations["communication_ownership_normalized"] = True


class FSDPMarkerCleanupPlugin:
    """Remove capture-only FSDP structural markers from exported templates."""

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        del comm_events
        for template_id, template in list(step_templates.items()):
            marker_ids = {
                op_id
                for op_id, node in template.nodes.items()
                if node.annotations.get("fsdp_marker")
            }
            if not marker_ids:
                continue
            cleaned = _copy_without_nodes(template, marker_ids)
            _refresh_graph_topology(cleaned)
            step_templates[template_id] = cleaned
        return specs


class GradientReductionOwnershipPlugin:
    """Keep only genuinely standalone gradient reductions in L2.

    FSDP/DDP reductions may be emitted both as an L1 communication node and as
    a schedule-level REDUCE_GRAD intent.  The L1 template is authoritative when
    the real collective is already inside F/B/I/W.  A real collective captured
    outside every compute template remains an explicit L2 action.
    """

    def __init__(self) -> None:
        self.internal_reductions = 0
        self.external_reductions = 0
        self.removed_noop_intents = 0

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        internal_op_ids: set[int] = set()
        for template in step_templates.values():
            if template.step_type not in _STAGE_TEMPLATE_TYPES:
                continue
            for node in template.nodes.values():
                if node.annotations.get("raw_op_type") not in {
                    "comm.reduce_scatter",
                    "comm.allreduce",
                }:
                    continue
                node.annotations.update({
                    "communication_owner": "L1_STAGE",
                    "gradient_reduction": True,
                })
                internal_op_ids.add(node.op_id)
        self.internal_reductions = len(internal_op_ids)

        event_by_op_id = {
            event.op_id: event
            for event in comm_events
            if event.op_id
            and event.comm_primitive in {"reduce_scatter", "allreduce"}
        }
        retained: list[ActionSpec] = []
        for spec in specs:
            if spec.action_type != "REDUCE_GRAD":
                retained.append(spec)
                continue
            if spec.is_noop or (spec.comm is not None and spec.comm.is_noop):
                self.removed_noop_intents += 1
                continue

            event = event_by_op_id.get(spec.comm_op_id)
            found = (
                _find_comm_node(step_templates, event)
                if event is not None
                else None
            )
            if found is not None:
                template_id, node = found
                template = step_templates[template_id]
                if template.step_type in _STAGE_TEMPLATE_TYPES:
                    node.annotations.update({
                        "communication_owner": "L1_STAGE",
                        "gradient_reduction": True,
                    })
                    template.annotations[
                        "communication_ownership_normalized"
                    ] = True
                    if node.op_id not in internal_op_ids:
                        internal_op_ids.add(node.op_id)
                        self.internal_reductions += 1
                    continue

            spec.annotations["communication_owner"] = "L2_STANDALONE"
            self.external_reductions += 1
            retained.append(spec)
        return retained


class EmbeddedStageCommunicationPlugin:
    """Mark every remaining communication node embedded in stage compute."""

    def __init__(self) -> None:
        self.stage_owned_collectives = 0

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        del comm_events
        counts_by_template: dict[str, int] = {}
        for template_id, template in step_templates.items():
            if template.step_type not in _STAGE_TEMPLATE_TYPES:
                continue
            collective_count = 0
            for node in template.nodes.values():
                if not str(node.annotations.get("raw_op_type", "")).startswith(
                    "comm."
                ):
                    continue
                node.annotations.setdefault(
                    "communication_owner",
                    "L1_STAGE",
                )
                collective_count += 1
                self.stage_owned_collectives += 1
            counts_by_template[template_id] = collective_count
            if collective_count:
                template.annotations[
                    "communication_ownership_normalized"
                ] = True
                template.annotations[
                    "stage_owned_collectives"
                ] = collective_count

        for spec in _iter_specs(specs):
            if spec.action_type == "COMPUTE":
                collective_count = counts_by_template.get(
                    spec.template_ref,
                    0,
                )
                spec.annotations[
                    "stage_owned_collectives"
                ] = collective_count
                spec.annotations[
                    "communication_ownership_normalized"
                ] = True
        return specs


def normalize_communication_ownership(
    *,
    step_templates: dict[str, StepGraph],
    specs: list[ActionSpec],
    comm_events: Iterable[CommEvent],
) -> CommunicationOwnershipResult:
    """Apply capture-side ownership plugins before L2 materialization."""

    events = list(comm_events)
    normalized = list(specs)
    plugins: list[CommunicationOwnershipPlugin] = [
        PipelineP2POwnershipPlugin(),
    ]
    fsdp_plugin = FSDPStageOwnershipPlugin()
    plugins.append(fsdp_plugin)
    plugins.append(FSDPMarkerCleanupPlugin())
    gradient_plugin = GradientReductionOwnershipPlugin()
    plugins.append(gradient_plugin)
    embedded_plugin = EmbeddedStageCommunicationPlugin()
    plugins.append(embedded_plugin)
    for plugin in plugins:
        normalized = plugin.apply(
            step_templates=step_templates,
            specs=normalized,
            comm_events=events,
        )
    return CommunicationOwnershipResult(
        specs=normalized,
        internal_fsdp_transitions=set(fsdp_plugin.internal_transitions),
        external_fsdp_transitions=set(fsdp_plugin.external_transitions),
        generated_templates=list(fsdp_plugin.generated_templates),
        internal_gradient_reductions=gradient_plugin.internal_reductions,
        external_gradient_reductions=gradient_plugin.external_reductions,
        removed_noop_gradient_intents=gradient_plugin.removed_noop_intents,
        stage_owned_collectives=embedded_plugin.stage_owned_collectives,
    )
