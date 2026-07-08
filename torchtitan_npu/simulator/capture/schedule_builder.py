# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assembles the L2 ScheduleGraph from a captured L1 template, the
RankTable, and recorded communication events. See design doc §5.5."""

from __future__ import annotations

import uuid

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph, StepInstance, TensorSlot, TimelineEntry
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.rank_table import RankTable


def build_schedule_graph(
    *,
    step_templates: dict[str, StepGraph],
    rank_table: RankTable,
    comm_events: list[CommEvent],
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
) -> ScheduleGraph:
    """Build L2 ScheduleGraph from captured L1 templates, RankTable, and
    communication events.  All timeline information comes from capture
    (seq_idx, P2P context), not from inference."""
    # Create instances for every step template per rank
    instances: list[StepInstance] = []
    for rank in range(rank_table.world_size):
        coords = rank_table.rank_coordinates.get(rank, {})
        for template_id, template in step_templates.items():
            instances.append(
                StepInstance(
                    instance_id=f"rank{rank}_{template_id}",
                    step_ref=template_id,
                    step_type=template.step_type,
                    micro_batch_idx=0,
                    pipeline_stage=coords.get("pp", 0),
                    device_ids=[rank],
                    dp_group=coords.get("dp_replicate", 0),
                )
            )

    # Build DataPasses from comm events
    data_passes: list[DataPass] = []
    for event in comm_events:
        if event.comm_primitive in ("p2p_send", "p2p_recv"):
            # P2P communication (pipeline parallelism): one-to-one transfer
            # Only create a DataPass for p2p_send (avoid duplicating with p2p_recv)
            if event.comm_primitive != "p2p_send":
                continue
            # Determine src and dst ranks from P2P context
            # p2p_peer_rank is the destination (for send) or source (for recv)
            # We need the sender's rank: it's the rank that issued the isend.
            # Under fake PG all ranks run in one process, so the "sender" is
            # the current rank (rank 0 in the fake PG).  The peer_rank is the
            # destination.  For a more accurate model, we use the PP stage
            # info: stage N sends to stage N+1 (forward) or N-1 (backward).
            src_stage = event.p2p_stage
            if "forward" in event.p2p_direction:
                dst_stage = src_stage + 1
            else:
                dst_stage = src_stage - 1
            # Find a representative rank for each stage
            src_rank = -1
            dst_rank = event.p2p_peer_rank
            for r in range(rank_table.world_size):
                coords = rank_table.rank_coordinates.get(r, {})
                if coords.get("pp", -1) == src_stage:
                    src_rank = r
                    break
            slot = TensorSlot(
                name=f"p2p_{event.p2p_direction}_{event.event_id}",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_exit_op=event.op_id,
                dst_entry_op=event.op_id,
            )
            data_passes.append(
                DataPass(
                    src_instance=f"rank{src_rank}",
                    dst_instance=f"rank{dst_rank}",
                    slots=[slot],
                    src_device=src_rank,
                    dst_device=dst_rank,
                    requires_communication=True,
                    comm_primitive=f"p2p_{event.p2p_direction}",
                )
            )
        else:
            # Collective communication (FSDP/TP/EP): all-to-all within group
            dim_name = rank_table.dim_by_group_name.get(event.group_name)
            if not dim_name and event.comm_dim:
                dim_name = event.comm_dim
            if event.comm_ranks:
                groups = event.comm_ranks
            else:
                groups = rank_table.process_groups.get(dim_name, []) if dim_name else []
            for group in groups:
                if len(group) < 2:
                    continue
                slot = TensorSlot(
                    name=f"{event.comm_primitive}_{event.event_id}",
                    shape=event.tensor_shape,
                    dtype=event.dtype,
                    volume_bytes=event.volume_bytes,
                    src_exit_op=event.op_id,
                    dst_entry_op=event.op_id,
                )
                for i, src_rank in enumerate(group):
                    for dst_rank in group[i + 1 :]:
                        data_passes.append(
                            DataPass(
                                src_instance=f"rank{src_rank}",
                                dst_instance=f"rank{dst_rank}",
                                slots=[slot],
                                src_device=src_rank,
                                dst_device=dst_rank,
                                requires_communication=True,
                                comm_primitive=event.comm_primitive,
                            )
                        )

    # Build execution_timeline from captured L0 OpNodes' seq_idx
    # All ranks share the same template, so we use rank 0's seq_idx as
    # the representative timeline.  For PP, the P2P events carry their
    # own stage/mb context, and non-P2P ops get stage/mb from _pp_context
    # via the phase_provider (which reads _pp_context["phase"]).
    execution_timeline: list[TimelineEntry] = []
    for template_id, template in step_templates.items():
        for op_id, node in template.nodes.items():
            ann = node.annotations
            phase = ann.get("phase", template.step_type)
            comm_type = ""
            comm_peer = -1
            # Check if this op is a communication op
            raw = ann.get("raw_op_type", "")
            if raw.startswith("comm."):
                # Find the matching CommEvent for P2P info
                for ev in comm_events:
                    if ev.op_id == op_id:
                        if ev.p2p_direction:
                            comm_type = ev.p2p_direction
                            comm_peer = ev.p2p_peer_rank
                        else:
                            comm_type = ev.comm_primitive
                        break
            execution_timeline.append(
                TimelineEntry(
                    seq_idx=node.seq_idx,
                    op_id=op_id,
                    rank=0,  # representative rank
                    pipeline_stage=-1,  # filled below
                    micro_batch_idx=-1,  # filled below
                    phase=phase,
                    comm_type=comm_type,
                    comm_peer_rank=comm_peer,
                )
            )
    # Sort by seq_idx (execution order)
    execution_timeline.sort(key=lambda e: e.seq_idx)

    # For P2P comm events, fill in pipeline_stage and micro_batch_idx
    # from the CommEvent's captured PP context
    for entry in execution_timeline:
        if entry.comm_type:
            for ev in comm_events:
                if ev.op_id == entry.op_id and ev.p2p_direction:
                    entry.pipeline_stage = ev.p2p_stage
                    entry.micro_batch_idx = ev.p2p_mb_idx
                    break

    # For non-P2P ops, fill in pipeline_stage and micro_batch_idx from
    # the captured _pp_context.  The _pp_context is updated by the patched
    # forward_one_chunk / backward_one_chunk, so every op captured between
    # two chunk calls has the correct stage/mb.  We reconstruct this by
    # finding the nearest preceding P2P event with the same phase and
    # using its stage/mb.
    # Build a list of (seq_idx, stage, mb, phase) from P2P events as anchors
    pp_anchors: list[tuple[int, int, int, str]] = []
    for entry in execution_timeline:
        if entry.comm_type and entry.pipeline_stage >= 0:
            # Determine phase from comm_type: forward_send/recv -> forward, backward_send/recv -> backward
            if "forward" in entry.comm_type:
                anchor_phase = "forward"
            elif "backward" in entry.comm_type:
                anchor_phase = "backward"
            else:
                anchor_phase = entry.phase
            pp_anchors.append((entry.seq_idx, entry.pipeline_stage, entry.micro_batch_idx, anchor_phase))
    pp_anchors.sort(key=lambda x: x[0])

    # For each non-P2P entry, find the nearest preceding anchor with
    # matching phase and use its stage/mb
    for entry in execution_timeline:
        if entry.pipeline_stage >= 0:
            continue  # already filled (P2P)
        # Find nearest preceding anchor with matching phase
        best_anchor = None
        for a_seq, a_stage, a_mb, a_phase in pp_anchors:
            if a_seq > entry.seq_idx:
                break
            if a_phase == entry.phase:
                best_anchor = (a_stage, a_mb)
        if best_anchor is not None:
            entry.pipeline_stage = best_anchor[0]
            entry.micro_batch_idx = best_anchor[1]

    dp_degree = rank_table.dim_degrees.get("dp_replicate", 1) * rank_table.dim_degrees.get(
        "fsdp", rank_table.dim_degrees.get("dp_shard", 1)
    )

    return ScheduleGraph(
        schedule_id=uuid.uuid4().hex[:12],
        workload_type="train",
        step_templates=step_templates,
        instances=instances,
        data_passes=data_passes,
        dp_degree=dp_degree,
        tp_degree=rank_table.dim_degrees.get("tp", 1),
        pp_degree=rank_table.dim_degrees.get("pp", 1),
        num_micro_batches=num_micro_batches,
        pipeline_schedule=pipeline_schedule,
        gradient_accumulation=gradient_accumulation,
        execution_timeline=execution_timeline,
        annotations={"rank_table": rank_table.to_dict()},
    )
