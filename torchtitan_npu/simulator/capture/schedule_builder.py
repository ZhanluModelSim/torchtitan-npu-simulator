# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assembles the L2 ScheduleGraph from captured L1 templates, communication
events, and timeline events.  All fields come from capture — no RankTable
traversal, no all-to-all expansion, no P2P anchor inference.

See docs/design/schedule-capture-design.md for the design rationale."""

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
    timeline_events: list[dict] | None = None,
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
    rank: int = 0,
) -> ScheduleGraph:
    """Build L2 ScheduleGraph from captured data.

    All timeline information comes from capture (seq_idx, _pp_context,
    CommEvent fields) — not from inference or RankTable traversal.
    """
    # 1. StepInstance: one per template for this rank (captured, not generated)
    instances: list[StepInstance] = []
    coords = rank_table.rank_coordinates.get(rank, {})
    for template_id, template in step_templates.items():
        instances.append(
            StepInstance(
                instance_id=f"rank{rank}_{template_id}",
                step_ref=template_id,
                step_type=template.step_type,
                micro_batch_idx=0,
                pipeline_stage=coords.get("pp", rank),
                device_ids=[rank],
                dp_group=coords.get("dp_replicate", 0),
            )
        )

    # 2. DataPass: from CommEvent directly (no all-to-all expansion)
    data_passes: list[DataPass] = []
    for event in comm_events:
        if event.comm_primitive in ("p2p_send", "p2p_recv"):
            # P2P: only create DataPass for send (avoid duplicating with recv)
            if event.comm_primitive != "p2p_send":
                continue
            src_stage = event.p2p_stage
            dst_rank = event.p2p_peer_rank
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
                    src_instance=f"rank{src_stage}",
                    dst_instance=f"rank{dst_rank}",
                    slots=[slot],
                    src_device=src_stage,
                    dst_device=dst_rank,
                    requires_communication=True,
                    comm_primitive=f"p2p_{event.p2p_direction}",
                )
            )
        else:
            # Collective: one DataPass per call (no all-to-all expansion)
            slot = TensorSlot(
                name=f"{event.comm_primitive}_{event.event_id}",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_exit_op=event.op_id,
                dst_entry_op=event.op_id,
            )
            data_passes.append(
                DataPass(
                    src_instance=f"rank{rank}",
                    dst_instance=f"group:{event.comm_dim}",
                    slots=[slot],
                    src_device=rank,
                    dst_device=None,
                    requires_communication=True,
                    comm_primitive=event.comm_primitive,
                    comm_group_ranks=event.comm_ranks,
                )
            )

    # 3. execution_timeline: merge L0 ops, timeline events, and comm events
    execution_timeline: list[TimelineEntry] = []

    # 3a. MB 0 L0 ops → timeline entries (op_id has value, action="compute")
    for template_id, template in step_templates.items():
        for op_id, node in template.nodes.items():
            ann = node.annotations
            comm_type = ""
            comm_peer = -1
            raw = ann.get("raw_op_type", "")
            if raw.startswith("comm."):
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
                    rank=rank,
                    pipeline_stage=ann.get("pp_stage", -1),
                    micro_batch_idx=ann.get("pp_mb_idx", -1),
                    phase=ann.get("phase", template.step_type),
                    comm_type=comm_type,
                    comm_peer_rank=comm_peer,
                    action="compute" if not comm_type else "comm",
                )
            )

    # 3b. MB 1+ timeline events (op_id=-1, action="forward_one_chunk"/etc.)
    if timeline_events:
        for ev in timeline_events:
            execution_timeline.append(
                TimelineEntry(
                    seq_idx=ev["seq_idx"],
                    op_id=-1,
                    rank=rank,
                    pipeline_stage=ev["pp_stage"],
                    micro_batch_idx=ev["pp_mb_idx"],
                    phase=ev["phase"],
                    action=ev["action"],
                )
            )

    # 3c. Comm events that were NOT captured as L0 ops (MB 1+ pass-through)
    captured_op_ids = {e.op_id for e in execution_timeline if e.op_id > 0}
    for event in comm_events:
        if event.op_id > 0 and event.op_id in captured_op_ids:
            continue  # already in timeline via L0 op
        # This comm event happened during MB 1+ (no L0 op)
        execution_timeline.append(
            TimelineEntry(
                seq_idx=event.seq_idx,
                op_id=event.op_id if event.op_id > 0 else -1,
                rank=rank,
                pipeline_stage=event.p2p_stage if event.p2p_stage >= 0 else rank,
                micro_batch_idx=event.p2p_mb_idx if event.p2p_mb_idx >= 0 else -1,
                phase="comm",
                comm_type=event.p2p_direction or event.comm_primitive,
                comm_peer_rank=event.p2p_peer_rank,
                action="comm",
            )
        )

    # 4. Sort by seq_idx (execution order)
    execution_timeline.sort(key=lambda e: e.seq_idx)

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
