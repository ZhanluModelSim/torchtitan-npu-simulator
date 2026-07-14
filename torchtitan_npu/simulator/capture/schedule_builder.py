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
    # 1. StepInstance: one per captured compute chunk (microbatch × stage ×
    #    comp_type). For PP steps every forward_one_chunk / backward_one_chunk /
    #    backward_weight_one_chunk call produced a timeline_event carrying
    #    (pp_mb_idx, pp_stage, comp_type) — including pass-through microbatches
    #    whose L0 graph was deduped. Instantiate the matching template for each
    #    so the L2 schedule reflects every microbatch, not just MB 0.
    instances: list[StepInstance] = []
    coords = rank_table.rank_coordinates.get(rank, {})
    seen_instance_ids: set[str] = set()
    # Map template_id -> (step_type, fsdp_state) for fallback when a chunk's
    # comp_type has no captured template (e.g. only MB 1+ ran a class but MB 0
    # ran a different one — rare; fall back to the matching comp_type template).
    template_by_comp: dict[str, tuple[str, StepGraph]] = {}
    for template_id, template in step_templates.items():
        template_by_comp[template.step_type] = (template_id, template)

    def _make_instance(mb_idx: int, stage: int, comp_type: str, fsdp_state: str) -> None:
        template_id = f"s{stage}_{comp_type}"
        step_type = comp_type
        # Fall back to whatever template exists for this comp_type if the exact
        # stage's template wasn't captured (shouldn't normally happen).
        if template_id not in step_templates and comp_type in template_by_comp:
            template_id, _ = template_by_comp[comp_type]
        instance_id = f"rank{rank}_s{stage}_mb{mb_idx}_{comp_type}"
        if instance_id in seen_instance_ids:
            return
        seen_instance_ids.add(instance_id)
        instances.append(
            StepInstance(
                instance_id=instance_id,
                step_ref=template_id,
                step_type=step_type,
                micro_batch_idx=mb_idx,
                pipeline_stage=stage,
                device_ids=[rank],
                dp_group=coords.get("dp_replicate", 0),
                comp_type=comp_type,
                fsdp_state=fsdp_state,
            )
        )

    if timeline_events:
        for ev in timeline_events:
            comp_type = ev.get("comp_type") or ""
            if not comp_type:
                # Legacy timeline events (no comp_type): infer from action.
                action = ev.get("action", "")
                comp_type = "F" if "forward" in action else ("W" if "weight" in action else "B")
            _make_instance(
                ev.get("pp_mb_idx", 0),
                ev.get("pp_stage", coords.get("pp", rank)),
                comp_type,
                "NA",
            )
    # For non-PP steps (no timeline_events) or any template without a chunk
    # event, emit one MB 0 instance per captured template so the templates are
    # still represented in the L2 graph.
    if not instances:
        for template_id, template in step_templates.items():
            comp_type = template.step_type
            instances.append(
                StepInstance(
                    instance_id=f"rank{rank}_{template_id}",
                    step_ref=template_id,
                    step_type=comp_type,
                    micro_batch_idx=0,
                    pipeline_stage=coords.get("pp", rank),
                    device_ids=[rank],
                    dp_group=coords.get("dp_replicate", 0),
                    comp_type=comp_type,
                )
            )
            seen_instance_ids.add(f"rank{rank}_{template_id}")

    # 2. DataPass: only from L2 comm events (PP/FSDP), not L1 (CP)
    data_passes: list[DataPass] = []
    for event in comm_events:
        if event.comm_layer == "L1":
            continue  # CP comm → L1 StepGraph internal, not L2 DataPass
        if event.comm_primitive in ("p2p_send", "p2p_recv"):
            if event.comm_primitive != "p2p_send":
                continue
            src_stage = event.p2p_stage
            dst_rank = event.p2p_peer_rank
            slot = TensorSlot(
                name=f"p2p_{event.p2p_direction}_{event.event_id}",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_exit_op=event.src_exit_op or event.op_id,
                dst_entry_op=event.dst_entry_op or event.op_id,
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
            # FSDP collective: one DataPass per call
            slot = TensorSlot(
                name=f"{event.comm_primitive}_{event.event_id}",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
                src_exit_op=event.src_exit_op or event.op_id,
                dst_entry_op=event.dst_entry_op or event.op_id,
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

    # 3. execution_timeline: only L2-level events (no L0 compute, no L1 CP comm)
    execution_timeline: list[TimelineEntry] = []

    # 3a. MB 0 L0 ops → only L2 comm ops appear in timeline (not compute ops)
    for template_id, template in step_templates.items():
        for op_id, node in template.nodes.items():
            ann = node.annotations
            raw = ann.get("raw_op_type", "")
            if not raw.startswith("comm."):
                continue  # skip L0 compute ops in L2 timeline
            # Find matching CommEvent
            for ev in comm_events:
                if ev.op_id == op_id:
                    if ev.comm_layer == "L1":
                        break  # skip L1 (CP) comm in L2 timeline
                    # L2 comm: add to timeline
                    comm_type = ev.p2p_direction or ev.comm_primitive
                    comm_peer = ev.p2p_peer_rank if ev.p2p_direction else -1
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
                            action="comm",
                            comp_type=ann.get("comp_type", ""),
                        )
                    )
                    break

    # 3b. every microbatch's compute chunks (scheduling:
    # forward_one_chunk/backward_one_chunk/backward_weight_one_chunk). These
    # cover ALL microbatches — MB 0's captured chunks plus the pass-through
    # duplicates — so the L2 timeline reflects the full schedule, and each
    # entry carries its comp_type so consumers can pair it with the right L1
    # template.
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
                    comp_type=ev.get("comp_type", ""),
                )
            )

    # 3c. L2 comm events from MB 1+ (not captured as L0 ops)
    captured_op_ids = {e.op_id for e in execution_timeline if e.op_id > 0}
    for event in comm_events:
        if event.comm_layer == "L1":
            continue  # skip CP comm
        if event.op_id > 0 and event.op_id in captured_op_ids:
            continue  # already in timeline via L0 op
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
                comp_type=event.comp_type,
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
