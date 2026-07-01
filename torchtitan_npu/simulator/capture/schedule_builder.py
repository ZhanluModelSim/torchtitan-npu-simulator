# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assembles the L2 ScheduleGraph from a captured L1 template, the
RankTable, and recorded communication events. See design doc §5.5."""

from __future__ import annotations

import uuid

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph, StepInstance, TensorSlot
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
    """Build one StepInstance per logical rank -- all ranks share the
    single captured template when `pipeline_parallel_degree == 1` (the
    acceptance config's case; see design doc §5.5 for the general
    per-pipeline-stage template note, out of scope for this task) -- plus
    one DataPass per communication-group member pair for every recorded
    CommEvent whose `group_name` resolves to a known RankTable dimension.
    """
    template_id = next(iter(step_templates), "")
    template_step_type = step_templates[template_id].step_type if template_id else "forward"

    instances: list[StepInstance] = []
    for rank in range(rank_table.world_size):
        coords = rank_table.rank_coordinates.get(rank, {})
        instances.append(
            StepInstance(
                instance_id=f"rank{rank}",
                step_ref=template_id,
                step_type=template_step_type,
                micro_batch_idx=0,
                pipeline_stage=coords.get("pp", 0),
                device_ids=[rank],
                dp_group=coords.get("dp_replicate", 0),
            )
        )

    data_passes: list[DataPass] = []
    for event in comm_events:
        dim_name = rank_table.dim_by_group_name.get(event.group_name)
        groups = rank_table.process_groups.get(dim_name, []) if dim_name else []
        for group in groups:
            if len(group) < 2:
                continue
            slot = TensorSlot(
                name=f"{event.comm_primitive}_{event.event_id}",
                src_exit_op="",
                dst_entry_op="",
                shape=event.tensor_shape,
                dtype=event.dtype,
                volume_bytes=event.volume_bytes,
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
        annotations={"rank_table": rank_table.to_dict()},
    )
