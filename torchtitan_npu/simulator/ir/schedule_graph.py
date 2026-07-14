# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L2 ScheduleGraph: describes how StepGraph instances are orchestrated --
parallel strategy, pipeline, microbatch loop, multi-device coordination.
See spec/L2-ScheduleGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan_npu.simulator.ir.step_graph import StepGraph


@dataclass
class TimelineEntry:
    """One op's position in the execution timeline, captured (not inferred)."""

    seq_idx: int           # global execution sequence number (from capture)
    op_id: int             # L0 OpNode ID (-1 for MB 1+ pass-through)
    rank: int              # which rank executed this op
    pipeline_stage: int    # PP stage (-1 if not PP)
    micro_batch_idx: int   # microbatch index (-1 if not PP)
    phase: str             # "forward" / "backward" / "optimizer" / "comm"
    comm_type: str = ""    # "fwd_send" / "fwd_recv" / "bwd_send" / "bwd_recv" / "allgather" / ...
    comm_peer_rank: int = -1  # for P2P: peer rank
    action: str = ""       # "compute" / "forward_one_chunk" / "backward_one_chunk" / "comm"
    # Compute-graph class ("F"/"B"/"I"/"W"/"F_RECOMPUTE"/"OPTIMIZER") for this
    # timeline entry. Lets L2 consumers distinguish input-grad vs weight-grad
    # backward chunks and pick the correct L1 template per microbatch.
    comp_type: str = ""


@dataclass
class StepInstance:
    """One concrete execution of a StepGraph template."""

    instance_id: str
    step_ref: str
    step_type: str
    micro_batch_idx: int
    pipeline_stage: int
    device_ids: list[int]
    dp_group: int
    estimated_runtime: float = 0.0
    # Compute-graph class of the referenced template (mirrors step_type but
    # kept on the instance for convenience).
    comp_type: str = ""
    # FSDP sharding state active during this instance's execution.
    fsdp_state: str = "NA"


@dataclass
class TensorSlot:
    """A named tensor transferred between two StepInstances."""

    name: str
    shape: tuple[int | str, ...]
    dtype: str
    volume_bytes: int
    src_exit_op: int = 0
    dst_entry_op: int = 0
    is_incremental: bool = False


@dataclass
class DataPass:
    """A data dependency (possibly requiring communication) between two
    StepInstances."""

    src_instance: str
    dst_instance: str
    slots: list[TensorSlot]
    src_device: int | None = None
    dst_device: int | None = None
    requires_communication: bool = False
    comm_primitive: str = ""
    comm_group_ranks: list[list[int]] = field(default_factory=list)


@dataclass
class ScheduleGraph:
    """Orchestration graph: StepGraph templates + concrete StepInstances +
    the DataPasses that connect them."""

    schedule_id: str
    workload_type: str
    step_templates: dict[str, StepGraph]
    instances: list[StepInstance]
    instance_map: dict[str, StepInstance] = field(default_factory=dict)
    data_passes: list[DataPass] = field(default_factory=list)
    ctrl_edges: list[tuple[str, str]] = field(default_factory=list)
    dp_degree: int = 1
    tp_degree: int = 1
    pp_degree: int = 1
    num_micro_batches: int = 1
    pipeline_schedule: str = "none"
    gradient_accumulation: int = 1
    zero_stage: int = 0
    timeline: list = field(default_factory=list)
    execution_timeline: list[TimelineEntry] = field(default_factory=list)
    annotations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_map and self.instances:
            self.instance_map = {instance.instance_id: instance for instance in self.instances}

    def export_l1_schedule_csv(self, path: str, *, max_ranks: int | None = None) -> None:
        """Export L2→L1: per-rank L1 subgraph execution schedule.

        Outputs one CSV per rank under ``path/`` directory (created if
        needed).  Each row = one captured execution event (forward chunk,
        backward chunk, P2P send/recv, collective comm), ordered by
        captured ``seq_idx``.

        Columns:
            seq_idx, phase, microbatch, action, comm_type, comm_peer_rank,
            comm_dim, comm_ranks, op_id, tensor_shape, volume_bytes

        All data from captured execution_timeline + CommEvent fields.
        No scheduling rules are inferred or hardcoded.
        """
        import csv
        import os

        os.makedirs(path, exist_ok=True)

        # Group execution_timeline entries by rank
        # (currently all entries have rank=0 since capture is single-process;
        # for PP, the pipeline_stage field distinguishes stages)
        # We export one file per pipeline_stage (since all ranks in the same
        # stage share the same template/schedule)
        stages_seen: set[int] = set()
        for entry in self.execution_timeline:
            stage = entry.pipeline_stage if entry.pipeline_stage >= 0 else 0
            stages_seen.add(stage)

        for stage in sorted(stages_seen):
            # Filter entries for this stage
            stage_entries = [
                e for e in self.execution_timeline
                if (e.pipeline_stage if e.pipeline_stage >= 0 else 0) == stage
            ]
            stage_entries.sort(key=lambda e: e.seq_idx)

            fname = os.path.join(path, f"stage_{stage}_l1_schedule.csv")
            with open(fname, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    "seq_idx", "phase", "comp_type", "microbatch", "action",
                    "comm_type", "comm_peer_rank", "comm_dim", "comm_ranks",
                    "op_id",
                ])
                for e in stage_entries:
                    # Use captured action field directly
                    if e.action:
                        action = e.action
                    elif e.comm_type:
                        action = e.comm_type
                    else:
                        action = "compute"
                    w.writerow([
                        e.seq_idx, e.phase, e.comp_type,
                        e.micro_batch_idx if e.micro_batch_idx >= 0 else "",
                        action,
                        e.comm_type, e.comm_peer_rank,
                        "", "",  # comm_dim/comm_ranks not in TimelineEntry
                        e.op_id,
                    ])
