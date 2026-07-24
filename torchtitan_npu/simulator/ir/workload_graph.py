# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L3 WorkloadGraph: the outermost container -- holds a ScheduleGraph
template plus iteration semantics and data-flow cadence. See
spec/L3-WorkloadGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph
from torchtitan_npu.simulator.ir.schedule_plan import SchedulePlan
from torchtitan_npu.simulator.ir.step_graph import StepGraph


@dataclass
class DataFlow:
    """Describes one input or output data stream of the workload."""

    source: str
    tensor_shape: tuple[int | str, ...]
    dtype: str
    volume_per_iter: int
    is_streaming: bool = False
    interleave_strategy: str = "synced"


@dataclass
class IterationSpec:
    """One training/inference iteration: which ScheduleGraph it runs, and
    how many microbatches it contains."""

    schedule: ScheduleGraph
    microbatch_count: int
    iteration_time_est: float = 0.0


@dataclass
class WorkloadGraph:
    """Top-level container for a complete workload: train/inference/rag/
    recommendation, iteration semantics, and cross-iteration data flow."""

    workload_id: str
    workload_type: str
    step_templates: dict[str, StepGraph]
    iteration: IterationSpec
    num_iterations: int
    warmup_iterations: int = 0
    data_inputs: list[DataFlow] = field(default_factory=list)
    data_outputs: list[DataFlow] = field(default_factory=list)
    cross_iter_passes: list[DataPass] = field(default_factory=list)
    # Structured L2 scheduling view (ordered ScheduleActions + DataSlots).
    # None only when build_workload_graph was called without a plan (e.g. a
    # legacy caller); the flat ScheduleGraph remains on `iteration.schedule`.
    schedule_plan: SchedulePlan | None = None
    total_runtime_est: float = 0.0
    total_cost_est: float = 0.0

    def export_schedule_csv(self, path: str) -> None:
        """Export L3→L2: inter-rank scheduling relationships.

        Outputs ``rank_schedule.csv`` showing how ranks (PP stages)
        execute in parallel or serial, with P2P communication dependencies
        between them.

        Columns:
            schedule_order, seq_idx, pipeline_stage, phase, microbatch, action,
            comm_type, comm_peer_rank, depends_on_stage, depends_on_seq

        All data from captured execution_timeline.  ``depends_on_stage``
        and ``depends_on_seq`` are derived from exact ``transfer_id`` joins.
        Stage/direction/microbatch matching is retained only for legacy
        captures without transfer IDs.
        """
        import csv

        schedule = self.iteration.schedule
        timeline = schedule.execution_timeline

        # Build a lookup: for each P2P send (stage S, direction fwd/bwd send,
        # microbatch M), find its seq_idx.  Then a P2P recv with matching
        # stage/direction/mb can look up the send's seq_idx as its dependency.
        # send_key = (stage, direction_without_send/recv_suffix, mb_idx)
        # Per-rank exports may not contain the peer's send row. In that case
        # transfer_id remains the complete cross-rank join key even though
        # depends_on_seq is empty in this individual CSV.
        send_seq_lookup: dict[tuple[int, str, int], int] = {}
        send_transfer_lookup: dict[str, int] = {}
        for entry in timeline:
            if entry.comm_type and "send" in entry.comm_type:
                base_dir = entry.comm_type.replace("_send", "")
                key = (entry.pipeline_stage, base_dir, entry.micro_batch_idx)
                send_seq_lookup[key] = entry.seq_idx
                if entry.transfer_id:
                    send_transfer_lookup[entry.transfer_id] = entry.seq_idx

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "schedule_order", "seq_idx", "pipeline_stage", "phase", "microbatch", "action",
                "comm_type", "comm_peer_rank",
                "transfer_id", "depends_on_stage", "depends_on_seq",
            ])
            for entry in sorted(
                timeline,
                key=lambda e: (
                    e.schedule_order if e.schedule_order >= 0 else e.seq_idx,
                    e.seq_idx,
                ),
            ):
                depends_on_stage = ""
                depends_on_seq = ""
                if entry.comm_type and "recv" in entry.comm_type:
                    # A recv depends on the matching send from the peer stage.
                    # For forward_recv: send came from stage-1 (prev stage)
                    # For backward_recv: send came from stage+1 (next stage)
                    base_dir = entry.comm_type.replace("_recv", "")
                    if "forward" in base_dir:
                        send_stage = entry.pipeline_stage - 1
                    else:
                        send_stage = entry.pipeline_stage + 1
                    depends_on_stage = send_stage
                    key = (send_stage, base_dir, entry.micro_batch_idx)
                    depends_on_seq = (
                        send_transfer_lookup.get(entry.transfer_id, "")
                        if entry.transfer_id
                        else send_seq_lookup.get(key, "")
                    )

                stage = entry.pipeline_stage if entry.pipeline_stage >= 0 else 0
                action = (
                    entry.comm_type
                    or entry.action
                    or f"{entry.phase}_one_chunk"
                )
                w.writerow([
                    entry.schedule_order,
                    entry.seq_idx,
                    stage,
                    entry.phase,
                    entry.micro_batch_idx if entry.micro_batch_idx >= 0 else "",
                    action,
                    entry.comm_type,
                    entry.comm_peer_rank if entry.comm_peer_rank >= 0 else "",
                    entry.transfer_id,
                    depends_on_stage,
                    depends_on_seq,
                ])
