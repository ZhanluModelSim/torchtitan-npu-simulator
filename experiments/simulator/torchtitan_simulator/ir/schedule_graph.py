# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L2 ScheduleGraph: describes how StepGraph instances are orchestrated --
parallel strategy, pipeline, microbatch loop, multi-device coordination.
See spec/L2-ScheduleGraph.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan_simulator.ir.step_graph import StepGraph


@dataclass
class TimelineEntry:
    """One op's position in the execution timeline, captured (not inferred)."""

    seq_idx: int  # source L0 sequence number, not L2 schedule order
    op_id: int  # L0 OpNode ID (-1 for MB 1+ pass-through)
    rank: int  # which rank executed this op
    pipeline_stage: int  # PP stage (-1 if not PP)
    micro_batch_idx: int  # microbatch index (-1 if not PP)
    phase: str  # "forward" / "backward" / "optimizer" / "comm"
    comm_type: str = ""  # "fwd_send" / "fwd_recv" / "bwd_send" / "bwd_recv" / "allgather" / ...
    comm_peer_rank: int = -1  # for P2P: peer rank
    action: str = ""  # "compute" / "forward_one_chunk" / "backward_one_chunk" / "comm"
    # Compute-graph class ("F"/"B"/"I"/"W"/"F_RECOMPUTE"/"OPTIMIZER") for this
    # timeline entry. Lets L2 consumers distinguish input-grad vs weight-grad
    # backward chunks and pick the correct L1 template per microbatch.
    comp_type: str = ""
    schedule_order: int = -1
    transfer_id: str = ""


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
