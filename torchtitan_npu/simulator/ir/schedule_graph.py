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


@dataclass
class TensorSlot:
    """A named tensor transferred between two StepInstances."""

    name: str
    src_exit_op: str
    dst_entry_op: str
    shape: tuple[int | str, ...]
    dtype: str
    volume_bytes: int
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
    annotations: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instance_map and self.instances:
            self.instance_map = {instance.instance_id: instance for instance in self.instances}
