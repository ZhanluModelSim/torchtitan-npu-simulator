# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L2 SchedulePlan: the structured scheduling view of how L1 StepGraph
templates are instantiated and orchestrated across one training/inference
iteration, plus the data (DataSlot) that flows between those
instantiations.  See docs/design/l2-l3-schedule-plan-design.md.

A ``SchedulePlan`` is an ordered list of ``ScheduleAction``s (one per
pipeline compute chunk / FSDP unshard-reshard / P2P send-recv / optimizer
step), each referencing an L1 template (for COMPUTE) and the DataSlots it
consumes/produces.  Producer→consumer edges are encoded implicitly via
``DataSlot.producer_action_id`` / ``consumer_action_ids`` — a slot produced
by action A and consumed by action B is a data dependency A→B.

This replaces the old flat ``ScheduleGraph.execution_timeline`` trace as
the primary L2 object: the trace only recorded *what ran* (captured seq
events), not the *plan structure* (which (stage, comp_type, microbatch)
runs when, and what data crosses stage boundaries).  ``pipeline_order_with_comms``
(the runtime schedule's lowered plan, containing F/B/I/W + UNSHARD/RESHARD/
SEND_F/RECV_F/SEND_B/RECV_B/REDUCE_GRAD + OVERLAP_F_B) is the structural
source; capture enriches each action with seq_idx / template_ref / DataSlot
shapes / L0 op-level src_exit_op↔dst_entry_op linkage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torchtitan_npu.simulator.ir.step_graph import StepGraph


@dataclass
class DataSlot:
    """A tensor flowing between two ScheduleActions.

    ``kind`` controls how the slot is interpreted:
      * ``activation``       — forward output of stage S, input of stage S+1
      * ``grad_input``       — input-grad of stage S, sent back to stage S-1
      * ``param_full``       — FSDP all-gathered full parameter (UNSHARD→compute)
      * ``param_shard``      — FSDP sharded param (RESHARD→next-iter UNSHARD)
      * ``grad_reduced``     — DP-reduced grad (REDUCE_GRAD→OPTIMIZER)
      * ``optimizer_state``  — updated param (OPTIMIZER→next iter)
      * ``dataloader_input`` — first-stage forward input (external)
      * ``kv_cache``         — inference cross-iteration KV cache
    """

    slot_id: str
    kind: str
    shape: tuple[int | str, ...] = ()
    dtype: str = ""
    volume_bytes: int = 0
    producer_action_id: str = ""           # action that produced this slot
    consumer_action_ids: list[str] = field(default_factory=list)
    src_stage: int = -1                    # P2P/collective source stage
    dst_stage: int = -1                    # destination stage
    mb_idx: int = -1                       # microbatch (-1 if not mb-specific)
    comm_primitive: str = ""               # p2p_send | allgather | reduce_scatter | allreduce | "" (local)
    is_local_transfer: bool = False        # V-schedule same-rank adjacent stage (set_local_*_input, no comm)
    src_exit_op: int = 0                   # L1 template exit op_id (producer side)
    dst_entry_op: int = 0                  # L1 template entry op_id (consumer side)


@dataclass
class ScheduleAction:
    """One scheduled unit: a compute chunk, a comm op, an FSDP/optimizer op.

    ``action_type`` ∈ {COMPUTE, UNSHARD, RESHARD, SEND_F, RECV_F, SEND_B,
    RECV_B, REDUCE_GRAD, OPTIMIZER, LR_SCHEDULER, OVERLAP_F_B, LOSS}.
    ``comp_type``/``template_ref`` are set only for COMPUTE (and OVERLAP_F_B
    sub-actions).  OVERLAP_F_B uses ``sub_actions`` to carry its F + B
    components (the plan keeps the composite, capture enriches the
    sub-actions); all other action types leave ``sub_actions`` None.
    """

    action_id: str
    rank: int
    stage: int                              # PP stage (-1 if non-stage action, e.g. OPTIMIZER)
    mb_idx: int                             # microbatch (-1 if not mb-specific)
    action_type: str
    comp_type: str = ""                     # COMPUTE: F / B / I / W / F_RECOMPUTE / OPTIMIZER
    template_ref: str = ""                  # COMPUTE: L1 step_template id (s{stage}_{comp_type})
    seq_idx: int = 0                        # execution order (from capture; plan-index fallback)
    consumes: list[str] = field(default_factory=list)   # DataSlot ids
    produces: list[str] = field(default_factory=list)   # DataSlot ids
    duration_est: float = 0.0
    sub_actions: list[ScheduleAction] | None = None     # OVERLAP_F_B
    # For comm/FSDP actions: the L0 OpNode id that implements this action
    # (the allgather/reduce_scatter/p2p synthetic op captured by comm_events).
    # Set by build_schedule_plan; lookup via SchedulePlan.find_op_node().
    # 0 = no captured L0 op (e.g. FSDP no-op when mesh size 1 -> is_noop=True).
    comm_op_id: int = 0
    # True when the plan action ran but produced no real comm (e.g. FSDP
    # unshard/reshard with a 1-size mesh = no collective). The action is
    # still recorded for schedule completeness, but there is no L0 op / no
    # data transfer to replay.
    is_noop: bool = False
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulePlan:
    """The L2 scheduling view: ordered actions + the DataSlots flowing
    between them.  Built by ``build_schedule_plan`` from
    ``pipeline_order_with_comms`` (runtime schedules) or the captured
    timeline (single-stage schedules), enriched with capture data."""

    plan_id: str
    workload_type: str                      # train | inference | eval
    step_templates: dict[str, StepGraph]    # L1 templates (COMPUTE actions reference these)
    actions: list[ScheduleAction]           # ordered (by seq_idx, then plan order)
    data_slots: dict[str, DataSlot]
    pp_degree: int = 1
    tp_degree: int = 1
    dp_degree: int = 1
    num_micro_batches: int = 1
    pipeline_schedule: str = "none"
    gradient_accumulation: int = 1
    annotations: dict[str, Any] = field(default_factory=dict)

    @property
    def action_map(self) -> dict[str, ScheduleAction]:
        """id -> action (rebuilt on demand; cheap enough for typical plan sizes)."""
        am: dict[str, ScheduleAction] = {}
        for a in self.actions:
            am[a.action_id] = a
            if a.sub_actions:
                for s in a.sub_actions:
                    am[s.action_id] = s
        return am

    def compute_actions(self) -> list[ScheduleAction]:
        """All COMPUTE actions (including OVERLAP_F_B sub-actions flattened)."""
        out: list[ScheduleAction] = []
        for a in self.actions:
            if a.action_type == "COMPUTE":
                out.append(a)
            elif a.action_type == "OVERLAP_F_B" and a.sub_actions:
                out.extend(s for s in a.sub_actions if s.action_type == "COMPUTE")
        return out

    def find_op_node(self, op_id: int) -> Any:
        """Locate the L0 ``OpNode`` for a captured op_id (e.g. an action's
        ``comm_op_id``) across all L1 step_templates. Returns the OpNode or
        None. This is the entry point for simulation replay: given an
        UNSHARD/RESHARD/SEND/RECV action, ``action.comm_op_id`` +
        ``find_op_node(comm_op_id)`` yields the actual op (shape, comm_bytes,
        flops, module_path, …) to replay."""
        if not op_id:
            return None
        for sg in self.step_templates.values():
            if op_id in sg.nodes:
                return sg.nodes[op_id]
        return None

    def find_template_for_op(self, op_id: int) -> StepGraph | None:
        """Which L1 template holds the given op_id (for replay navigation)."""
        if not op_id:
            return None
        for sg in self.step_templates.values():
            if op_id in sg.nodes:
                return sg
        return None

    def export_schedule_plan_csv(self, path: str) -> None:
        """One row per action: seq, action_type, stage, mb, comp_type,
        template_ref, consumes→slot_ids, produces→slot_ids, plus one row
        per DataSlot (shape/bytes/comm/local/producer/consumers)."""
        import csv
        import os

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        slots = self.data_slots
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "seq_idx", "action_id", "action_type", "stage", "mb_idx",
                "comp_type", "template_ref", "rank",
                "consumes", "produces", "sub_actions", "annotations",
            ])
            for a in sorted(self.actions, key=lambda x: x.seq_idx):
                w.writerow([
                    a.seq_idx, a.action_id, a.action_type, a.stage,
                    a.mb_idx if a.mb_idx >= 0 else "",
                    a.comp_type, a.template_ref, a.rank,
                    ";".join(a.consumes), ";".join(a.produces),
                    ";".join(s.action_id for s in (a.sub_actions or [])),
                    ";".join(f"{k}={v}" for k, v in a.annotations.items()),
                ])
            w.writerow(["# DataSlots", "", "", "", "", "", "", "", "", "", "", ""])
            w.writerow([
                "slot_id", "kind", "stage_src", "stage_dst", "mb_idx",
                "shape", "dtype", "bytes", "comm_primitive", "is_local",
                "producer_action", "consumer_actions", "src_exit_op", "dst_entry_op",
            ])
            for s in slots.values():
                w.writerow([
                    s.slot_id, s.kind, s.src_stage, s.dst_stage,
                    s.mb_idx if s.mb_idx >= 0 else "",
                    list(s.shape), s.dtype, s.volume_bytes,
                    s.comm_primitive or ("local" if s.is_local_transfer else ""),
                    int(s.is_local_transfer),
                    s.producer_action_id, ";".join(s.consumer_action_ids),
                    s.src_exit_op, s.dst_entry_op,
                ])
