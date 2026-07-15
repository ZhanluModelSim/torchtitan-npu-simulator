# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assembles the L2 ScheduleGraph from captured L1 templates, communication
events, and timeline events.  All fields come from capture — no RankTable
traversal, no all-to-all expansion, no P2P anchor inference.

See docs/design/schedule-capture-design.md for the design rationale."""

from __future__ import annotations

import itertools
import uuid
from typing import Any

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph, StepInstance, TensorSlot, TimelineEntry
from torchtitan_npu.simulator.ir.schedule_plan import DataSlot, ScheduleAction, SchedulePlan
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.rank_table import RankTable

_slot_counter = itertools.count()


def _slot_id() -> str:
    return f"slot_{next(_slot_counter)}"


# Map torch._ComputationType.value -> (action_type, comp_type)
_CT_MAP: dict[str, tuple[str, str]] = {
    "F": ("COMPUTE", "F"),
    "B": ("COMPUTE", "B"),
    "I": ("COMPUTE", "I"),
    "W": ("COMPUTE", "W"),
    "UNSHARD": ("UNSHARD", ""),
    "RESHARD": ("RESHARD", ""),
    "SEND_F": ("SEND_F", ""),
    "RECV_F": ("RECV_F", ""),
    "SEND_B": ("SEND_B", ""),
    "RECV_B": ("RECV_B", ""),
    "REDUCE_GRAD": ("REDUCE_GRAD", ""),
    "OVERLAP_F_B": ("OVERLAP_F_B", ""),
}


def build_schedule_plan(
    *,
    step_templates: dict[str, StepGraph],
    rank_table: RankTable,
    comm_events: list[CommEvent],
    timeline_events: list[dict] | None = None,
    pp_schedule_obj: Any | None = None,
    pipeline_schedule: str = "none",
    num_micro_batches: int = 1,
    gradient_accumulation: int = 1,
    rank: int = 0,
) -> SchedulePlan:
    """Build the structured L2 SchedulePlan: an ordered action list +
    the DataSlots flowing between actions.

    The action skeleton comes from ``pp_schedule_obj.pipeline_order_with_comms``
    (runtime zero-bubble schedules: ZBV/DualPipe/Interleaved/LoopedBFS) — the
    lowered plan that already contains F/B/I/W + UNSHARD/RESHARD/SEND/RECV/
    REDUCE_GRAD + OVERLAP_F_B in deadlock-safe order.  Single-stage
    schedules (1F1B/GPipe) have no such plan; their skeleton is synthesized
    from the captured timeline events.  Non-PP steps fall back to one action
    per captured L1 template.

    Capture enriches each action: COMPUTE seq_idx/template_ref from the
    timeline, P2P/FSDP DataSlot shapes from CommEvents, and same-rank
    adjacent-stage local transfers are synthesized (V-schedule
    ``set_local_*_input`` produces no comm event).  OPTIMIZER/LR_SCHEDULER
    are appended after the plan from the captured optimizer-phase nodes.
    """
    timeline_events = timeline_events or []
    actions: list[ScheduleAction] = []
    data_slots: dict[str, DataSlot] = {}
    _action_seq = itertools.count()

    # --- lookups from capture -------------------------------------------------
    # (stage, mb, comp_type) -> seq_idx  (from timeline: forward/backward_*_one_chunk)
    tl_seq: dict[tuple[int, int, str], int] = {}
    for ev in timeline_events:
        stage = int(ev.get("pp_stage", -1))
        mb = int(ev.get("pp_mb_idx", -1))
        ct = str(ev.get("comp_type", "") or "")
        if stage >= 0 and mb >= 0 and ct:
            tl_seq[(stage, mb, ct)] = int(ev.get("seq_idx", 0))
    # (stage, mb, direction_base) -> CommEvent  (P2P)
    p2p_by_key: dict[tuple[int, int, str], CommEvent] = {}
    for ev in comm_events:
        d = ev.p2p_direction or ""
        if not d:
            continue
        base = d.replace("_send", "").replace("_recv", "")  # "forward" / "backward"
        p2p_by_key[(int(ev.p2p_stage), int(ev.p2p_mb_idx), base)] = ev
    # stage -> [allgather CommEvents], stage -> [reduce_scatter CommEvents].
    # Only keep CommEvents whose op_id resolves to a real `comm.*` L0 op — the
    # allgather/reduce_scatter fired during DYNAMIC-mode metadata inference
    # (framework shape-inference forward) also records a CommEvent but its
    # op_id is stale (pointing at the last recorded aten op, since L0 capture
    # is skipped during inference), which would mis-pair UNSHARD/RESHARD plan
    # actions. Filtering by "op_id resolves to a comm.* op" drops those.
    def _is_real_comm_event(ev: CommEvent) -> bool:
        if not ev.op_id:
            return False
        for sg in step_templates.values():
            n = sg.nodes.get(ev.op_id)
            if n is not None:
                return n.annotations.get("raw_op_type", "").startswith("comm.")
        return False

    unshard_by_stage: dict[int, list[CommEvent]] = {}
    reshard_by_stage: dict[int, list[CommEvent]] = {}
    for ev in comm_events:
        if ev.comm_layer != "L2":
            continue
        if ev.comm_primitive == "allgather" and _is_real_comm_event(ev):
            unshard_by_stage.setdefault(int(ev.p2p_stage) if ev.p2p_stage >= 0 else rank, []).append(ev)
        elif ev.comm_primitive == "reduce_scatter" and _is_real_comm_event(ev):
            reshard_by_stage.setdefault(int(ev.p2p_stage) if ev.p2p_stage >= 0 else rank, []).append(ev)

    # --- V-shape same-rank detection -----------------------------------------
    stage_to_rank: dict[int, int] = {}
    if pp_schedule_obj is not None:
        s2r = getattr(pp_schedule_obj, "stage_index_to_group_rank", None)
        if isinstance(s2r, dict):
            stage_to_rank = {int(k): int(v) for k, v in s2r.items()}

    def same_rank(a: int, b: int) -> bool:
        if a not in stage_to_rank or b not in stage_to_rank:
            return False
        return stage_to_rank[a] == stage_to_rank[b]

    def find_compute(stage: int, mb: int, comp_type: str) -> ScheduleAction | None:
        for a in actions:
            if a.action_type == "COMPUTE" and a.stage == stage and a.mb_idx == mb and a.comp_type == comp_type:
                return a
            if a.action_type == "OVERLAP_F_B" and a.sub_actions:
                for s in a.sub_actions:
                    if s.action_type == "COMPUTE" and s.stage == stage and s.mb_idx == mb and s.comp_type == comp_type:
                        return s
        return None

    def find_template_exit_shape(stage: int, comp_type: str) -> tuple[tuple, str, int]:
        """Best-effort (shape, dtype, bytes) of a template's exit tensor."""
        sg = step_templates.get(f"s{stage}_{comp_type}")
        if not sg or not sg.nodes:
            return (), "", 0
        from torchtitan_npu.simulator.capture.tensor_utils import tensor_volume_bytes
        # pick the last topological exit node's first output
        for op_id in reversed(list(sg.nodes.keys())):
            n = sg.nodes[op_id]
            if n.outputs:
                m = n.outputs[0]
                return tuple(m.shape), str(m.dtype), tensor_volume_bytes(tuple(m.shape), str(m.dtype))
        return (), "", 0

    # --- map a torch _Action -> ScheduleAction --------------------------------
    def map_action(a: Any, seq_hint: int) -> ScheduleAction:
        ct_val = getattr(a, "computation_type", None)
        ct = getattr(ct_val, "value", str(ct_val))
        action_type, comp_type = _CT_MAP.get(ct, ("COMPUTE", ""))
        stage = int(getattr(a, "stage_index", -1))
        mb = getattr(a, "microbatch_index", None)
        mb = int(mb) if mb is not None else -1
        if action_type == "OVERLAP_F_B" and getattr(a, "sub_actions", None):
            subs = [map_action(s, seq_hint) for s in a.sub_actions]
            action_id = next(_action_seq)
            return ScheduleAction(
                id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank, stage=-1, mb_idx=-1,
                action_type="OVERLAP_F_B", seq_idx=seq_hint, sub_actions=subs,
            )
        seq = seq_hint
        if action_type == "COMPUTE" and comp_type and stage >= 0 and mb >= 0:
            seq = tl_seq.get((stage, mb, comp_type), seq_hint)
        tmpl = f"s{stage}_{comp_type}" if (action_type == "COMPUTE" and comp_type) else ""
        action_id = next(_action_seq)
        return ScheduleAction(
            id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank, stage=stage, mb_idx=mb,
            action_type=action_type, comp_type=comp_type, template_ref=tmpl, seq_idx=seq,
        )

    # --- 1. action skeleton ---------------------------------------------------
    plan_obj = None
    if pp_schedule_obj is not None:
        plan_obj = getattr(pp_schedule_obj, "pipeline_order_with_comms", None)
    if plan_obj and rank in plan_obj:
        # runtime schedule: lower the plan for this rank
        for i, a in enumerate(plan_obj[rank]):
            actions.append(map_action(a, i))
    elif timeline_events:
        # single-stage PP (1F1B/GPipe): synthesize COMPUTE actions from timeline
        for ev in sorted(timeline_events, key=lambda e: e.get("seq_idx", 0)):
            ct = str(ev.get("comp_type", "") or "")
            if not ct:
                act = ev.get("action", "")
                ct = "F" if "forward" in act else ("W" if "weight" in act else "B")
            stage = int(ev.get("pp_stage", rank))
            mb = int(ev.get("pp_mb_idx", 0))
            action_id = next(_action_seq)
            actions.append(ScheduleAction(
                id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank, stage=stage, mb_idx=mb,
                action_type="COMPUTE", comp_type=ct, template_ref=f"s{stage}_{ct}",
                seq_idx=int(ev.get("seq_idx", 0)),
            ))
        # comm actions (P2P + FSDP) from CommEvents
        for ev in comm_events:
            d = ev.p2p_direction or ""
            if d:
                base = d.replace("_send", "").replace("_recv", "")
                atype = ("SEND_F" if "send" in d and base == "forward" else
                         "RECV_F" if "recv" in d and base == "forward" else
                         "SEND_B" if "send" in d else "RECV_B")
            elif ev.comm_primitive == "allgather":
                atype = "UNSHARD"
            elif ev.comm_primitive == "reduce_scatter":
                atype = "RESHARD"
            else:
                continue
            action_id = next(_action_seq)
            actions.append(ScheduleAction(
                id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank,
                stage=int(ev.p2p_stage) if ev.p2p_stage >= 0 else rank,
                mb_idx=int(ev.p2p_mb_idx) if ev.p2p_mb_idx >= 0 else -1,
                action_type=atype, seq_idx=int(ev.seq_idx),
            ))
    else:
        # non-PP: one action per captured template, ordered F < B < OPTIMIZER
        order = {"F": 0, "B": 1, "OPTIMIZER": 2}
        for tid in sorted(step_templates, key=lambda t: order.get(step_templates[t].step_type, 9)):
            sg = step_templates[tid]
            ct = sg.step_type
            min_seq = min((n.seq_idx for n in sg.nodes.values()), default=0)
            action_id = next(_action_seq)
            actions.append(ScheduleAction(
                id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank, stage=rank, mb_idx=0,
                action_type="COMPUTE" if ct != "OPTIMIZER" else "OPTIMIZER",
                comp_type=ct, template_ref=tid, seq_idx=min_seq,
            ))

    # --- 2. DataSlots: P2P activations / grad_inputs -------------------------
    def add_slot(slot: DataSlot) -> None:
        data_slots[slot.slot_id] = slot
        # wire producer/consumer action lists
        if slot.producer_action_id:
            pa = _lookup(slot.producer_action_id)
            if pa and slot.slot_id not in pa.produces:
                pa.produces.append(slot.slot_id)
        for cid in slot.consumer_action_ids:
            ca = _lookup(cid)
            if ca and slot.slot_id not in ca.consumes:
                ca.consumes.append(slot.slot_id)

    def _lookup(aid: str) -> ScheduleAction | None:
        for a in actions:
            if a.action_id == aid:
                return a
            if a.sub_actions:
                for s in a.sub_actions:
                    if s.action_id == aid:
                        return s
        return None

    # P2P forward_send: activation F(S) -> F(S+1)
    for ev in comm_events:
        d = ev.p2p_direction or ""
        if "send" not in d:
            continue
        stage = int(ev.p2p_stage)
        mb = int(ev.p2p_mb_idx)
        if "forward" in d:
            src_ct, dst_ct, dst_stage = "F", "F", stage + 1
            kind = "activation"
        else:  # backward_send: grad_input from I/B(S) -> B(S-1)
            src_ct = "I"  # may be B for full backward; resolve below
            dst_ct, dst_stage, kind = "B", stage - 1, "grad_input"
        prod = find_compute(stage, mb, src_ct) or find_compute(stage, mb, "B")
        cons = find_compute(dst_stage, mb, dst_ct)
        slot = DataSlot(
            slot_id=_slot_id(), kind=kind,
            shape=ev.tensor_shape, dtype=ev.dtype, volume_bytes=ev.volume_bytes,
            producer_action_id=prod.action_id if prod else "",
            consumer_action_ids=[cons.action_id] if cons else [],
            src_stage=stage, dst_stage=dst_stage, mb_idx=mb,
            comm_primitive="p2p_send", is_local_transfer=False,
            src_exit_op=ev.src_exit_op, dst_entry_op=ev.dst_entry_op,
        )
        add_slot(slot)

    # --- 3. DataSlots: V-schedule local transfers (synthesized, no CommEvent) --
    for a in list(actions):
        if a.action_type != "COMPUTE" or a.comp_type != "F" or a.stage < 0 or a.mb_idx < 0:
            continue
        s, mb = a.stage, a.mb_idx
        if not same_rank(s, s + 1):
            continue  # cross-rank: already covered by P2P SEND_F
        cons = find_compute(s + 1, mb, "F")
        if not cons:
            continue
        shape, dtype, bytes_ = find_template_exit_shape(s, "F")
        slot = DataSlot(
            slot_id=_slot_id(), kind="activation", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=a.action_id, consumer_action_ids=[cons.action_id],
            src_stage=s, dst_stage=s + 1, mb_idx=mb,
            comm_primitive="", is_local_transfer=True,
        )
        add_slot(slot)
    # local backward: I/B(S) -> B(S-1) same rank
    for a in list(actions):
        if a.action_type != "COMPUTE" or a.comp_type not in ("I", "B") or a.stage < 0 or a.mb_idx < 0:
            continue
        s, mb = a.stage, a.mb_idx
        if not same_rank(s, s - 1):
            continue
        cons = find_compute(s - 1, mb, "B")
        if not cons:
            continue
        shape, dtype, bytes_ = find_template_exit_shape(s, a.comp_type)
        slot = DataSlot(
            slot_id=_slot_id(), kind="grad_input", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=a.action_id, consumer_action_ids=[cons.action_id],
            src_stage=s, dst_stage=s - 1, mb_idx=mb,
            comm_primitive="", is_local_transfer=True,
        )
        add_slot(slot)

    # --- 4. DataSlots: FSDP unshard/reshard (param_full / param_shard) --------
    # Order-match each UNSHARD/RESHARD plan action to the next captured
    # allgather/reduce_scatter CommEvent on that stage (plan order ~= exec
    # order ~= CommEvent capture order, so a per-stage FIFO pop is correct).
    # This links the action to its L0 comm op (comm_op_id) + the L1 template
    # holding it (template_ref), and fills the DataSlot shape/bytes. When no
    # CommEvent exists (FSDP no-op, e.g. mesh size 1) the action is marked
    # is_noop=True (schedule-completeness, no replay data).
    unshard_iters: dict[int, int] = {s: 0 for s in unshard_by_stage}
    reshard_iters: dict[int, int] = {s: 0 for s in reshard_by_stage}

    def _template_holding(op_id: int) -> str:
        if not op_id:
            return ""
        for tid, sg in step_templates.items():
            if op_id in sg.nodes:
                return tid
        return ""

    for a in actions:
        if a.action_type != "UNSHARD":
            continue
        s = a.stage if a.stage >= 0 else rank
        evs = unshard_by_stage.get(s, [])
        idx = unshard_iters.get(s, 0)
        ev = evs[idx] if idx < len(evs) else None
        if ev is not None:
            unshard_iters[s] = idx + 1
            a.comm_op_id = ev.op_id
            a.template_ref = _template_holding(ev.op_id)
            shape, dtype, bytes_ = ev.tensor_shape, ev.dtype, ev.volume_bytes
            comm = "allgather"
            src_exit, dst_entry = ev.src_exit_op, ev.dst_entry_op
        else:
            a.is_noop = True
            shape, dtype, bytes_, comm, src_exit, dst_entry = (), "", 0, "", 0, 0
        # consumer = next COMPUTE on that stage after this unshard
        cons = None
        for ca in actions:
            if (ca.action_type == "COMPUTE" and ca.stage == s
                    and ca.seq_idx >= a.seq_idx and ca is not a):
                if cons is None or ca.seq_idx < cons.seq_idx:
                    cons = ca
        slot = DataSlot(
            slot_id=_slot_id(), kind="param_full", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=a.action_id,
            consumer_action_ids=[cons.action_id] if cons else [],
            src_stage=s, dst_stage=s, comm_primitive=comm,
            src_exit_op=src_exit, dst_entry_op=dst_entry,
        )
        add_slot(slot)
    for a in actions:
        if a.action_type != "RESHARD":
            continue
        s = a.stage if a.stage >= 0 else rank
        evs = reshard_by_stage.get(s, [])
        idx = reshard_iters.get(s, 0)
        ev = evs[idx] if idx < len(evs) else None
        if ev is not None:
            reshard_iters[s] = idx + 1
            a.comm_op_id = ev.op_id
            a.template_ref = _template_holding(ev.op_id)
            shape, dtype, bytes_ = ev.tensor_shape, ev.dtype, ev.volume_bytes
            comm = "reduce_scatter"
        else:
            a.is_noop = True
            shape, dtype, bytes_, comm = (), "", 0, ""
        # producer = preceding backward COMPUTE on that stage
        prod = None
        for ca in actions:
            if (ca.action_type == "COMPUTE" and ca.stage == s
                    and ca.comp_type in ("B", "I", "W") and ca.seq_idx <= a.seq_idx):
                if prod is None or ca.seq_idx > prod.seq_idx:
                    prod = ca
        slot = DataSlot(
            slot_id=_slot_id(), kind="param_shard", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=prod.action_id if prod else "",
            consumer_action_ids=[a.action_id],
            src_stage=s, dst_stage=s, mb_idx=-1,
            comm_primitive=comm,
        )
        add_slot(slot)

    # --- 5. REDUCE_GRAD -> OPTIMIZER (grad_reduced) ---------------------------
    reduce_actions = [a for a in actions if a.action_type == "REDUCE_GRAD"]
    grad_reduced_slots: list[str] = []
    for a in reduce_actions:
        slot = DataSlot(
            slot_id=_slot_id(), kind="grad_reduced", shape=(), dtype="",
            producer_action_id=a.action_id, consumer_action_ids=[],
            src_stage=a.stage if a.stage >= 0 else rank, dst_stage=a.stage if a.stage >= 0 else rank,
        )
        add_slot(slot)
        grad_reduced_slots.append(slot.slot_id)

    # --- 6. OPTIMIZER action (from captured optimizer L1 template) -----------
    opt_tmpl = step_templates.get(f"s{rank}_OPTIMIZER") or step_templates.get("s-1_OPTIMIZER")
    if opt_tmpl:
        # already added in non-PP path; for PP add it now at the tail
        if not any(a.action_type == "OPTIMIZER" for a in actions):
            min_seq = min((n.seq_idx for n in opt_tmpl.nodes.values()), default=(len(actions)))
            action_id = next(_action_seq)
            act = ScheduleAction(
                id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank, stage=rank, mb_idx=-1,
                action_type="OPTIMIZER", comp_type="OPTIMIZER",
                template_ref=f"s{rank}_OPTIMIZER", seq_idx=min_seq,
            )
            actions.append(act)
        opt_action = next(a for a in actions if a.action_type == "OPTIMIZER")
        for sid in grad_reduced_slots:
            data_slots[sid].consumer_action_ids.append(opt_action.action_id)
            if sid not in opt_action.consumes:
                opt_action.consumes.append(sid)

    # --- assemble -------------------------------------------------------------
    actions.sort(key=lambda a: a.seq_idx)
    dp_degree = rank_table.dim_degrees.get("dp_replicate", 1) * rank_table.dim_degrees.get(
        "fsdp", rank_table.dim_degrees.get("dp_shard", 1)
    )
    from collections import Counter as _Ctr
    comm_summary = dict(_Ctr(
        (ev.comm_primitive, ev.comm_layer, ev.p2p_stage) for ev in comm_events
    ))
    return SchedulePlan(
        plan_id=uuid.uuid4().hex[:12],
        workload_type="train",
        step_templates=step_templates,
        actions=actions,
        data_slots=data_slots,
        pp_degree=rank_table.dim_degrees.get("pp", 1),
        tp_degree=rank_table.dim_degrees.get("tp", 1),
        dp_degree=dp_degree,
        num_micro_batches=num_micro_batches,
        pipeline_schedule=pipeline_schedule,
        gradient_accumulation=gradient_accumulation,
        annotations={"rank_table": rank_table.to_dict(), "comm_events_summary": comm_summary},
    )


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
