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
from typing import Any, Iterator

from torchtitan_npu.simulator.capture.comm_events import CommEvent
from torchtitan_npu.simulator.capture.schedule_assemblers import SingleStageTraceAssembler, pp_transfer_id
from torchtitan_npu.simulator.capture.schedule_validation import validate_schedule_plan
from torchtitan_npu.simulator.ir.schedule_graph import DataPass, ScheduleGraph, StepInstance, TensorSlot, TimelineEntry
from torchtitan_npu.simulator.ir.schedule_plan import CommDetail, DataSlot, ScheduleAction, SchedulePlan
from torchtitan_npu.simulator.ir.step_graph import StepGraph
from torchtitan_npu.simulator.memory.records import FSDPResidencyEvent
from torchtitan_npu.simulator.rank_table import RankTable

_slot_counter = itertools.count()


def _slot_id(rank: int) -> str:
    return f"r{rank}_slot_{next(_slot_counter)}"


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
    fsdp_residency_events: list[FSDPResidencyEvent] | None = None,
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
    fsdp_residency_events = fsdp_residency_events or []
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
    # stage -> [allgather CommEvents]. RESHARD is a local full-parameter
    # release in PyTorch's lowered pipeline schedule; it is not a
    # reduce-scatter collective.
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
    for ev in comm_events:
        if ev.comm_layer != "L2":
            continue
        if ev.comm_primitive == "allgather" and _is_real_comm_event(ev):
            unshard_by_stage.setdefault(int(ev.p2p_stage) if ev.p2p_stage >= 0 else rank, []).append(ev)

    # --- V-shape same-rank detection -----------------------------------------
    stage_to_rank: dict[int, int] = {}
    if pp_schedule_obj is not None:
        s2r = getattr(pp_schedule_obj, "stage_index_to_group_rank", None)
        if isinstance(s2r, dict):
            stage_to_rank = {int(k): int(v) for k, v in s2r.items()}

    def iter_actions(source: list[ScheduleAction]) -> Iterator[ScheduleAction]:
        for action in source:
            yield action
            if action.sub_actions:
                yield from iter_actions(action.sub_actions)

    def same_rank(a: int, b: int) -> bool:
        if a not in stage_to_rank or b not in stage_to_rank:
            return False
        return stage_to_rank[a] == stage_to_rank[b]

    def find_compute(stage: int, mb: int, comp_type: str) -> ScheduleAction | None:
        for a in iter_actions(actions):
            if a.action_type == "COMPUTE" and a.stage == stage and a.mb_idx == mb and a.comp_type == comp_type:
                return a
        return None

    def find_action_by(stage: int, mb: int, action_type: str) -> ScheduleAction | None:
        """Locate a (non-compute) plan action by (stage, mb, action_type) —
        used to wire SEND_F/RECV_F/SEND_B/RECV_B to the DataSlot they
        transport and to attach their CommDetail."""
        for a in iter_actions(actions):
            if a.action_type == action_type and a.stage == stage and a.mb_idx == mb:
                return a
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
                action_type="OVERLAP_F_B", seq_idx=seq_hint, schedule_order=seq_hint, sub_actions=subs,
            )
        seq = seq_hint
        if action_type == "COMPUTE" and comp_type and stage >= 0 and mb >= 0:
            seq = tl_seq.get((stage, mb, comp_type), seq_hint)
        tmpl = f"s{stage}_{comp_type}" if (action_type == "COMPUTE" and comp_type) else ""
        action_id = next(_action_seq)
        return ScheduleAction(
            id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank, stage=stage, mb_idx=mb,
            action_type=action_type, comp_type=comp_type, template_ref=tmpl, seq_idx=seq,
            schedule_order=seq_hint,
        )

    # --- 1. action skeleton ---------------------------------------------------
    plan_obj = None
    is_single_stage_trace = False
    if pp_schedule_obj is not None:
        plan_obj = getattr(pp_schedule_obj, "pipeline_order_with_comms", None)
    if plan_obj and rank in plan_obj:
        # runtime schedule: lower the plan for this rank
        for i, a in enumerate(plan_obj[rank]):
            actions.append(map_action(a, i))
    elif timeline_events:
        is_single_stage_trace = True
        specs = SingleStageTraceAssembler(
            timeline_events=timeline_events,
            comm_events=comm_events,
            fsdp_residency_events=fsdp_residency_events,
            rank=rank,
        ).build()
        for schedule_order, spec in enumerate(specs):
            action_id = next(_action_seq)
            actions.append(ScheduleAction(
                id=f"{action_id}", action_id=f"r{rank}_a{action_id}", rank=rank,
                stage=spec.stage if spec.stage >= 0 else rank,
                mb_idx=spec.mb_idx,
                action_type=spec.action_type,
                comp_type=spec.comp_type,
                template_ref=spec.template_ref,
                seq_idx=spec.seq_idx,
                schedule_order=schedule_order,
                comm_op_id=spec.comm_op_id,
                comm=spec.comm,
                annotations=dict(spec.annotations),
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
                comp_type=ct, template_ref=tid, seq_idx=min_seq, schedule_order=min_seq,
            ))

    def _schedule_order(action: ScheduleAction) -> int:
        return action.schedule_order if action.schedule_order >= 0 else action.seq_idx

    def _action_position(action: ScheduleAction) -> tuple[int, int]:
        return _schedule_order(action), action.seq_idx

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
        for a in iter_actions(actions):
            if a.action_id == aid:
                return a
        return None

    # P2P forward_send: activation F(S) -> F(S+1)
    for ev in comm_events:
        d = ev.p2p_direction or ""
        if d not in {"forward_send", "backward_send"}:
            continue
        stage = int(ev.p2p_stage)
        mb = int(ev.p2p_mb_idx)
        transfer_id = ev.transfer_id or pp_transfer_id(ev)
        if "forward" in d:
            src_ct, dst_ct, dst_stage = "F", "F", stage + 1
            kind = "activation"
            send_at, recv_at = "SEND_F", "RECV_F"
        else:  # backward_send: grad_input from I/B(S) -> B(S-1)
            src_ct = "I"  # may be B for full backward; resolve below
            dst_ct, dst_stage, kind = "B", stage - 1, "grad_input"
            send_at, recv_at = "SEND_B", "RECV_B"
        prod = find_compute(stage, mb, src_ct) or find_compute(stage, mb, "B")
        cons = find_compute(dst_stage, mb, dst_ct)
        send_act = find_action_by(stage, mb, send_at)
        recv_act = find_action_by(dst_stage, mb, recv_at)
        slot_kind = (
            ("activation_local" if kind == "activation" else "grad_local")
            if is_single_stage_trace
            else kind
        )
        consumers = (
            [send_act.action_id]
            if is_single_stage_trace and send_act is not None
            else ([cons.action_id] if cons else [])
        )
        slot = DataSlot(
            slot_id=_slot_id(rank), kind=slot_kind,
            shape=ev.tensor_shape, dtype=ev.dtype, volume_bytes=ev.volume_bytes,
            producer_action_id=prod.action_id if prod else "",
            consumer_action_ids=consumers,
            src_stage=stage, dst_stage=dst_stage, mb_idx=mb,
            comm_primitive="p2p_send", is_local_transfer=False,
            src_exit_op=ev.src_exit_op, dst_entry_op=ev.dst_entry_op,
        )
        add_slot(slot)
        # Wire the SEND/RECV plan actions to this slot + attach a CommDetail
        # (direct data-pass-level field, no 2-hop lookup needed).
        cd_send = CommDetail(
            primitive="p2p_send", role="send", shape=ev.tensor_shape, dtype=ev.dtype,
            volume_bytes=ev.volume_bytes, src_stage=stage, dst_stage=dst_stage, mb_idx=mb,
            peer_rank=ev.p2p_peer_rank, comm_group_ranks=ev.comm_ranks,
            src_exit_op=ev.src_exit_op, dst_entry_op=ev.dst_entry_op,
            slot_id=slot.slot_id, comm_op_id=ev.op_id, transfer_id=transfer_id,
        )
        if send_act is not None:
            send_act.comm = cd_send
            target = send_act.consumes if is_single_stage_trace else send_act.produces
            if slot.slot_id not in target:
                target.append(slot.slot_id)
        if recv_act is not None:
            recv_act.comm = CommDetail(
                primitive="p2p_send", role="recv", shape=ev.tensor_shape, dtype=ev.dtype,
                volume_bytes=ev.volume_bytes, src_stage=stage, dst_stage=dst_stage, mb_idx=mb,
                peer_rank=ev.p2p_peer_rank, comm_group_ranks=ev.comm_ranks,
                src_exit_op=ev.src_exit_op, dst_entry_op=ev.dst_entry_op,
                slot_id=slot.slot_id, comm_op_id=ev.op_id, transfer_id=transfer_id,
            )
            if slot.slot_id not in recv_act.consumes:
                recv_act.consumes.append(slot.slot_id)

    # P2P *_recv (cross-rank receive side): the SEND section above only matched
    # *_send CommEvents and looked for a RECV action on stage+1 — but in PP the
    # RECV_F/RECV_B action lives on the RECEIVING rank (a different process), so
    # on this rank the SEND-side find_action_by(dst, mb, RECV_*) returns None
    # and the RECV action got no comm. Each *_recv CommEvent IS captured on the
    # receiving rank (patched_irecv records p2p_direction="*_recv",
    # p2p_stage=receiving stage, p2p_peer_rank=sender), so process it here to
    # attach the RECV action's CommDetail directly + wire a recv-side DataSlot
    # (producer=RECV action, consumer=the local COMPUTE that consumes it).
    for ev in comm_events:
        d = ev.p2p_direction or ""
        if d not in {"forward_recv", "backward_recv"}:
            continue
        recv_stage = int(ev.p2p_stage)
        mb = int(ev.p2p_mb_idx)
        transfer_id = ev.transfer_id or pp_transfer_id(ev)
        if "forward" in d:
            cons_ct, src_stage, kind, recv_at = "F", recv_stage - 1, "activation", "RECV_F"
        else:  # backward_recv: grad_input consumed by B(recv_stage, mb)
            cons_ct, src_stage, kind, recv_at = "B", recv_stage + 1, "grad_input", "RECV_B"
        recv_act = find_action_by(recv_stage, mb, recv_at)
        cons = find_compute(recv_stage, mb, cons_ct) or find_compute(recv_stage, mb, "B")
        # a RECV action receives the tensor and feeds the local COMPUTE; model
        # the recv-side slot with producer=RECV action, consumer=local COMPUTE.
        slot = DataSlot(
            slot_id=_slot_id(rank),
            kind=("activation_recv" if kind == "activation" else "grad_recv") if is_single_stage_trace else kind,
            shape=ev.tensor_shape, dtype=ev.dtype, volume_bytes=ev.volume_bytes,
            producer_action_id=recv_act.action_id if recv_act else "",
            consumer_action_ids=[cons.action_id] if cons else [],
            src_stage=src_stage, dst_stage=recv_stage, mb_idx=mb,
            comm_primitive="p2p_send", is_local_transfer=False,
            src_exit_op=ev.src_exit_op, dst_entry_op=ev.dst_entry_op,
        )
        add_slot(slot)
        if recv_act is not None:
            recv_act.comm = CommDetail(
                primitive="p2p_send", role="recv", shape=ev.tensor_shape, dtype=ev.dtype,
                volume_bytes=ev.volume_bytes, src_stage=src_stage, dst_stage=recv_stage,
                mb_idx=mb, peer_rank=ev.p2p_peer_rank, comm_group_ranks=ev.comm_ranks,
                src_exit_op=ev.src_exit_op, dst_entry_op=ev.dst_entry_op,
                slot_id=slot.slot_id, comm_op_id=ev.op_id,
                transfer_id=transfer_id,
            )
            if slot.slot_id not in recv_act.produces:
                recv_act.produces.append(slot.slot_id)

    # --- 3. DataSlots: V-schedule local transfers (synthesized, no CommEvent) --
    for a in list(iter_actions(actions)):
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
            slot_id=_slot_id(rank), kind="activation", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=a.action_id, consumer_action_ids=[cons.action_id],
            src_stage=s, dst_stage=s + 1, mb_idx=mb,
            comm_primitive="", is_local_transfer=True,
        )
        add_slot(slot)
    # local backward: I/B(S) -> B(S-1) same rank
    for a in list(iter_actions(actions)):
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
            slot_id=_slot_id(rank), kind="grad_input", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=a.action_id, consumer_action_ids=[cons.action_id],
            src_stage=s, dst_stage=s - 1, mb_idx=mb,
            comm_primitive="", is_local_transfer=True,
        )
        add_slot(slot)

    if is_single_stage_trace:
        pp_degree = rank_table.dim_degrees.get("pp", 1)
        forward_actions = [
            action
            for action in iter_actions(actions)
            if action.action_type == "COMPUTE" and action.comp_type == "F"
        ]
        for forward in forward_actions:
            backward = find_compute(forward.stage, forward.mb_idx, "B")
            if backward is None:
                raise RuntimeError(
                    f"1F1B forward action {forward.action_id} has no matching backward "
                    f"for stage {forward.stage}, microbatch {forward.mb_idx}"
                )
            add_slot(DataSlot(
                slot_id=_slot_id(rank), kind="forward_state", volume_bytes=0,
                producer_action_id=forward.action_id,
                consumer_action_ids=[backward.action_id],
                src_stage=forward.stage, dst_stage=forward.stage, mb_idx=forward.mb_idx,
            ))
            if forward.stage == 0 and not any(
                data_slots[slot_id].kind == "activation_recv"
                for slot_id in forward.consumes
            ):
                add_slot(DataSlot(
                    slot_id=_slot_id(rank), kind="dataloader_input", volume_bytes=0,
                    producer_action_id="", consumer_action_ids=[forward.action_id],
                    src_stage=-1, dst_stage=forward.stage, mb_idx=forward.mb_idx,
                    external=True,
                ))
            if backward.stage == pp_degree - 1 and not any(
                data_slots[slot_id].kind == "grad_recv"
                for slot_id in backward.consumes
            ):
                add_slot(DataSlot(
                    slot_id=_slot_id(rank), kind="loss_grad", volume_bytes=0,
                    producer_action_id="", consumer_action_ids=[backward.action_id],
                    src_stage=-1, dst_stage=backward.stage, mb_idx=backward.mb_idx,
                    external=True,
                ))

    # --- 4. DataSlots: FSDP unshard/reshard -----------------------------------
    # UNSHARD order-matches the next captured allgather on the same stage.
    # RESHARD performs no collective: it releases the full parameter after
    # the preceding compute, so it receives a zero-byte control dependency.
    unshard_iters: dict[tuple[int, str], int] = {}
    unshard_event_by_residency: dict[tuple[int, str, str], CommEvent] = {}

    fsdp_degree = max(
        rank_table.dim_degrees.get("dp_shard", 1),
        rank_table.dim_degrees.get("fsdp", 1),
        rank_table.dim_degrees.get("efsdp", 1),
    )

    fsdp_stage_usage: dict[int, bool] = {}
    for stage in getattr(pp_schedule_obj, "_stages", ()) or ():
        stage_index = getattr(stage, "stage_index", None)
        submod = getattr(stage, "submod", None)
        if stage_index is not None:
            fsdp_stage_usage[int(stage_index)] = bool(
                submod is not None and hasattr(submod, "unshard") and hasattr(submod, "reshard")
            )

    def _mark_fsdp_noop_or_raise(action: ScheduleAction, stage: int) -> None:
        stage_uses_fsdp = fsdp_stage_usage.get(stage)
        if stage_uses_fsdp is not False and fsdp_degree > 1:
            raise RuntimeError(
                f"{action.action_type} action {action.action_id} on stage {stage} "
                f"requires FSDP communication (shard degree {fsdp_degree}) but no "
                "captured CommEvent was matched"
            )
        action.is_noop = True
        action.comm = CommDetail(
            src_stage=stage,
            dst_stage=stage,
            mb_idx=-1,
            is_noop=True,
        )

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
        group_id = str(a.annotations.get("fsdp_group_id", ""))
        comp_type = str(a.annotations.get("residency_comp_type", ""))
        residency_key = (s, group_id, comp_type)
        ev = unshard_event_by_residency.get(residency_key) if group_id else None
        if ev is None:
            context_events = [
                event
                for event in unshard_by_stage.get(s, [])
                if not comp_type or event.comp_type == comp_type
            ]
            evs = context_events or unshard_by_stage.get(s, [])
            context_key = (s, comp_type if context_events else "")
            idx = unshard_iters.get(context_key, 0)
            ev = evs[idx] if idx < len(evs) else None
            if ev is not None:
                unshard_iters[context_key] = idx + 1
                if group_id:
                    # MB1+ L0 capture is folded. Reuse the MB0 communication
                    # template while keeping each residency transition distinct.
                    unshard_event_by_residency[residency_key] = ev
        if ev is not None:
            a.comm_op_id = ev.op_id
            a.template_ref = _template_holding(ev.op_id)
            shape, dtype, bytes_ = ev.tensor_shape, ev.dtype, ev.volume_bytes
            comm = "allgather"
            src_exit, dst_entry = ev.src_exit_op, ev.dst_entry_op
        else:
            _mark_fsdp_noop_or_raise(a, s)
            continue
        # consumer = next COMPUTE on that stage after this unshard
        cons = None
        for ca in iter_actions(actions):
            if (ca.action_type == "COMPUTE" and ca.stage == s
                    and _schedule_order(ca) >= _schedule_order(a) and ca is not a):
                if cons is None or _action_position(ca) < _action_position(cons):
                    cons = ca
        if cons is None:
            raise RuntimeError(
                f"UNSHARD action {a.action_id} on stage {s} has real communication "
                "but no following compute consumer"
            )
        slot = DataSlot(
            slot_id=_slot_id(rank), kind="param_full", shape=shape, dtype=dtype, volume_bytes=bytes_,
            producer_action_id=a.action_id,
            consumer_action_ids=[cons.action_id] if cons else [],
            src_stage=s, dst_stage=s, comm_primitive=comm,
            src_exit_op=src_exit, dst_entry_op=dst_entry,
        )
        add_slot(slot)
        a.comm = CommDetail(
            primitive=comm, role="collective", shape=shape, dtype=dtype,
            volume_bytes=bytes_, src_stage=s, dst_stage=s, mb_idx=-1,
            comm_group_ranks=ev.comm_ranks if ev else [],
            src_exit_op=src_exit, dst_entry_op=dst_entry,
            slot_id=slot.slot_id, comm_op_id=a.comm_op_id, is_noop=a.is_noop,
        )
    active_unshards: dict[tuple[int, str], ScheduleAction] = {}
    for a in actions:
        s = a.stage if a.stage >= 0 else rank
        group_id = str(a.annotations.get("fsdp_group_id", "__stage__"))
        residency_key = (s, group_id)
        if a.action_type == "UNSHARD":
            if residency_key in active_unshards:
                raise RuntimeError(
                    f"UNSHARD action {a.action_id} duplicates active FSDP residency "
                    f"for stage {s}, group {group_id}"
                )
            active_unshards[residency_key] = a
            continue
        if a.action_type != "RESHARD":
            continue
        unshard = active_unshards.pop(residency_key, None)
        if unshard is None:
            raise RuntimeError(
                f"RESHARD action {a.action_id} on stage {s}, group {group_id} "
                "has no active preceding UNSHARD"
            )
        if unshard.is_noop:
            a.is_noop = True
            a.comm = CommDetail(src_stage=s, dst_stage=s, mb_idx=-1, is_noop=True)
            continue

        # Reshard may follow forward or backward depending on FSDP policy.
        prod = None
        for ca in iter_actions(actions):
            if (ca.action_type == "COMPUTE" and ca.stage == s
                    and ca.comp_type in ("F", "B", "I", "W")
                    and _schedule_order(ca) >= _schedule_order(unshard)
                    and _schedule_order(ca) <= _schedule_order(a)):
                if prod is None or _action_position(ca) > _action_position(prod):
                    prod = ca
        if prod is None:
            raise RuntimeError(
                f"RESHARD action {a.action_id} on stage {s} has an active full parameter "
                "but no preceding compute producer"
            )
        slot = DataSlot(
            slot_id=_slot_id(rank), kind="control", volume_bytes=0,
            producer_action_id=prod.action_id,
            consumer_action_ids=[a.action_id],
            src_stage=s, dst_stage=s, mb_idx=-1,
        )
        add_slot(slot)

    if active_unshards:
        unresolved = ", ".join(
            f"stage={stage}/group={group_id}/action={action.action_id}"
            for (stage, group_id), action in active_unshards.items()
        )
        raise RuntimeError(f"UNSHARD actions without matching RESHARD: {unresolved}")

    # --- 5. REDUCE_GRAD -> OPTIMIZER (grad_reduced) ---------------------------
    reduce_actions = [a for a in actions if a.action_type == "REDUCE_GRAD"]
    grad_reduced_slots: list[str] = []
    for a in reduce_actions:
        if is_single_stage_trace:
            producer = None
            for candidate in iter_actions(actions):
                if (candidate.action_type == "COMPUTE" and candidate.comp_type == "B"
                        and candidate.stage == a.stage
                        and _schedule_order(candidate) <= _schedule_order(a)):
                    if producer is None or _action_position(candidate) > _action_position(producer):
                        producer = candidate
            if producer is None:
                raise RuntimeError(
                    f"REDUCE_GRAD action {a.action_id} on stage {a.stage} "
                    "has no preceding backward producer"
                )
            add_slot(DataSlot(
                slot_id=_slot_id(rank), kind="grad_local", volume_bytes=0,
                producer_action_id=producer.action_id,
                consumer_action_ids=[a.action_id],
                src_stage=a.stage, dst_stage=a.stage, mb_idx=a.mb_idx,
            ))
        slot = DataSlot(
            slot_id=_slot_id(rank), kind="grad_reduced", shape=(), dtype="",
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
                schedule_order=max((_schedule_order(a) for a in actions), default=-1) + 1,
            )
            actions.append(act)
        opt_action = next(a for a in actions if a.action_type == "OPTIMIZER")
        for sid in grad_reduced_slots:
            data_slots[sid].consumer_action_ids.append(opt_action.action_id)
            if sid not in opt_action.consumes:
                opt_action.consumes.append(sid)
        if is_single_stage_trace and not grad_reduced_slots:
            for backward in [
                action
                for action in iter_actions(actions)
                if action.action_type == "COMPUTE" and action.comp_type == "B"
            ]:
                add_slot(DataSlot(
                    slot_id=_slot_id(rank), kind="grad_local", volume_bytes=0,
                    producer_action_id=backward.action_id,
                    consumer_action_ids=[opt_action.action_id],
                    src_stage=backward.stage, dst_stage=backward.stage, mb_idx=backward.mb_idx,
                ))

    # --- assemble -------------------------------------------------------------
    actions.sort(key=_schedule_order)
    dp_degree = rank_table.dim_degrees.get("dp_replicate", 1) * rank_table.dim_degrees.get(
        "fsdp", rank_table.dim_degrees.get("dp_shard", 1)
    )
    from collections import Counter as _Ctr
    comm_summary = dict(_Ctr(
        (ev.comm_primitive, ev.comm_layer, ev.p2p_stage) for ev in comm_events
    ))
    plan = SchedulePlan(
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
        annotations={
            "rank_table": rank_table.to_dict(),
            "comm_events_summary": comm_summary,
            "assembler": "single_stage_trace" if is_single_stage_trace else "runtime_or_non_pp",
        },
    )
    validate_schedule_plan(plan, strict_1f1b=is_single_stage_trace)
    return plan


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
