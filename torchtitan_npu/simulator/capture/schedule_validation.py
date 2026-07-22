# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SchedulePlan invariants and a structural 1F1B readiness replay."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from torchtitan_npu.simulator.ir.schedule_plan import ScheduleAction, SchedulePlan


_SEND_TYPES = {"SEND_F", "SEND_B"}
_RECV_TYPES = {"RECV_F", "RECV_B"}
_P2P_TYPES = _SEND_TYPES | _RECV_TYPES


def _flatten(actions: Iterable[ScheduleAction]) -> list[ScheduleAction]:
    flattened: list[ScheduleAction] = []
    for action in actions:
        flattened.append(action)
        if action.sub_actions:
            flattened.extend(_flatten(action.sub_actions))
    return flattened


def validate_schedule_plan(plan: SchedulePlan, *, strict_1f1b: bool = False) -> None:
    actions = _flatten(plan.actions)
    action_map: dict[str, ScheduleAction] = {}
    for action in actions:
        if action.action_id in action_map:
            raise RuntimeError(f"duplicate ScheduleAction id: {action.action_id}")
        action_map[action.action_id] = action

    for slot_id, slot in plan.data_slots.items():
        if slot.slot_id != slot_id:
            raise RuntimeError(f"DataSlot key/id mismatch: key={slot_id}, id={slot.slot_id}")
        if not slot.external and not slot.producer_action_id and slot.consumer_action_ids:
            raise RuntimeError(f"internal consumed slot {slot_id} has no producer")
        if slot.producer_action_id:
            producer = action_map.get(slot.producer_action_id)
            if producer is None:
                raise RuntimeError(
                    f"slot {slot_id} references missing producer {slot.producer_action_id}"
                )
            if slot_id not in producer.produces:
                raise RuntimeError(
                    f"slot {slot_id} producer {producer.action_id} lacks reciprocal produces reference"
                )
        for consumer_id in slot.consumer_action_ids:
            consumer = action_map.get(consumer_id)
            if consumer is None:
                raise RuntimeError(f"slot {slot_id} references missing consumer {consumer_id}")
            if slot_id not in consumer.consumes:
                raise RuntimeError(
                    f"slot {slot_id} consumer {consumer_id} lacks reciprocal consumes reference"
                )

    for action in actions:
        for slot_id in action.consumes:
            if slot_id not in plan.data_slots:
                raise RuntimeError(f"action {action.action_id} consumes missing slot {slot_id}")
        for slot_id in action.produces:
            if slot_id not in plan.data_slots:
                raise RuntimeError(f"action {action.action_id} produces missing slot {slot_id}")
            if strict_1f1b and plan.data_slots[slot_id].producer_action_id != action.action_id:
                raise RuntimeError(
                    f"action {action.action_id} claims slot {slot_id}, but its producer is "
                    f"{plan.data_slots[slot_id].producer_action_id!r}"
                )
        if action.is_noop and (action.consumes or action.produces):
            raise RuntimeError(f"no-op action {action.action_id} has blocking DataSlots")

    if not strict_1f1b:
        return

    compute: dict[tuple[int, int], dict[str, ScheduleAction]] = defaultdict(dict)
    stage_microbatches: dict[int, set[int]] = defaultdict(set)
    for action in actions:
        if action.action_type == "COMPUTE" and action.comp_type in {"F", "B"}:
            if not action.template_ref or action.template_ref not in plan.step_templates:
                raise RuntimeError(
                    f"1F1B compute action {action.action_id} has missing template "
                    f"{action.template_ref!r}"
                )
            key = (action.stage, action.mb_idx)
            if action.comp_type in compute[key]:
                raise RuntimeError(
                    f"duplicate 1F1B {action.comp_type} action for stage={action.stage}, mb={action.mb_idx}"
                )
            compute[key][action.comp_type] = action
            stage_microbatches[action.stage].add(action.mb_idx)
        if action.action_type in _P2P_TYPES:
            expected_role = "send" if action.action_type in _SEND_TYPES else "recv"
            if action.comm is None or action.comm.role != expected_role:
                raise RuntimeError(
                    f"P2P action {action.action_id} has invalid communication role"
                )
            transfer_id = action.comm.transfer_id
            if not transfer_id:
                raise RuntimeError(f"P2P action {action.action_id} has no transfer_id")
            if action.action_type == "RECV_F" and action.stage == 0:
                raise RuntimeError(f"first PP stage has RECV_F action {action.action_id}")
            if action.action_type == "SEND_F" and action.stage == plan.pp_degree - 1:
                raise RuntimeError(f"last PP stage has SEND_F action {action.action_id}")
            if action.action_type == "RECV_B" and action.stage == plan.pp_degree - 1:
                raise RuntimeError(f"last PP stage has RECV_B action {action.action_id}")
            if action.action_type == "SEND_B" and action.stage == 0:
                raise RuntimeError(f"first PP stage has SEND_B action {action.action_id}")
        if action.action_type == "UNSHARD" and not action.is_noop:
            if action.comm is None or action.comm.primitive != "allgather":
                raise RuntimeError(f"UNSHARD action {action.action_id} has no all-gather")
        if action.action_type == "RESHARD":
            if action.comm is not None and action.comm.primitive:
                raise RuntimeError(f"RESHARD action {action.action_id} incorrectly carries communication")
        if action.action_type == "REDUCE_GRAD" and not action.is_noop:
            if action.comm is None or not action.comm.primitive:
                raise RuntimeError(f"REDUCE_GRAD action {action.action_id} has no collective")

    for (stage, mb_idx), pair in compute.items():
        if set(pair) != {"F", "B"}:
            raise RuntimeError(
                f"incomplete 1F1B compute pair for stage={stage}, mb={mb_idx}: {sorted(pair)}"
            )
        if pair["F"].schedule_order >= pair["B"].schedule_order:
            raise RuntimeError(f"1F1B backward precedes forward for stage={stage}, mb={mb_idx}")

    expected_microbatches = set(range(plan.num_micro_batches))
    for stage, actual_microbatches in stage_microbatches.items():
        if actual_microbatches != expected_microbatches:
            raise RuntimeError(
                f"stage {stage} has incomplete microbatches: "
                f"expected={sorted(expected_microbatches)}, actual={sorted(actual_microbatches)}"
            )

    adjacency: dict[str, set[str]] = defaultdict(set)
    indegree = {action.action_id: 0 for action in actions}
    for slot in plan.data_slots.values():
        if not slot.producer_action_id:
            continue
        for consumer_id in slot.consumer_action_ids:
            if consumer_id not in adjacency[slot.producer_action_id]:
                adjacency[slot.producer_action_id].add(consumer_id)
                indegree[consumer_id] += 1
    ready = [action_id for action_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        action_id = ready.pop()
        visited += 1
        for consumer_id in adjacency[action_id]:
            indegree[consumer_id] -= 1
            if indegree[consumer_id] == 0:
                ready.append(consumer_id)
    if visited != len(actions):
        cyclic = sorted(action_id for action_id, degree in indegree.items() if degree > 0)
        raise RuntimeError(f"SchedulePlan has cyclic local dependencies: {cyclic}")


def validate_1f1b_transfer_pairs(plans: Iterable[SchedulePlan]) -> None:
    roles: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for plan in plans:
        for action in _flatten(plan.actions):
            if action.action_type not in _P2P_TYPES or action.comm is None:
                continue
            roles[action.comm.transfer_id][action.comm.role].append(action.action_id)
    for transfer_id, by_role in roles.items():
        if len(by_role.get("send", [])) != 1 or len(by_role.get("recv", [])) != 1:
            raise RuntimeError(
                f"unpaired PP transfer {transfer_id}: "
                f"send={by_role.get('send', [])}, recv={by_role.get('recv', [])}"
            )


def replay_1f1b_readiness(plans: Iterable[SchedulePlan]) -> None:
    """Prove that local dependencies and P2P rendezvous can complete.

    This intentionally assigns zero duration to every action. Hardware timing
    remains the responsibility of the upper DES; this replay catches dangling
    slots, impossible rank cursors, and unpaired communication.
    """
    plans = list(plans)
    for plan in plans:
        validate_schedule_plan(plan, strict_1f1b=True)
    validate_1f1b_transfer_pairs(plans)

    actions_by_rank = {
        plan.actions[0].rank if plan.actions else index: sorted(plan.actions, key=lambda a: a.schedule_order)
        for index, plan in enumerate(plans)
    }
    slots = {slot_id: slot for plan in plans for slot_id, slot in plan.data_slots.items()}
    ready_slots = {slot_id for slot_id, slot in slots.items() if slot.external}
    cursors = {rank: 0 for rank in actions_by_rank}
    done: set[str] = set()
    posted: dict[str, dict[str, ScheduleAction]] = defaultdict(dict)

    def complete(action: ScheduleAction) -> None:
        done.add(action.action_id)
        ready_slots.update(action.produces)

    while True:
        progressed = False
        for rank, actions in actions_by_rank.items():
            cursor = cursors[rank]
            if cursor >= len(actions):
                continue
            action = actions[cursor]
            if action.action_type in _RECV_TYPES:
                posted[action.comm.transfer_id]["recv"] = action  # type: ignore[union-attr]
                cursors[rank] += 1
                progressed = True
            elif set(action.consumes).issubset(ready_slots):
                if action.action_type in _SEND_TYPES:
                    posted[action.comm.transfer_id]["send"] = action  # type: ignore[union-attr]
                else:
                    complete(action)
                cursors[rank] += 1
                progressed = True

        for transfer_id, pair in list(posted.items()):
            if set(pair) == {"send", "recv"}:
                complete(pair["send"])
                complete(pair["recv"])
                del posted[transfer_id]
                progressed = True

        if all(cursors[rank] == len(actions) for rank, actions in actions_by_rank.items()) and not posted:
            return
        if not progressed:
            blocked = []
            for rank, actions in actions_by_rank.items():
                cursor = cursors[rank]
                if cursor < len(actions):
                    action = actions[cursor]
                    missing = sorted(set(action.consumes) - ready_slots)
                    blocked.append(f"rank={rank} action={action.action_id} missing={missing}")
            unmatched = sorted(posted)
            raise RuntimeError(
                "1F1B readiness replay deadlocked: "
                + "; ".join(blocked)
                + f"; unmatched_transfers={unmatched}"
            )
