# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Assign communication ownership without rewriting captured operator graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torchtitan_simulator.capture.op_mapping import communication_primitive

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torchtitan_simulator.capture.comm_events import CommEvent
    from torchtitan_simulator.capture.schedule_assemblers import ActionSpec
    from torchtitan_simulator.ir.op_node import OpNode
    from torchtitan_simulator.ir.step_graph import StepGraph

_COMPUTE_TYPES = {"F", "B", "I", "W", "F_RECOMPUTE"}
_STAGE_TEMPLATE_TYPES = _COMPUTE_TYPES | {"OPTIMIZER"}


@dataclass(slots=True)
class CommunicationOwnershipResult:
    specs: list[ActionSpec]
    internal_fsdp_transitions: set[str]
    external_fsdp_transitions: set[str]
    internal_gradient_reductions: int
    external_gradient_reductions: int
    removed_noop_gradient_intents: int
    stage_owned_collectives: int


def _iter_specs(specs: Iterable[ActionSpec]) -> Iterable[ActionSpec]:
    for spec in specs:
        yield spec
        if spec.sub_actions:
            yield from _iter_specs(spec.sub_actions)


def _find_node(
    step_templates: dict[str, StepGraph],
    op_id: int,
) -> tuple[str, OpNode] | None:
    if not op_id:
        return None
    for template_id, template in step_templates.items():
        node = template.nodes.get(op_id)
        if node is not None:
            return template_id, node
    return None


def _find_comm_node(
    step_templates: dict[str, StepGraph],
    event: CommEvent,
) -> tuple[str, OpNode] | None:
    if not event.op_id:
        return None
    found = _find_node(step_templates, event.op_id)
    if found is None:
        return None
    _template_id, node = found
    if communication_primitive(node.annotations.get("raw_op_type", "")) != event.comm_primitive:
        return None
    return found


class GradientReductionOwnershipPlugin:
    """Keep only genuinely standalone gradient reductions in L2.

    FSDP/DDP reductions may be emitted both as an L1 communication node and as
    a schedule-level REDUCE_GRAD intent.  The L1 template is authoritative when
    the real collective is already inside F/B/I/W.  A real collective captured
    outside every compute template remains an explicit L2 action.
    """

    def __init__(self) -> None:
        self.internal_reductions = 0
        self.external_reductions = 0
        self.removed_noop_intents = 0

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        internal_op_ids: set[int] = set()
        for template in step_templates.values():
            if template.step_type not in _STAGE_TEMPLATE_TYPES:
                continue
            for node in template.nodes.values():
                if communication_primitive(node.annotations.get("raw_op_type", "")) not in {
                    "reduce_scatter",
                    "allreduce",
                }:
                    continue
                if node.annotations.get("gradient_reduction") is False:
                    continue
                node.annotations.update(
                    {
                        "communication_owner": "L1_STAGE",
                        "gradient_reduction": True,
                    }
                )
                internal_op_ids.add(node.op_id)
        self.internal_reductions = len(internal_op_ids)

        event_by_op_id = {
            event.op_id: event
            for event in comm_events
            if event.op_id
            and event.comm_primitive in {"reduce_scatter", "allreduce"}
            and event.gradient_reduction is not False
        }
        retained: list[ActionSpec] = []
        for spec in specs:
            if spec.action_type != "REDUCE_GRAD":
                retained.append(spec)
                continue
            if spec.is_noop or (spec.comm is not None and spec.comm.is_noop):
                self.removed_noop_intents += 1
                continue

            event = event_by_op_id.get(spec.comm_op_id)
            found = _find_comm_node(step_templates, event) if event is not None else None
            if found is not None:
                template_id, node = found
                template = step_templates[template_id]
                if template.step_type in _STAGE_TEMPLATE_TYPES:
                    node.annotations.update(
                        {
                            "communication_owner": "L1_STAGE",
                            "gradient_reduction": True,
                        }
                    )
                    template.annotations["communication_ownership_normalized"] = True
                    if node.op_id not in internal_op_ids:
                        internal_op_ids.add(node.op_id)
                        self.internal_reductions += 1
                    continue

            spec.annotations["communication_owner"] = "L2_STANDALONE"
            self.external_reductions += 1
            retained.append(spec)
        return retained


class EmbeddedStageCommunicationPlugin:
    """Mark every remaining communication node embedded in stage compute."""

    def __init__(self) -> None:
        self.stage_owned_collectives = 0

    def apply(
        self,
        *,
        step_templates: dict[str, StepGraph],
        specs: list[ActionSpec],
        comm_events: list[CommEvent],
    ) -> list[ActionSpec]:
        del comm_events
        counts_by_template: dict[str, int] = {}
        for template_id, template in step_templates.items():
            if template.step_type not in _STAGE_TEMPLATE_TYPES:
                continue
            collective_count = 0
            for node in template.nodes.values():
                if communication_primitive(node.annotations.get("raw_op_type", "")) is None:
                    continue
                node.annotations.setdefault(
                    "communication_owner",
                    "L1_STAGE",
                )
                collective_count += 1
                self.stage_owned_collectives += 1
            counts_by_template[template_id] = collective_count
            if collective_count:
                template.annotations["communication_ownership_normalized"] = True
                template.annotations["stage_owned_collectives"] = collective_count

        for spec in _iter_specs(specs):
            if spec.action_type == "COMPUTE":
                collective_count = counts_by_template.get(
                    spec.template_ref,
                    0,
                )
                spec.annotations["stage_owned_collectives"] = collective_count
                spec.annotations["communication_ownership_normalized"] = True
        return specs


class FSDPTransitionOwnershipPlugin:
    """Remove duplicate L2 residency actions only for actual embedded work."""

    def __init__(self) -> None:
        self.internal_transitions: set[str] = set()
        self.external_transitions: set[str] = set()

    def apply(self, *, step_templates, specs, comm_events):
        events = {
            event.fsdp_transition_id: event
            for event in comm_events
            if event.comm_primitive == "allgather" and event.fsdp_transition_id
        }
        for spec in specs:
            if spec.action_type != "UNSHARD":
                continue
            transition = str(spec.annotations.get("fsdp_transition_id", ""))
            if not transition:
                continue
            if spec.annotations.get("fsdp_schedule_source") == "intent":
                self.external_transitions.add(transition)
                continue
            if spec.annotations.get("fsdp_schedule_source") != "state":
                continue
            event = events.get(transition)
            found = _find_comm_node(step_templates, event) if event is not None else None
            if found is None:
                if spec.annotations.get("shard_world_size") == 1:
                    self.internal_transitions.add(transition)
                    continue
                raise ValueError(
                    f"FSDP transition {transition!r} has no captured all-gather; template reuse is unsupported"
                )
            template_id, node = found
            assert event is not None
            if step_templates[template_id].step_type in _STAGE_TEMPLATE_TYPES:
                self.internal_transitions.add(transition)
                node.annotations.update(
                    communication_owner="L1_STAGE",
                    fsdp_group_id=event.fsdp_group_id,
                    fsdp_transition_id=transition,
                )

        retained = []
        for spec in specs:
            if spec.action_type not in {"UNSHARD", "RESHARD"}:
                retained.append(spec)
                continue
            transition = str(spec.annotations.get("fsdp_transition_id", ""))
            if transition in self.internal_transitions:
                continue
            if spec.is_noop or spec.annotations.get("fsdp_intent_noop"):
                continue
            spec.annotations["communication_owner"] = (
                "L2_PREFETCH" if transition in self.external_transitions else "L2_STANDALONE"
            )
            retained.append(spec)
        return retained


def normalize_communication_ownership(
    *,
    step_templates: dict[str, StepGraph],
    specs: list[ActionSpec],
    comm_events: Iterable[CommEvent],
) -> CommunicationOwnershipResult:
    """Assign L1/L2 accounting; preserve all captured nodes and dependencies."""
    events = list(comm_events)
    fsdp_plugin = FSDPTransitionOwnershipPlugin()
    gradient_plugin = GradientReductionOwnershipPlugin()
    embedded_plugin = EmbeddedStageCommunicationPlugin()
    normalized = list(specs)
    for plugin in (fsdp_plugin, gradient_plugin, embedded_plugin):
        normalized = plugin.apply(step_templates=step_templates, specs=normalized, comm_events=events)
    return CommunicationOwnershipResult(
        specs=normalized,
        internal_fsdp_transitions=fsdp_plugin.internal_transitions,
        external_fsdp_transitions=fsdp_plugin.external_transitions,
        internal_gradient_reductions=gradient_plugin.internal_reductions,
        external_gradient_reductions=gradient_plugin.external_reductions,
        removed_noop_gradient_intents=gradient_plugin.removed_noop_intents,
        stage_owned_collectives=embedded_plugin.stage_owned_collectives,
    )
