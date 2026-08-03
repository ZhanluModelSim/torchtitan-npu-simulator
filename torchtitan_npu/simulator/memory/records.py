# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dataclasses shared by capture, static memory planning, and exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TensorRef:
    tensor_id: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    num_bytes: int
    requires_grad: bool = False


@dataclass(frozen=True, slots=True)
class RawMemoryEvent:
    event_id: int
    op_id: int
    seq_idx: int
    raw_op_type: str
    op_type: str
    phase: str
    module_path: str
    inputs: tuple[TensorRef, ...]
    outputs: tuple[TensorRef, ...]
    execution_kind: str
    pp_stage: int = -1
    pp_mb_idx: int = -1
    comp_type: str = ""


@dataclass(frozen=True, slots=True)
class CheckpointBoundaryEvent:
    checkpoint_id: str
    seq_idx: int
    inputs: tuple[TensorRef, ...]
    outputs: tuple[TensorRef, ...]
    marker_kind: str = "module"
    pp_stage: int = -1
    pp_mb_idx: int = -1
    comp_type: str = "F"


@dataclass(frozen=True, slots=True)
class FSDPResidencyEvent:
    group_id: str
    action: str
    seq_idx: int
    phase: str
    num_bytes: int
    tensor_ids: tuple[int, ...] = ()
    capture_process_rank: int = -1
    pp_stage: int = -1
    pp_mb_idx: int = -1
    comp_type: str = ""
    parent_compute_instance_id: str = ""
    shard_world_size: int = -1
    action_order: int = -1
    transition_id: str = ""
    schedule_source: str = "state"
    module_fqn: str = ""
    prefetch_source_fqn: str = ""
    prefetch_type: str = ""


@dataclass(slots=True)
class TensorLifetime:
    tensor_id: str
    kind: str
    num_bytes: int
    birth_seq: int
    death_seq: int
    producer_op: int
    producer_raw_op: str = ""
    producer_phase: str = ""
    consumer_ops: list[int] = field(default_factory=list)
    consumer_seqs: list[int] = field(default_factory=list)
    consumer_phases: list[str] = field(default_factory=list)
    alias_of: str = ""
    shape: tuple[int, ...] = ()
    dtype: str = ""
    modeled_num_bytes: int | None = None
    modeled_dtype: str = ""
    residency_policy: str = "resident"
    reason: str = ""

    @property
    def resident_num_bytes(self) -> int:
        return self.num_bytes if self.modeled_num_bytes is None else self.modeled_num_bytes

    def mark_consumer(self, op_id: int, seq_idx: int, phase: str) -> None:
        self.consumer_ops.append(op_id)
        self.consumer_seqs.append(seq_idx)
        self.consumer_phases.append(phase)
        if seq_idx > self.death_seq:
            self.death_seq = seq_idx


@dataclass(frozen=True, slots=True)
class MemoryTimelineEvent:
    seq_idx: int
    phase: str
    op_id: int
    action: str
    tensor_id: str
    kind: str
    num_bytes: int
    active_bytes_after: int
    reason: str = ""


@dataclass(frozen=True, slots=True)
class MemoryActionSpan:
    action_id: str
    action_type: str
    stage: int
    microbatch: int
    comp_type: str
    phase: str
    start_seq: int
    end_seq: int
    source_seq_idx: int = 0


@dataclass(frozen=True, slots=True)
class CheckpointTensorRecord:
    checkpoint_id: str
    marker_kind: str
    seq_idx: int
    tensor_id: str
    role: str
    shape: tuple[int, ...]
    dtype: str
    num_bytes: int
    modeled_num_bytes: int
    requires_grad: bool
    is_saved_activation: bool
    residency_policy: str
    pp_stage: int = -1
    pp_mb_idx: int = -1
    comp_type: str = "F"


@dataclass(slots=True)
class MemoryPlan:
    metric: str = "active_tensor_bytes"
    parameter_storage_dtype: str = ""
    offload_ac_saved_tensors: bool = False
    persistent_param_bytes: int = 0
    peak_active_bytes: int = 0
    model_active_bytes_peak: int = 0
    forward_peak_active_bytes: int = 0
    backward_peak_active_bytes: int = 0
    optimizer_peak_active_bytes: int = 0
    peak_seq_idx: int = 0
    peak_phase: str = ""
    raw_events: list[RawMemoryEvent] = field(default_factory=list)
    tensor_lifetimes: list[TensorLifetime] = field(default_factory=list)
    checkpoint_tensors: list[CheckpointTensorRecord] = field(default_factory=list)
    activation_offload_tensors: list[CheckpointTensorRecord] = field(default_factory=list)
    timeline_events: list[MemoryTimelineEvent] = field(default_factory=list)
    action_spans: list[MemoryActionSpan] = field(default_factory=list)
    unclassified_ops: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def _checkpoint_saved_activations_by_marker(self) -> dict[str, dict[str, Any]]:
        instances_by_marker: dict[
            str,
            dict[tuple[int, int, int, str], dict[str, int]],
        ] = {}
        for item in self.checkpoint_tensors:
            if not item.is_saved_activation or item.role != "input":
                continue
            instance_key = (
                item.seq_idx,
                item.pp_stage,
                item.pp_mb_idx,
                item.comp_type,
            )
            instance = instances_by_marker.setdefault(
                item.checkpoint_id,
                {},
            ).setdefault(
                instance_key,
                {
                    "logical_bytes": 0,
                    "modeled_bytes": 0,
                    "tensor_count": 0,
                },
            )
            instance["logical_bytes"] += item.num_bytes
            instance["modeled_bytes"] += item.modeled_num_bytes
            instance["tensor_count"] += 1

        result: dict[str, dict[str, Any]] = {}
        for marker, instances in sorted(instances_by_marker.items()):
            variant_counts: dict[tuple[int, int, int], int] = {}
            for instance in instances.values():
                variant = (
                    instance["logical_bytes"],
                    instance["modeled_bytes"],
                    instance["tensor_count"],
                )
                variant_counts[variant] = variant_counts.get(variant, 0) + 1

            size_variants = [
                {
                    "logical_bytes": logical_bytes,
                    "modeled_bytes": modeled_bytes,
                    "tensor_count": tensor_count,
                    "instance_count": instance_count,
                }
                for (
                    logical_bytes,
                    modeled_bytes,
                    tensor_count,
                ), instance_count in sorted(variant_counts.items())
            ]
            consistent = len(size_variants) == 1
            marker_kinds = {
                item.marker_kind
                for item in self.checkpoint_tensors
                if item.is_saved_activation
                and item.role == "input"
                and item.checkpoint_id == marker
            }
            result[marker] = {
                "marker_kind": (
                    next(iter(marker_kinds))
                    if len(marker_kinds) == 1
                    else "mixed"
                ),
                "logical_bytes_per_instance": (
                    size_variants[0]["logical_bytes"] if consistent else None
                ),
                "modeled_bytes_per_instance": (
                    size_variants[0]["modeled_bytes"] if consistent else None
                ),
                "tensor_count_per_instance": (
                    size_variants[0]["tensor_count"] if consistent else None
                ),
                "instance_count": len(instances),
                "logical_bytes_total": sum(
                    instance["logical_bytes"] for instance in instances.values()
                ),
                "modeled_bytes_total": sum(
                    instance["modeled_bytes"] for instance in instances.values()
                ),
                "pp_stages": sorted(
                    {
                        stage
                        for _seq_idx, stage, _microbatch, _comp_type in instances
                        if stage >= 0
                    }
                ),
                "microbatches": sorted(
                    {
                        microbatch
                        for _seq_idx, _stage, microbatch, _comp_type in instances
                        if microbatch >= 0
                    }
                ),
                "size_variants": size_variants,
            }
        return result

    def _checkpoint_prefetch_by_marker(self) -> dict[str, dict[str, Any]]:
        instances_by_marker: dict[
            str,
            dict[tuple[int, int, int, str], dict[str, int]],
        ] = {}
        marker_kinds: dict[str, set[str]] = {}
        for item in self.checkpoint_tensors:
            if not item.is_saved_activation or item.role not in {
                "input",
                "recompute_saved",
            }:
                continue
            instance_key = (
                item.seq_idx,
                item.pp_stage,
                item.pp_mb_idx,
                item.comp_type,
            )
            instance = instances_by_marker.setdefault(
                item.checkpoint_id,
                {},
            ).setdefault(
                instance_key,
                {
                    "boundary_logical_bytes": 0,
                    "boundary_modeled_bytes": 0,
                    "boundary_tensor_count": 0,
                    "recompute_saved_logical_bytes": 0,
                    "recompute_saved_modeled_bytes": 0,
                    "recompute_saved_tensor_count": 0,
                },
            )
            prefix = "boundary" if item.role == "input" else "recompute_saved"
            instance[f"{prefix}_logical_bytes"] += item.num_bytes
            instance[f"{prefix}_modeled_bytes"] += item.modeled_num_bytes
            instance[f"{prefix}_tensor_count"] += 1
            marker_kinds.setdefault(item.checkpoint_id, set()).add(item.marker_kind)

        result: dict[str, dict[str, Any]] = {}
        for marker, instances in sorted(instances_by_marker.items()):
            variants: dict[tuple[int, ...], int] = {}
            enriched_instances: list[dict[str, int]] = []
            for instance in instances.values():
                enriched = {
                    **instance,
                    "prefetch_logical_bytes": (
                        instance["boundary_logical_bytes"]
                        + instance["recompute_saved_logical_bytes"]
                    ),
                    "modeled_bytes": (
                        instance["boundary_modeled_bytes"]
                        + instance["recompute_saved_modeled_bytes"]
                    ),
                    "tensor_count": (
                        instance["boundary_tensor_count"]
                        + instance["recompute_saved_tensor_count"]
                    ),
                }
                enriched_instances.append(enriched)
                variant = tuple(enriched[key] for key in (
                    "boundary_logical_bytes",
                    "boundary_modeled_bytes",
                    "boundary_tensor_count",
                    "recompute_saved_logical_bytes",
                    "recompute_saved_modeled_bytes",
                    "recompute_saved_tensor_count",
                    "prefetch_logical_bytes",
                    "modeled_bytes",
                    "tensor_count",
                ))
                variants[variant] = variants.get(variant, 0) + 1

            size_variants = [
                {
                    "boundary_logical_bytes": values[0],
                    "boundary_modeled_bytes": values[1],
                    "boundary_tensor_count": values[2],
                    "recompute_saved_logical_bytes": values[3],
                    "recompute_saved_modeled_bytes": values[4],
                    "recompute_saved_tensor_count": values[5],
                    "prefetch_logical_bytes": values[6],
                    "modeled_bytes": values[7],
                    "tensor_count": values[8],
                    "instance_count": instance_count,
                }
                for values, instance_count in sorted(variants.items())
            ]
            consistent = len(size_variants) == 1
            per_instance = size_variants[0] if consistent else {}
            kinds = marker_kinds[marker]
            result[marker] = {
                "marker_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                "boundary_logical_bytes_per_instance": per_instance.get(
                    "boundary_logical_bytes"
                ),
                "recompute_saved_logical_bytes_per_instance": per_instance.get(
                    "recompute_saved_logical_bytes"
                ),
                "prefetch_logical_bytes_per_instance": per_instance.get(
                    "prefetch_logical_bytes"
                ),
                "boundary_modeled_bytes_per_instance": per_instance.get(
                    "boundary_modeled_bytes"
                ),
                "recompute_saved_modeled_bytes_per_instance": per_instance.get(
                    "recompute_saved_modeled_bytes"
                ),
                "modeled_bytes_per_instance": per_instance.get("modeled_bytes"),
                "boundary_tensor_count_per_instance": per_instance.get(
                    "boundary_tensor_count"
                ),
                "recompute_saved_tensor_count_per_instance": per_instance.get(
                    "recompute_saved_tensor_count"
                ),
                "tensor_count_per_instance": per_instance.get("tensor_count"),
                "instance_count": len(instances),
                "prefetch_logical_bytes_total": sum(
                    instance["prefetch_logical_bytes"]
                    for instance in enriched_instances
                ),
                "modeled_bytes_total": sum(
                    instance["modeled_bytes"] for instance in enriched_instances
                ),
                "pp_stages": sorted(
                    {
                        stage
                        for _seq_idx, stage, _microbatch, _comp_type in instances
                        if stage >= 0
                    }
                ),
                "microbatches": sorted(
                    {
                        microbatch
                        for _seq_idx, _stage, microbatch, _comp_type in instances
                        if microbatch >= 0
                    }
                ),
                "size_variants": size_variants,
            }
        return result

    def _activation_prefetch_records(self) -> list[CheckpointTensorRecord]:
        records = [
            item
            for item in self.checkpoint_tensors
            if item.is_saved_activation
            and item.role in {"input", "recompute_saved"}
            and item.residency_policy == "offloaded"
        ]
        records.extend(self.activation_offload_tensors)

        unique_records: list[CheckpointTensorRecord] = []
        seen: set[tuple[str, int, int, int, str]] = set()
        for item in records:
            key = (
                item.checkpoint_id,
                item.seq_idx,
                item.pp_stage,
                item.pp_mb_idx,
                item.tensor_id,
            )
            if key in seen:
                continue
            seen.add(key)
            unique_records.append(item)
        return unique_records

    def _activation_prefetch_by_marker(self) -> dict[str, dict[str, Any]]:
        instances_by_marker: dict[
            str,
            dict[tuple[int, int, int], dict[str, int]],
        ] = {}
        marker_kinds: dict[str, set[str]] = {}
        for item in self._activation_prefetch_records():
            instance_key = (
                item.pp_stage,
                item.pp_mb_idx,
                (
                    0
                    if item.pp_stage >= 0 or item.pp_mb_idx >= 0
                    else item.seq_idx
                ),
            )
            instance = instances_by_marker.setdefault(
                item.checkpoint_id,
                {},
            ).setdefault(
                instance_key,
                {
                    "logical_bytes": 0,
                    "modeled_bytes": 0,
                    "tensor_count": 0,
                },
            )
            instance["logical_bytes"] += item.num_bytes
            instance["modeled_bytes"] += item.modeled_num_bytes
            instance["tensor_count"] += 1
            marker_kinds.setdefault(item.checkpoint_id, set()).add(item.marker_kind)

        result: dict[str, dict[str, Any]] = {}
        for marker, instances in sorted(instances_by_marker.items()):
            variants: dict[tuple[int, int, int], int] = {}
            for instance in instances.values():
                variant = (
                    instance["logical_bytes"],
                    instance["modeled_bytes"],
                    instance["tensor_count"],
                )
                variants[variant] = variants.get(variant, 0) + 1

            size_variants = [
                {
                    "logical_bytes": logical_bytes,
                    "modeled_bytes": modeled_bytes,
                    "tensor_count": tensor_count,
                    "instance_count": instance_count,
                }
                for (
                    logical_bytes,
                    modeled_bytes,
                    tensor_count,
                ), instance_count in sorted(variants.items())
            ]
            consistent = len(size_variants) == 1
            per_instance = size_variants[0] if consistent else {}
            kinds = marker_kinds[marker]
            result[marker] = {
                "marker_kind": next(iter(kinds)) if len(kinds) == 1 else "mixed",
                "logical_bytes_per_instance": per_instance.get("logical_bytes"),
                "modeled_bytes_per_instance": per_instance.get("modeled_bytes"),
                "tensor_count_per_instance": per_instance.get("tensor_count"),
                "instance_count": len(instances),
                "logical_bytes_total": sum(
                    instance["logical_bytes"] for instance in instances.values()
                ),
                "modeled_bytes_total": sum(
                    instance["modeled_bytes"] for instance in instances.values()
                ),
                "pp_stages": sorted(
                    {
                        stage
                        for stage, _microbatch, _occurrence in instances
                        if stage >= 0
                    }
                ),
                "microbatches": sorted(
                    {
                        microbatch
                        for _stage, microbatch, _occurrence in instances
                        if microbatch >= 0
                    }
                ),
                "size_variants": size_variants,
            }
        return result

    def to_summary_dict(self) -> dict[str, Any]:
        recompute_saved_records = [
            item
            for item in self.checkpoint_tensors
            if item.role == "recompute_saved"
        ]
        recompute_saved_lifetimes = [
            item
            for item in self.tensor_lifetimes
            if item.kind == "checkpoint_saved_for_recompute"
        ]
        attributed_recompute_saved_ids = {
            item.tensor_id for item in recompute_saved_records
        }
        unattributed_recompute_saved = [
            item
            for item in recompute_saved_lifetimes
            if item.tensor_id not in attributed_recompute_saved_ids
        ]
        offloaded_activation_lifetimes = [
            item
            for item in self.tensor_lifetimes
            if item.residency_policy == "offloaded"
            and item.kind in {
                "checkpoint_saved_activation",
                "checkpoint_saved_for_recompute",
                "offloaded_activation",
            }
        ]
        activation_prefetch_records = self._activation_prefetch_records()
        unattributed_prefetch_records = [
            item
            for item in activation_prefetch_records
            if item.marker_kind == "unattributed"
        ]
        return {
            "metric": self.metric,
            "parameter_storage_dtype": self.parameter_storage_dtype or "captured",
            "offload_ac_saved_tensors": self.offload_ac_saved_tensors,
            "persistent_param_bytes": self.persistent_param_bytes,
            "active_bytes_peak": self.peak_active_bytes,
            "model_active_bytes_peak": self.model_active_bytes_peak,
            "forward_active_bytes_peak": self.forward_peak_active_bytes,
            "backward_active_bytes_peak": self.backward_peak_active_bytes,
            "optimizer_active_bytes_peak": self.optimizer_peak_active_bytes,
            "peak_seq_idx": self.peak_seq_idx,
            "peak_phase": self.peak_phase,
            "raw_memory_event_count": len(self.raw_events),
            "tensor_lifetime_count": len(self.tensor_lifetimes),
            "checkpoint_tensor_count": len(self.checkpoint_tensors),
            "checkpoint_saved_activation_count": sum(
                item.is_saved_activation and item.role == "input"
                for item in self.checkpoint_tensors
            ),
            "checkpoint_tensor_logical_bytes": sum(
                item.num_bytes
                for item in self.checkpoint_tensors
                if item.is_saved_activation and item.role == "input"
            ),
            "checkpoint_tensor_modeled_bytes": sum(
                item.modeled_num_bytes
                for item in self.checkpoint_tensors
                if item.is_saved_activation and item.role == "input"
            ),
            "checkpoint_recompute_saved_tensor_count": len(
                recompute_saved_lifetimes
            ),
            "checkpoint_recompute_saved_logical_bytes": sum(
                item.num_bytes for item in recompute_saved_lifetimes
            ),
            "checkpoint_recompute_saved_modeled_bytes": sum(
                item.resident_num_bytes for item in recompute_saved_lifetimes
            ),
            "checkpoint_prefetch_unattributed_tensor_count": len(
                unattributed_recompute_saved
            ),
            "checkpoint_prefetch_unattributed_logical_bytes": sum(
                item.num_bytes for item in unattributed_recompute_saved
            ),
            "checkpoint_saved_activations": (
                self._checkpoint_saved_activations_by_marker()
            ),
            "checkpoint_prefetch": self._checkpoint_prefetch_by_marker(),
            "activation_offload_tensor_count": len(
                offloaded_activation_lifetimes
            ),
            "activation_offload_logical_bytes": sum(
                item.num_bytes for item in offloaded_activation_lifetimes
            ),
            "activation_offload_modeled_bytes": sum(
                item.resident_num_bytes for item in offloaded_activation_lifetimes
            ),
            "activation_prefetch_tensor_count": len(
                activation_prefetch_records
            ),
            "activation_prefetch_logical_bytes": sum(
                item.num_bytes for item in activation_prefetch_records
            ),
            "activation_prefetch_unattributed_tensor_count": len(
                unattributed_prefetch_records
            ),
            "activation_prefetch_unattributed_logical_bytes": sum(
                item.num_bytes for item in unattributed_prefetch_records
            ),
            "activation_prefetch": self._activation_prefetch_by_marker(),
            "timeline_event_count": len(self.timeline_events),
            "memory_action_span_count": len(self.action_spans),
            "unclassified_op_count": len(self.unclassified_ops),
            "included": [
                "local parameter tensors",
                "external inputs and labels observed by dispatch",
                "non-alias operator outputs by use-def liveness",
                "collective communication outputs observed by CommEvent",
            ],
            "excluded": [
                "allocator reserved/cache",
                "fragmentation",
                "kernel workspace",
                "device internal temporary buffers",
            ],
            "notes": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.to_summary_dict()
        data["tensor_lifetimes"] = [asdict(item) for item in self.tensor_lifetimes]
        data["checkpoint_tensors"] = [asdict(item) for item in self.checkpoint_tensors]
        data["activation_offload_tensors"] = [
            asdict(item) for item in self.activation_offload_tensors
        ]
        data["timeline_events"] = [asdict(item) for item in self.timeline_events]
        data["action_spans"] = [asdict(item) for item in self.action_spans]
        data["unclassified_ops"] = list(self.unclassified_ops)
        return data
