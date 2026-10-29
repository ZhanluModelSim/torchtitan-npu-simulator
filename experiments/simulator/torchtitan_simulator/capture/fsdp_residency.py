# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit FSDP unshard/reshard residency capture."""

from __future__ import annotations

import contextvars
import itertools
from contextlib import nullcontext
from typing import Any

import torch

_transition_counter = itertools.count()
_active_fsdp_transition: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "simulator_active_fsdp_transition",
    default=("", ""),
)
_active_fsdp_communication_context: contextvars.ContextVar[tuple[str, str, str]] = contextvars.ContextVar(
    "simulator_active_fsdp_communication_context",
    default=("", "", ""),
)
_active_fsdp_prefetch_source: contextvars.ContextVar[tuple[str, str]] = contextvars.ContextVar(
    "simulator_active_fsdp_prefetch_source",
    default=("", ""),
)
_active_fsdp_state_source: contextvars.ContextVar[str] = contextvars.ContextVar(
    "simulator_active_fsdp_state_source",
    default="",
)
_active_fsdp_reduction_dependencies: contextvars.ContextVar[tuple[torch.Tensor, ...]] = contextvars.ContextVar(
    "simulator_active_fsdp_reduction_dependencies",
    default=(),
)
_active_fsdp_reduction_groups: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "simulator_fsdp_reduction_groups",
    default=frozenset(),
)
_active_fsdp_group_dimensions: contextvars.ContextVar[dict[tuple[str, str], str] | None] = contextvars.ContextVar(
    "simulator_fsdp_group_dimensions",
    default=None,
)


def get_active_fsdp_transition() -> tuple[str, str]:
    return _active_fsdp_transition.get()


def get_active_fsdp_communication_context() -> tuple[str, str, str]:
    return _active_fsdp_communication_context.get()


def get_active_fsdp_group_dimension(group, comm_primitive: str) -> str:
    """Resolve only the actual collective group used by the active FSDP call."""
    from torchtitan_simulator.capture.comm_events import _canonical_collective_group

    group = _canonical_collective_group(group)
    group_name = str(getattr(group, "group_name", group))
    dimensions = _active_fsdp_group_dimensions.get()
    return dimensions.get((comm_primitive, group_name), "") if dimensions is not None else ""


def is_active_fsdp_reduction_group(group) -> bool:
    from torchtitan_simulator.capture.comm_events import _canonical_collective_group

    return str(getattr(_canonical_collective_group(group), "group_name", group)) in _active_fsdp_reduction_groups.get()


def get_active_fsdp_reduction_dependencies(group) -> tuple[torch.Tensor, ...]:
    """Return parameter gradients consumed by the active FSDP reduction."""
    return _active_fsdp_reduction_dependencies.get() if is_active_fsdp_reduction_group(group) else ()


def _module_fqn(param_group: Any) -> str:
    return str(getattr(param_group, "_module_fqn", "") or "")


def _transition_prefetch(param_group: Any) -> tuple[str, str]:
    return (
        str(getattr(param_group, "_sim_prefetch_source_fqn", "") or ""),
        str(getattr(param_group, "_sim_prefetch_type", "") or ""),
    )


def _memory_tracking_enabled() -> bool:
    from torchtitan_simulator.capture.comm_events import get_active_recorder

    recorder = get_active_recorder()
    return recorder is not None and recorder.memory_tracking_enabled


def _to_local_tensor(value: object) -> torch.Tensor | None:
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(value, DTensor):
            local_tensor = getattr(value, "_local_tensor", None)
            return local_tensor if isinstance(local_tensor, torch.Tensor) else value.to_local()
    except Exception:
        pass
    return value if isinstance(value, torch.Tensor) else None


def _residency_metadata(param_group: Any) -> tuple[int, tuple[torch.Tensor, ...]]:
    tracked_tensors: dict[int, torch.Tensor] = {}
    byte_tensors: dict[int, torch.Tensor] = {}

    def track(value: object, *, count_bytes: bool = False) -> None:
        tensor = _to_local_tensor(value)
        if tensor is None:
            return
        tracked_tensors[id(tensor)] = tensor
        if count_bytes:
            byte_tensors[id(tensor)] = tensor

    for fsdp_param in param_group.fsdp_params:
        for tensor in getattr(fsdp_param, "all_gather_outputs", ()):
            track(tensor, count_bytes=True)
        track(getattr(fsdp_param, "sharded_param", None))
        track(getattr(fsdp_param, "_sharded_local_tensor", None))
        track(getattr(fsdp_param, "_sharded_param_data", None))
        track(getattr(fsdp_param, "_sharded_post_forward_param_data", None))
        track(getattr(fsdp_param, "_unsharded_param", None))
        for tensor in getattr(fsdp_param, "_unsharded_inner_tensors", ()):
            track(tensor)

    num_bytes = sum(tensor.numel() * tensor.element_size() for tensor in byte_tensors.values())
    return num_bytes, tuple(tracked_tensors.values())


def _post_backward_dependency_tensors(
    param_group: Any,
) -> tuple[torch.Tensor, ...]:
    """Collect the parameter-group gradients that make its RS launchable.

    FSDP packs these gradients into a temporary flat reduce-scatter buffer.
    Keep the actual gradient producers as additional launch dependencies
    alongside the dispatch-visible packing and native collective operands.
    """
    dependencies: dict[int, torch.Tensor] = {}
    for fsdp_param in getattr(param_group, "fsdp_params", ()):
        gradient: object | None = None
        if getattr(fsdp_param, "unsharded_accumulated_grad", None) is not None:
            gradient = getattr(
                fsdp_param,
                "unsharded_accumulated_grad_data",
                None,
            )
        if gradient is None:
            unsharded_param = getattr(fsdp_param, "unsharded_param", None)
            gradient = getattr(unsharded_param, "grad", None)
        tensor = _to_local_tensor(gradient)
        if tensor is not None:
            dependencies[id(tensor)] = tensor
    return tuple(dependencies.values())


def _record_residency(
    param_group: Any,
    action: str,
    metadata: tuple[int, tuple[torch.Tensor, ...]],
    shard_world_size: int,
    transition_id: str,
    *,
    include_schedule: bool = True,
    include_memory: bool = True,
    schedule_source: str = "state",
) -> None:
    from torchtitan_simulator.capture.comm_events import get_active_recorder
    from torchtitan_simulator.capture.dispatch_capture import get_active_capture

    recorder = get_active_recorder()
    capture = get_active_capture()
    num_bytes, tensors = metadata
    if recorder is None:
        return
    tensor_ids = tuple(sorted(capture.tensor_id(tensor) if capture is not None else id(tensor) for tensor in tensors))
    # All-gather buffers witness the underlying storage even when a parameter
    # wrapper exposes a separate tensor identity to the consuming operator.
    storage_keys = (
        tuple(sorted({capture._storage_identity(tensor)[0] for tensor in tensors}))
        if capture is not None and include_memory
        else ()
    )
    allocation_sizes: dict[str, int] = {}
    if capture is not None and include_memory:
        allocation_tensors = (
            tensors
            if action == "staging_free"
            else (
                tensor
                for parameter in getattr(param_group, "fsdp_params", ())
                for tensor in getattr(parameter, "all_gather_outputs", ())
            )
        )
        for value in allocation_tensors:
            tensor = _to_local_tensor(value)
            if tensor is None:
                continue
            key, storage_bytes = capture._storage_identity(tensor)
            allocation_sizes[key] = max(storage_bytes, tensor.numel() * tensor.element_size())
        if allocation_sizes and sum(allocation_sizes.values()) != num_bytes:
            raise ValueError("FSDP allocation storage bytes disagree with its residency byte ledger")
    training_state = str(getattr(param_group, "_training_state", "")).lower()
    phase = "backward" if "backward" in training_state else "forward"
    if capture is not None and capture.phase_provider is not None:
        phase = capture.phase_provider()
    recorder.record_fsdp_residency(
        group_id=str(id(param_group)),
        action=action,
        phase=phase,
        num_bytes=num_bytes,
        tensor_ids=tensor_ids,
        storage_keys=storage_keys,
        allocation_storage_bytes=tuple(sorted(allocation_sizes.items())),
        shard_world_size=shard_world_size,
        transition_id=transition_id,
        include_schedule=include_schedule,
        include_memory=include_memory,
        schedule_source=schedule_source,
        module_fqn=_module_fqn(param_group),
        prefetch_source_fqn=_transition_prefetch(param_group)[0],
        prefetch_type=_transition_prefetch(param_group)[1],
    )


def _shard_world_size(param_group: Any) -> int:
    try:
        return int(param_group._all_gather_process_group.size())
    except (AttributeError, RuntimeError, TypeError, ValueError):
        mesh_info = getattr(param_group, "mesh_info", None)
        shard_mesh_size = getattr(mesh_info, "shard_mesh_size", -1)
        return int(shard_mesh_size) if shard_mesh_size is not None else -1


def _uses_fsdp_sharding(param_group: Any) -> bool:
    """Return whether the composable data-parallel group shards parameters.

    PyTorch's ``replicate()`` and ``fully_shard()`` both use
    ``FSDPParamGroup`` internally. Replicated groups carry ``DDPMeshInfo`` and
    perform no parameter all-gather, so they must not emit FSDP residency
    transitions in the simulator.
    """
    from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo

    return isinstance(getattr(param_group, "mesh_info", None), FSDPMeshInfo)


def _set_stage_fsdp_state(meta_env: Any, state: str) -> None:
    """Synchronize the PP capture context with an FSDP state transition."""
    stage = meta_env._pp_context.get("stage", -1)
    if isinstance(stage, int):
        meta_env._fsdp_state[stage] = state


def install_fsdp_residency_hooks(scope) -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext, FSDPParamGroup
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState

    import torchtitan_simulator.meta_env as meta_env

    original_unshard = FSDPParamGroup.unshard
    original_wait_for_unshard = FSDPParamGroup.wait_for_unshard
    original_reshard = FSDPParamGroup.reshard
    original_finalize_backward = FSDPParamGroup.finalize_backward
    original_post_backward = FSDPParamGroup.post_backward
    orig_prefetch_unshard = FSDPParamGroup._prefetch_unshard
    orig_state_pre_forward = FSDPState._pre_forward
    orig_state_pre_backward = FSDPState._pre_backward
    original_state_post_forward = FSDPState._post_forward
    original_comm_setattr = FSDPCommContext.__setattr__
    # Store provenance only: retaining a result tensor here would itself extend
    # the lifetime that this observer is supposed to measure.
    staging_owners: dict[int, tuple[Any, str]] = {}
    deferred_releases: list[dict[str, Any]] = []

    def record_staging_release(param_group, pending, transition):
        buffer = getattr(pending, "all_gather_output", None)
        if buffer is not None and transition:
            _record_residency(
                param_group,
                "staging_free",
                (buffer.numel() * buffer.element_size(), (buffer,)),
                _shard_world_size(param_group),
                transition,
                include_schedule=False,
            )

    def patched_comm_setattr(self, name, value):
        previous = getattr(self, name, None) if name == "all_gather_state" else None
        original_comm_setattr(self, name, value)
        if previous is not None and previous is not value:
            from torchtitan_simulator.capture.dispatch_capture import get_active_capture

            pending = previous.all_gather_result
            owner = staging_owners.pop(id(pending), None)
            capture = get_active_capture()
            if owner is not None and capture is not None:
                group, transition = owner
                buffer = pending.all_gather_output
                deferred_releases.append(
                    dict(
                        group_id=str(id(group)),
                        action="staging_free",
                        phase=capture.phase_provider() if capture.phase_provider else "forward",
                        num_bytes=buffer.numel() * buffer.element_size(),
                        tensor_ids=(capture.tensor_id(buffer),),
                        storage_keys=(capture._storage_identity(buffer)[0],),
                        allocation_storage_bytes=(
                            (capture._storage_identity(buffer)[0], buffer.numel() * buffer.element_size()),
                        ),
                        shard_world_size=_shard_world_size(group),
                        transition_id=transition,
                        module_fqn=_module_fqn(group),
                        include_schedule=False,
                    )
                )

    def flush_deferred_releases():
        from torchtitan_simulator.capture.comm_events import get_active_recorder

        recorder = get_active_recorder()
        if recorder is not None:
            for metadata in deferred_releases:
                recorder.record_fsdp_residency(**metadata)
        deferred_releases.clear()

    def patched_unshard(self, async_op=False):
        from torchtitan_simulator.capture.comm_events import get_active_recorder
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        if not _uses_fsdp_sharding(self):
            capture = get_active_capture()
            scaffold_scope = (
                capture.annotate_dispatch_events("replicate_unshard_scaffold") if capture is not None else nullcontext()
            )
            with scaffold_scope:
                return original_unshard(self, async_op)

        recorder = get_active_recorder()
        capture_rank = recorder.capture_process_rank if recorder is not None else -1
        group_id = str(id(self))
        transition_id = f"fsdp:r{capture_rank}:g{group_id}:u{next(_transition_counter)}"
        event_count_before = len(recorder.events) if recorder is not None else 0
        was_unsharded = self.is_unsharded
        previous_transition_id = getattr(self, "_sim_active_transition_id", "")
        previous_pending_transition_id = getattr(self, "_sim_pending_transition_id", "")
        explicit_schedule_action = meta_env._pp_context.get("comp_type") == "UNSHARD"
        previous_comm_layer = meta_env._comm_layer
        meta_env._comm_layer = "L2"
        token = _active_fsdp_transition.set((transition_id, group_id))
        prefetch_source_fqn, prefetch_type = _active_fsdp_prefetch_source.get()
        communication_token = _active_fsdp_communication_context.set(
            (_module_fqn(self), prefetch_source_fqn, prefetch_type)
        )
        group_token = _active_fsdp_group_dimensions.set(
            {("allgather", str(self._all_gather_process_group.group_name)): "fsdp"}
        )
        capture = get_active_capture()
        scaffold_scope = (
            capture.annotate_dispatch_events("fsdp_unshard_scaffold") if capture is not None else nullcontext()
        )
        try:
            # Retain FSDP buffer operations alongside communication semantics.
            with scaffold_scope:
                result = original_unshard(self, async_op)
        finally:
            _active_fsdp_group_dimensions.reset(group_token)
            _active_fsdp_communication_context.reset(communication_token)
            _active_fsdp_transition.reset(token)
            meta_env._comm_layer = previous_comm_layer
        launched_collective = recorder is not None and any(
            event.fsdp_transition_id == transition_id for event in recorder.events[event_count_before:]
        )
        if launched_collective:
            self._sim_active_transition_id = transition_id
            self._sim_pending_transition_id = ""
            self._sim_prefetch_source_fqn = prefetch_source_fqn
            self._sim_prefetch_type = prefetch_type
        elif previous_transition_id or previous_pending_transition_id:
            pass
        elif not was_unsharded and self.is_unsharded:
            self._sim_active_transition_id = transition_id
        elif not was_unsharded:
            # Async unshard has been requested but residency starts only when
            # wait_for_unshard observes the state transition.
            self._sim_pending_transition_id = transition_id
            self._sim_prefetch_source_fqn = prefetch_source_fqn
            self._sim_prefetch_type = prefetch_type
        effective_transition_id = getattr(self, "_sim_active_transition_id", "") or getattr(
            self, "_sim_pending_transition_id", ""
        )
        if explicit_schedule_action:
            _record_residency(
                self,
                "alloc",
                (_residency_metadata(self) if _memory_tracking_enabled() else (0, ())),
                _shard_world_size(self),
                effective_transition_id,
                include_memory=False,
                schedule_source="intent",
            )
        return result

    def patched_wait_for_unshard(self):
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        if not _uses_fsdp_sharding(self):
            capture = get_active_capture()
            scaffold_scope = (
                capture.annotate_dispatch_events("replicate_unshard_wait_scaffold")
                if capture is not None
                else nullcontext()
            )
            with scaffold_scope:
                return original_wait_for_unshard(self)

        was_unsharded = self.is_unsharded
        shard_world_size = _shard_world_size(self)
        track_memory = _memory_tracking_enabled()
        pending = getattr(self, "_all_gather_result", None)
        transition = getattr(self, "_sim_active_transition_id", "")
        if track_memory and pending is not None:
            staging_owners[id(pending)] = (self, transition)
        _, sharded_tensors = _residency_metadata(self) if track_memory else (0, ())
        capture = get_active_capture()
        scaffold_scope = (
            capture.annotate_dispatch_events("fsdp_unshard_wait_scaffold") if capture is not None else nullcontext()
        )
        with scaffold_scope:
            result = original_wait_for_unshard(self)
        if not was_unsharded and self.is_unsharded:
            _set_stage_fsdp_state(meta_env, "UNSHARDED")
            if not getattr(self, "_sim_active_transition_id", ""):
                self._sim_active_transition_id = getattr(self, "_sim_pending_transition_id", "")
            self._sim_pending_transition_id = ""
        if not was_unsharded and self.is_unsharded:
            if track_memory:
                num_bytes, full_tensors = _residency_metadata(self)
                tensors = tuple({id(tensor): tensor for tensor in (*sharded_tensors, *full_tensors)}.values())
                metadata = (num_bytes, tensors)
            else:
                metadata = (0, ())
            self._sim_residency_shard_world_size = shard_world_size
            _record_residency(
                self,
                "alloc",
                metadata,
                shard_world_size,
                getattr(self, "_sim_active_transition_id", ""),
            )
        if track_memory and pending is not None:
            deferred = getattr(self.comm_ctx, "all_gather_state", None)
            # Native wait keeps its previous state in a local variable through
            # copy-out, even after clearing comm_ctx.all_gather_state. Release
            # it after the new full-parameter allocation, not at assignment.
            flush_deferred_releases()
            retained = deferred is not None and deferred.all_gather_result is pending
            if not retained and getattr(self, "_all_gather_result", None) is not pending:
                staging_owners.pop(id(pending), None)
                record_staging_release(self, pending, transition)
        return result

    def patched_finalize_backward(self):
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        pending = getattr(self, "_all_gather_result", None)
        transition = getattr(self, "_sim_active_transition_id", "")
        capture = get_active_capture()
        with (
            capture.annotate_dispatch_events("fsdp_finalize_backward_scaffold")
            if capture is not None
            else nullcontext()
        ):
            result = original_finalize_backward(self)
        if pending is not None and getattr(self, "_all_gather_result", None) is None:
            staging_owners.pop(id(pending), None)
            record_staging_release(self, pending, transition)
        return result

    def patched_reshard(self):
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        if not _uses_fsdp_sharding(self):
            capture = get_active_capture()
            scaffold_scope = (
                capture.annotate_dispatch_events("replicate_reshard_scaffold") if capture is not None else nullcontext()
            )
            with scaffold_scope:
                return original_reshard(self)

        previous_comm_layer = meta_env._comm_layer
        meta_env._comm_layer = "L2"
        was_unsharded = self.is_unsharded
        metadata: tuple[int, tuple[torch.Tensor, ...]] = (
            _residency_metadata(self) if was_unsharded and _memory_tracking_enabled() else (0, ())
        )
        transition_id = getattr(self, "_sim_active_transition_id", "")
        explicit_schedule_action = meta_env._pp_context.get("comp_type") == "RESHARD"
        if explicit_schedule_action:
            _record_residency(
                self,
                "free",
                metadata,
                _shard_world_size(self),
                transition_id,
                include_memory=False,
                schedule_source="intent",
            )
        capture = get_active_capture()
        scaffold_scope = (
            capture.annotate_dispatch_events("fsdp_reshard_scaffold") if capture is not None else nullcontext()
        )
        try:
            with scaffold_scope:
                result = original_reshard(self)
        finally:
            meta_env._comm_layer = previous_comm_layer
        if was_unsharded and not self.is_unsharded:
            _set_stage_fsdp_state(meta_env, "SHARDED")
            shard_world_size = getattr(self, "_sim_residency_shard_world_size", None)
            if shard_world_size is None:
                shard_world_size = _shard_world_size(self)
            _record_residency(
                self,
                "free",
                metadata,
                shard_world_size,
                transition_id,
            )
            self._sim_active_transition_id = ""
            self._sim_pending_transition_id = ""
            self._sim_prefetch_source_fqn = ""
            self._sim_prefetch_type = ""
        elif explicit_schedule_action:
            self._sim_pending_transition_id = ""
        return result

    def patched_post_backward(self, *args: Any, **kwargs: Any):
        """Observe native FSDP reduction operations without hiding dispatch facts.

        Communication ownership is annotated separately. Casts, accumulation,
        packing and aliases retain their actual tensor and storage dependencies.
        """
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        capture = get_active_capture()
        group_id = str(id(self))
        module_fqn = _module_fqn(self)
        transition_token = _active_fsdp_transition.set(("", group_id))
        communication_token = _active_fsdp_communication_context.set((module_fqn, "", ""))
        dependency_token = _active_fsdp_reduction_dependencies.set(_post_backward_dependency_tensors(self))
        groups = []
        group_dimensions = {}
        if _uses_fsdp_sharding(self):
            groups.append(self._reduce_scatter_process_group)
            group_dimensions[("reduce_scatter", str(groups[-1].group_name))] = "fsdp"
        if hasattr(self.mesh_info, "replicate_process_group"):
            groups.append(self._all_reduce_process_group)
            group_dimensions[("allreduce", str(groups[-1].group_name))] = "dp_replicate"
        group_token = _active_fsdp_reduction_groups.set(frozenset(str(group.group_name) for group in groups))
        dimension_token = _active_fsdp_group_dimensions.set(group_dimensions)
        scaffold_scope = (
            capture.annotate_dispatch_events("fsdp_post_backward_scaffold") if capture is not None else nullcontext()
        )
        try:
            with scaffold_scope:
                return original_post_backward(
                    self,
                    *args,
                    **kwargs,
                )
        finally:
            _active_fsdp_group_dimensions.reset(dimension_token)
            _active_fsdp_reduction_groups.reset(group_token)
            _active_fsdp_reduction_dependencies.reset(dependency_token)
            _active_fsdp_communication_context.reset(communication_token)
            _active_fsdp_transition.reset(transition_token)

    def _state_module_fqn(state: Any) -> str:
        for param_group in getattr(state, "_fsdp_param_groups", ()):
            if module_fqn := _module_fqn(param_group):
                return module_fqn
        return ""

    def patched_state_pre_forward(self, *args, **kwargs):
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        capture = get_active_capture()
        token = _active_fsdp_state_source.set(_state_module_fqn(self))
        try:
            with (
                capture.annotate_dispatch_events("fsdp_state_pre_forward_scaffold")
                if capture is not None
                else nullcontext()
            ):
                return orig_state_pre_forward(self, *args, **kwargs)
        finally:
            _active_fsdp_state_source.reset(token)

    def patched_state_pre_backward(self, *args, **kwargs):
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        capture = get_active_capture()
        token = _active_fsdp_state_source.set(_state_module_fqn(self))
        try:
            with (
                capture.annotate_dispatch_events("fsdp_state_pre_backward_scaffold")
                if capture is not None
                else nullcontext()
            ):
                return orig_state_pre_backward(self, *args, **kwargs)
        finally:
            _active_fsdp_state_source.reset(token)

    def patched_state_post_forward(self, *args, **kwargs):
        from torchtitan_simulator.capture.dispatch_capture import get_active_capture

        capture = get_active_capture()
        with (
            capture.annotate_dispatch_events("fsdp_state_post_forward_scaffold")
            if capture is not None
            else nullcontext()
        ):
            result = original_state_post_forward(self, *args, **kwargs)
        flush_deferred_releases()
        return result

    def patched_prefetch_unshard(
        target_param_group,
        pass_type,
    ):
        source_fqn = _active_fsdp_state_source.get()
        token = _active_fsdp_prefetch_source.set((source_fqn, str(pass_type)))
        try:
            return orig_prefetch_unshard(target_param_group, pass_type)
        finally:
            _active_fsdp_prefetch_source.reset(token)

    scope.replace(FSDPCommContext, "__setattr__", patched_comm_setattr)
    scope.replace(FSDPParamGroup, "finalize_backward", patched_finalize_backward)
    scope.replace(FSDPParamGroup, "unshard", patched_unshard)
    scope.replace(FSDPParamGroup, "wait_for_unshard", patched_wait_for_unshard)
    scope.replace(FSDPParamGroup, "reshard", patched_reshard)
    scope.replace(FSDPParamGroup, "post_backward", patched_post_backward)
    scope.replace(FSDPParamGroup, "_prefetch_unshard", staticmethod(patched_prefetch_unshard))
    scope.replace(FSDPState, "_pre_forward", patched_state_pre_forward)
    scope.replace(FSDPState, "_pre_backward", patched_state_pre_backward)
    scope.replace(FSDPState, "_post_forward", patched_state_post_forward)
