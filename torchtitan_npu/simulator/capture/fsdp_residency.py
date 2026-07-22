# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Explicit FSDP unshard/reshard residency capture."""

from __future__ import annotations

from typing import Any

import torch


def _memory_tracking_enabled() -> bool:
    from torchtitan_npu.simulator.capture.comm_events import get_active_recorder

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


def _record_residency(param_group: Any, action: str, metadata: tuple[int, tuple[torch.Tensor, ...]]) -> None:
    from torchtitan_npu.simulator.capture.comm_events import get_active_recorder
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    recorder = get_active_recorder()
    capture = get_active_capture()
    num_bytes, tensors = metadata
    if recorder is None:
        return
    tensor_ids = tuple(
        sorted(capture.tensor_id(tensor) if capture is not None else id(tensor) for tensor in tensors)
    )
    training_state = str(getattr(param_group, "_training_state", "")).lower()
    phase = "backward" if "backward" in training_state else "forward"
    recorder.record_fsdp_residency(
        group_id=str(id(param_group)),
        action=action,
        phase=phase,
        num_bytes=num_bytes,
        tensor_ids=tensor_ids,
    )


def _set_stage_fsdp_state(meta_env: Any, state: str) -> None:
    """Synchronize the PP capture context with an FSDP state transition."""
    stage = meta_env._pp_context.get("stage", -1)
    if isinstance(stage, int):
        meta_env._fsdp_state[stage] = state


def install_fsdp_residency_hooks() -> None:
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    import torchtitan_npu.simulator.meta_env as meta_env

    if hasattr(FSDPParamGroup, "_sim_orig_unshard"):
        return

    FSDPParamGroup._sim_orig_unshard = FSDPParamGroup.unshard
    FSDPParamGroup._sim_orig_wait_for_unshard = FSDPParamGroup.wait_for_unshard
    FSDPParamGroup._sim_orig_reshard = FSDPParamGroup.reshard

    def patched_unshard(self, async_op=False):  # noqa: ANN001, ANN202
        previous_comm_layer = meta_env._comm_layer
        meta_env._comm_layer = "L2"
        try:
            return FSDPParamGroup._sim_orig_unshard(self, async_op)
        finally:
            meta_env._comm_layer = previous_comm_layer

    def patched_wait_for_unshard(self):  # noqa: ANN001, ANN202
        was_unsharded = self.is_unsharded
        track_memory = _memory_tracking_enabled()
        _, sharded_tensors = _residency_metadata(self) if track_memory else (0, ())
        result = FSDPParamGroup._sim_orig_wait_for_unshard(self)
        if not was_unsharded and self.is_unsharded:
            _set_stage_fsdp_state(meta_env, "UNSHARDED")
        if not was_unsharded and self.is_unsharded:
            if track_memory:
                num_bytes, full_tensors = _residency_metadata(self)
                tensors = tuple({id(tensor): tensor for tensor in (*sharded_tensors, *full_tensors)}.values())
                metadata = (num_bytes, tensors)
            else:
                metadata = (0, ())
            _record_residency(self, "alloc", metadata)
        return result

    def patched_reshard(self):  # noqa: ANN001, ANN202
        previous_comm_layer = meta_env._comm_layer
        meta_env._comm_layer = "L2"
        was_unsharded = self.is_unsharded
        metadata: tuple[int, tuple[torch.Tensor, ...]] = (
            _residency_metadata(self) if was_unsharded and _memory_tracking_enabled() else (0, ())
        )
        try:
            result = FSDPParamGroup._sim_orig_reshard(self)
        finally:
            meta_env._comm_layer = previous_comm_layer
        if was_unsharded and not self.is_unsharded:
            _set_stage_fsdp_state(meta_env, "SHARDED")
            _record_residency(self, "free", metadata)
        return result

    FSDPParamGroup.unshard = patched_unshard
    FSDPParamGroup.wait_for_unshard = patched_wait_for_unshard
    FSDPParamGroup.reshard = patched_reshard
