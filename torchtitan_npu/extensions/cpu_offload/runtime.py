# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared CPU-offload transfer and gradient helpers."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.distributed._tensor import DTensor

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging, TransferHandle


@dataclass(frozen=True, slots=True)
class _TensorMetadata:
    storage_ptr: int
    storage_offset: int
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> _TensorMetadata:
        return cls(
            tensor.untyped_storage().data_ptr(),
            int(tensor.storage_offset()),
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
        )


@dataclass(frozen=True, slots=True)
class _CachedGradient:
    parameter: Tensor
    cpu_metadata: _TensorMetadata
    working: Tensor


@dataclass(slots=True)
class _PendingClip:
    coefficient: Tensor
    ready_event: Any | None
    gradients: dict[int, _CachedGradient]
    corrections: list[float] = field(default_factory=list)


class GradientClipChannel:
    """Explicit handoff of one step's clip result to the optimizers.

    The CPU-offload optimizer container owns the channel: it injects the
    channel into the optimizers it builds (like the staging lane) and
    exposes it to the patched ``clip_grad_norm_`` through the module-level
    active handle — the single residual implicit link, required because the
    upstream trainer invokes clip and ``optimizer.step`` as independent
    calls with no data path between them.
    """

    def __init__(self, *, preserve_coefficient_dtype: bool = False) -> None:
        self._preserve_coefficient_dtype = preserve_coefficient_dtype
        self._consumers = 0
        self._pending: _PendingClip | None = None

    # -- consumer registration (cached vs bounded clip path) ----------------

    def register_consumer(self) -> None:
        self._consumers += 1

    def unregister_consumer(self) -> None:
        self._consumers = max(0, self._consumers - 1)

    def has_consumer(self) -> bool:
        return self._consumers > 0

    # -- clip side (publisher) ----------------------------------------------

    def publish(
        self,
        entries: list[tuple[Tensor, Tensor, Tensor]],
        coefficient: Tensor,
        ready_event: Any | None,
    ) -> None:
        """Publish one step's NPU gradients for optimizer consumption."""
        if self._pending is not None:
            raise RuntimeError("clip gradients are already active")
        if coefficient.numel() != 1:
            raise ValueError("clip coefficient must be a scalar tensor")

        gradients: dict[int, _CachedGradient] = {}
        for parameter, cpu_gradient, working_gradient in entries:
            key = id(parameter)
            if key in gradients:
                raise ValueError("duplicate parameter in clip gradient cache")
            if cpu_gradient.device.type != "cpu":
                raise ValueError("clip gradient source must be on CPU")
            if working_gradient.device != coefficient.device:
                raise ValueError("working gradient and clip coefficient must use the same device")
            if working_gradient.shape != cpu_gradient.shape or working_gradient.dtype != cpu_gradient.dtype:
                raise ValueError("CPU and working gradients must have matching shape and dtype")
            gradients[key] = _CachedGradient(
                parameter,
                _TensorMetadata.from_tensor(cpu_gradient),
                working_gradient,
            )

        self._pending = _PendingClip(coefficient.reshape(()), ready_event, gradients)

    # -- optimizer side (consumer) -------------------------------------------

    def take_gradient(self, parameter: Tensor, cpu_gradient: Tensor) -> Tensor | None:
        """Take the cached working gradient for ``parameter`` exactly once."""
        state = self._pending
        if state is None:
            return None

        entry = state.gradients.get(id(parameter))
        if entry is None or entry.parameter is not parameter:
            raise RuntimeError("active clip state has no cached gradient for parameter")
        if entry.cpu_metadata != _TensorMetadata.from_tensor(cpu_gradient):
            raise RuntimeError("CPU gradient changed after clip staging")

        del state.gradients[id(parameter)]
        return entry.working

    def apply_pending(self, tensor: Tensor) -> Tensor:
        """Order after clip staging on the current stream and scale in place.

        The wait and the scaling are both issued on the *current* stream:
        eager ops always enqueue on the current stream, so a wait issued on
        any other stream cannot order this ``mul_``.
        """
        state = self._pending
        if state is None:
            return tensor
        if state.coefficient.device != tensor.device:
            raise RuntimeError(
                f"clip coefficient is on {state.coefficient.device}, working gradient is on {tensor.device}"
            )
        if state.ready_event is not None:
            current = torch.get_device_module(tensor.device).current_stream(tensor.device)
            current.wait_event(state.ready_event)
        # HostSparse follows eager coefficient precision even on steps without
        # a sparse correction. Ordinary offload retains its existing cast.
        coefficient = (
            state.coefficient if self._preserve_coefficient_dtype else state.coefficient.to(dtype=tensor.dtype)
        )
        tensor.mul_(coefficient)
        for correction in state.corrections:
            tensor.mul_(correction)
        return tensor

    def rescale_pending(self, correction: float) -> bool:
        """Include additional gradient norms before optimizers consume the cache."""
        state = self._pending
        if state is None:
            return False
        # Eager HostSparse scales gradients twice. Folding the factors into
        # one coefficient changes rounding (and can change Muon's direction).
        state.corrections.append(correction)
        return True

    # -- lifecycle ------------------------------------------------------------

    def clear_pending(self) -> None:
        self._pending = None

    def close(self) -> None:
        """Release the channel state; idempotent, resets consumers."""
        self.clear_pending()
        self._consumers = 0


_ACTIVE_CHANNEL: ContextVar[GradientClipChannel | None] = ContextVar(
    "torchtitan_npu_active_clip_channel",
    default=None,
)


def set_active_channel(channel: GradientClipChannel | None) -> None:
    """Expose the container's channel to the patched clip_grad_norm_."""
    _ACTIVE_CHANNEL.set(channel)


def get_active_channel() -> GradientClipChannel | None:
    return _ACTIVE_CHANNEL.get()


def local_tensor(tensor: Tensor) -> Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def require_cpu_tensor(tensor: Tensor, *, what: str) -> Tensor:
    local = local_tensor(tensor)
    if local.device.type != "cpu":
        raise ValueError(f"CPU-offloaded {what} requires CPU storage, got {local.device}")
    return local


@torch.no_grad()
def stage_gradient(
    parameter: Tensor,
    cpu_gradient: Tensor,
    destination: Tensor | Callable[[], Tensor],
    *,
    staging: CpuStaging,
    channel: GradientClipChannel | None = None,
    stream: Any,
) -> tuple[Tensor, tuple[TransferHandle, ...]]:
    """Use clip's staged NPU gradient when available, otherwise copy from CPU.

    ``channel`` supplies clip's cached NPU gradient and pending coefficient;
    ``stream`` only selects the transfer stream for the CPU→NPU copy. The
    clip-coefficient scaling and the copy-out always run on the caller's
    current stream, so the returned tensor is ready for any op enqueued on
    that stream after waiting the returned handles there.
    """
    working = channel.take_gradient(parameter, cpu_gradient) if channel is not None else None
    if callable(destination):
        destination = destination()
    if working is not None:
        working = working.view_as(destination)
        if channel is None:
            raise RuntimeError("cached gradient without a clip channel")
        channel.apply_pending(working)
        destination.copy_(working, non_blocking=True)
        if working.device.type == "npu":
            current = torch.get_device_module(working.device).current_stream(working.device)
            working.record_stream(current)
        return destination, ()

    handle = staging.submit_h2d(cpu_gradient, destination, stream=stream)
    current = torch.get_device_module(destination.device).current_stream(destination.device)
    handle.wait_on(current)
    if channel is not None:
        channel.apply_pending(destination)
    return destination, (handle,)
