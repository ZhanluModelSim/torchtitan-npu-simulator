# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded asynchronous transfers between CPU canonical and NPU tensors."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch_npu

from torchtitan_npu.extensions.novaswap.storage.pinned_cpu_memory_pool import (
    PinnedCpuStorage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_BOUNCE_SLOT_COUNT = 4


@dataclass(slots=True)
class TransferHandle:
    """Completion handle for one CPU↔NPU transfer."""

    event: Any
    _finalize: Callable[[], None] | None = None
    _completed: bool = False

    def wait_on(self, stream: Any) -> TransferHandle:
        """Order an NPU stream after this transfer without host synchronization."""
        if not self._completed:
            stream.wait_event(self.event)
        return self

    def wait(self) -> TransferHandle:
        """Synchronize the transfer and finish a pending host-side commit."""
        if self._completed:
            return self
        self.event.synchronize()
        try:
            if self._finalize is not None:
                self._finalize()
        finally:
            self._finalize = None
            self._completed = True
        return self


@dataclass(slots=True)
class _BounceSlot:
    raw: torch.Tensor | None = None
    handle: TransferHandle | None = None


class CpuStaging:
    """Stage logical CPU tensors through a bounded pinned-memory ring.

    Ownership and release protocol (one instance per owner — e.g. the
    optimizer container, one clip pipeline, one grad-accum pipeline):

    - The owner creates the staging and is the only caller of ``wait`` and
      ``close``; consumers merely submit transfers through it.
    - ``wait`` joins every ``track=True`` handle and releases the bounce
      ring; call it before reading CPU tensors written by D2H, or before
      letting go of any CPU canonical storage they alias.
    - Handles submitted with ``track=False`` are the caller's own
      responsibility and must be waited explicitly.
    - ``close`` is terminal: it joins tracked transfers, returns the pinned
      bounces to the pool, shuts down the pageable-commit executor thread,
      and makes any further submission raise.
    """

    def __init__(self, device: torch.device, *, owner: str) -> None:
        self.device = torch.device(device)  # pyrefly: ignore [read-only]
        if self.device.type != "npu" or self.device.index is None:
            raise ValueError(f"compute device must be an indexed NPU, got {self.device}")
        self.owner = owner
        self._stream = None
        self._executor: ThreadPoolExecutor | None = None
        self._bounce_slots = tuple(_BounceSlot() for _ in range(_BOUNCE_SLOT_COUNT))
        self._next_bounce_slot = 0
        self._pending: list[TransferHandle] = []
        self._closed = False

    @property
    def stream(self) -> torch_npu.npu.Stream:
        """The default stream used for deferred transfers."""
        if self._closed:
            raise RuntimeError(f"staging owner={self.owner!r} was closed")
        if self._stream is None:
            self._stream = torch_npu.npu.Stream(device=self.device)
        return self._stream

    @property
    def _completion_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cpu-staging",
            )
        return self._executor

    @staticmethod
    def validate_cpu_tensor(tensor: torch.Tensor) -> None:
        if tensor.device.type != "cpu":
            raise ValueError(f"staging requires a CPU canonical tensor, got {tensor.device}")

    @staticmethod
    def _validate_pair(
        cpu_tensor: torch.Tensor,
        npu_tensor: torch.Tensor,
        device: torch.device,
    ) -> None:
        CpuStaging.validate_cpu_tensor(cpu_tensor)
        if npu_tensor.device != device:
            raise ValueError(f"NPU working tensor must be on {device}, got {npu_tensor.device}")
        if cpu_tensor.shape != npu_tensor.shape or cpu_tensor.dtype != npu_tensor.dtype:
            raise ValueError("CPU canonical and NPU working tensors must have matching shape and dtype")

    @staticmethod
    def _complete_bounce(
        event: torch_npu.npu.Event,
        bounce: torch.Tensor,
        destination: torch.Tensor,
    ) -> None:
        event.synchronize()
        with torch.no_grad():
            destination.copy_(bounce)

    @staticmethod
    def _finish_slot(slot: _BounceSlot) -> None:
        if slot.handle is not None:
            slot.handle.wait()
            slot.handle = None

    def submit_h2d(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        stream: Any | None = None,
        track: bool = True,
    ) -> TransferHandle:
        """Queue CPU→NPU copy and return without waiting on the caller.

        ``stream`` lets an owner such as FSDP use its already-pipelined stream;
        when omitted, the staging stream is used. CPU sources are immediately
        ready, so no producer dependency is needed.
        """
        if self._closed:
            raise RuntimeError(f"staging owner={self.owner!r} was closed")
        self._validate_pair(source, destination, self.device)
        slot: _BounceSlot | None = None
        transfer_source = source
        if not source.is_pinned():
            slot, transfer_source = self._acquire_bounce(source)
            transfer_source.copy_(source)

        target = self.stream if stream is None else stream
        try:
            with torch.no_grad(), torch_npu.npu.stream(target):
                destination.copy_(transfer_source, non_blocking=True)
                complete = torch_npu.npu.Event()
                complete.record(target)
        except Exception:
            self._recover_transfer_error()
            raise

        handle = TransferHandle(complete)
        if slot is not None:
            slot.handle = handle
        if track:
            self._pending.append(handle)
        return handle

    def submit_d2h(
        self,
        source: torch.Tensor,
        destination: torch.Tensor,
        *,
        producer_stream: Any | None = None,
        stream: Any | None = None,
        track: bool = True,
    ) -> TransferHandle:
        """Queue NPU→CPU copy and return without waiting on the host."""
        if self._closed:
            raise RuntimeError(f"staging owner={self.owner!r} was closed")
        self._validate_pair(destination, source, self.device)
        slot: _BounceSlot | None = None
        transfer_destination = destination
        if not destination.is_pinned():
            slot, transfer_destination = self._acquire_bounce(destination)

        target = self.stream if stream is None else stream
        producer = producer_stream if producer_stream is not None else torch_npu.npu.current_stream(self.device)
        try:
            with torch.no_grad(), torch_npu.npu.stream(target):
                if producer is not target:
                    ready = torch_npu.npu.Event()
                    ready.record(producer)
                    target.wait_event(ready)
                transfer_destination.copy_(source, non_blocking=True)
                complete = torch_npu.npu.Event()
                complete.record(target)
        except Exception:
            self._recover_transfer_error()
            raise

        if slot is not None:
            future: Future[None] = self._completion_executor.submit(
                self._complete_bounce,
                complete,
                transfer_destination,
                destination,
            )
            handle = TransferHandle(complete, future.result)
            slot.handle = handle
        else:
            handle = TransferHandle(complete)
        if track:
            self._pending.append(handle)
        return handle

    def wait(self) -> None:
        """Finish all pending transfers and pageable host commits."""
        for handle in self._pending:
            handle.wait()
        self._pending.clear()
        self._release_bounces()

    def close(self) -> None:
        """Terminal release: join transfers, free bounces, stop the executor."""
        self.wait()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self._stream = None
        self._closed = True

    def _acquire_bounce(self, tensor: torch.Tensor) -> tuple[_BounceSlot, torch.Tensor]:
        slot = self._bounce_slots[self._next_bounce_slot]
        self._next_bounce_slot = (self._next_bounce_slot + 1) % len(self._bounce_slots)
        self._finish_slot(slot)

        size_bytes = tensor.numel() * tensor.element_size()
        if slot.raw is None or slot.raw.numel() < size_bytes:
            if slot.raw is not None:
                PinnedCpuStorage.free(slot.raw, owner=self.owner)
            slot.raw = PinnedCpuStorage.allocate(size_bytes, owner=self.owner)
        bounce = slot.raw[:size_bytes].view(tensor.dtype).view(tensor.shape)
        return slot, bounce

    def _release_bounces(self) -> None:
        for slot in self._bounce_slots:
            self._finish_slot(slot)
            if slot.raw is not None:
                PinnedCpuStorage.free(slot.raw, owner=self.owner)
                slot.raw = None
        self._next_bounce_slot = 0

    def _recover_transfer_error(self) -> None:
        if self._stream is not None:
            self.stream.synchronize()
        for handle in self._pending:
            with suppress(Exception):
                handle.wait()
        self._pending.clear()
        self._release_bounces()
