# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

import queue
import threading
from typing import Any, cast

import torch
import torch_npu

from .storage.pinned_cpu_memory_pool import PinnedCpuStorage

# TORCHTITAN-NPU MOD: resolve the copied backend from the plugin package.
# Remove when: this backend is moved to a shared upstream swap package.
from .swap_ir import ActionManager
from .swap_primitive import (
    SwapHandle,
    d2h,
    h2d,
    validate_tensor_for_swap,
)
from .swap_registry import SwapRegistry


class _AsyncReleaseWorker:
    """Release event-protected NPU or CPU storage on a background thread."""

    _STOP = object()

    def __init__(self) -> None:
        self._queue: queue.Queue[object] = queue.Queue()
        self._condition = threading.Condition()
        self._entries: dict[int, tuple[str, SwapHandle, str, str]] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="unified-swap-release-worker",
            daemon=True,
        )
        self._thread.start()

    def enqueue(self, tensor_name: str, handle: SwapHandle, release_target: str) -> None:
        if release_target not in {"cpu", "npu"}:
            raise ValueError(f"unsupported release target {release_target!r}")
        handle_id = id(handle)
        with self._condition:
            entry = self._entries.get(handle_id)
            if entry is not None and entry[2] == "pending":
                return
            self._entries[handle_id] = (
                tensor_name,
                handle,
                "pending",
                release_target,
            )
            self._condition.notify_all()
        self._queue.put(handle_id)

    def cancel_pending_release(self, tensor_name: str, handle: SwapHandle) -> None:
        handle_id = id(handle)
        with self._condition:
            entry = self._entries.get(handle_id)
            if entry is None:
                return
            entry_tensor, entry_handle, state, release_target = entry
            if entry_tensor != tensor_name or entry_handle is not handle:
                return
            if state == "pending":
                self._entries[handle_id] = (
                    entry_tensor,
                    entry_handle,
                    "cancelled",
                    release_target,
                )
                self._condition.notify_all()
            elif state == "released":
                self._entries.pop(handle_id, None)

    def drain_completed(self) -> None:
        with self._condition:
            finished = [
                handle_id for handle_id, (_, _, state, _) in self._entries.items() if state in {"released", "cancelled"}
            ]
            for handle_id in finished:
                self._entries.pop(handle_id, None)

    def wait_for_name(self, tensor_name: str) -> None:
        """Wait until retired storage owned by one tensor name is released."""
        with self._condition:
            while any(
                entry_tensor == tensor_name and state == "pending"
                for entry_tensor, _, state, _ in self._entries.values()
            ):
                self._condition.wait()
            finished = [
                handle_id
                for handle_id, (entry_tensor, _, state, _) in self._entries.items()
                if entry_tensor == tensor_name and state in {"released", "cancelled"}
            ]
            for handle_id in finished:
                self._entries.pop(handle_id, None)

    def clear(self) -> None:
        with self._condition:
            self._entries.clear()
            self._condition.notify_all()

    def shutdown(self) -> None:
        self.clear()
        self._queue.put(self._STOP)
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._STOP:
                    return
                handle_id = cast("int", item)
                entry = self._entry_if_pending(handle_id)
                if entry is None:
                    continue
                tensor_name, handle = entry
                if not self._wait_while_pending(handle_id, handle):
                    continue
                with self._condition:
                    entry = self._entries.get(handle_id)
                    if entry is None:
                        continue
                    entry_tensor, entry_handle, state, release_target = entry
                    if entry_tensor != tensor_name or entry_handle is not handle or state != "pending":
                        continue
                    if release_target == "npu":
                        _release_npu(handle)
                    else:
                        _release_cpu(handle)
                    self._entries[handle_id] = (
                        entry_tensor,
                        entry_handle,
                        "released",
                        release_target,
                    )
                    self._condition.notify_all()
            finally:
                self._queue.task_done()

    def _entry_if_pending(self, handle_id: int) -> tuple[str, SwapHandle] | None:
        with self._condition:
            entry = self._entries.get(handle_id)
            if entry is None:
                return None
            tensor_name, handle, state, _ = entry
            if state != "pending":
                return None
            return tensor_name, handle

    def _wait_while_pending(self, handle_id: int, handle: SwapHandle) -> bool:
        event = handle.swap_event
        if event is None:
            handle.is_completed = True
            return self._is_pending(handle_id, handle)

        if not self._is_pending(handle_id, handle):
            return False
        event.synchronize()
        handle.is_completed = True
        return self._is_pending(handle_id, handle)

    def _is_pending(self, handle_id: int, handle: SwapHandle) -> bool:
        with self._condition:
            return self._is_pending_locked(handle_id, handle)

    def _is_pending_locked(self, handle_id: int, handle: SwapHandle) -> bool:
        entry = self._entries.get(handle_id)
        if entry is None:
            return False
        _, entry_handle, state, _ = entry
        return entry_handle is handle and state == "pending"


class SwapEngine:
    """Execute actions resolved by globally unique tensor names."""

    _offload_stream = None
    _prefetch_stream = None
    _release_worker: _AsyncReleaseWorker | None = None
    _handles: dict[str, tuple[SwapHandle, ...]] = {}
    _ready = False

    @classmethod
    def _init(cls) -> None:
        if cls._ready:
            return
        cls._offload_stream = torch_npu.npu.Stream(device=torch_npu.npu.current_device())
        cls._prefetch_stream = torch_npu.npu.Stream(device=torch_npu.npu.current_device())
        cls._release_worker = _AsyncReleaseWorker()
        cls._handles = {}
        cls._ready = True

    @classmethod
    def on_operator_hook(cls, hook_name: str) -> None:
        if not cls._ready:
            cls._init()
        action_list = ActionManager.get_actions_by_name(hook_name)
        for action in action_list:
            cls.execute(action)

    @classmethod
    def execute(cls, action: Any) -> None:
        if not cls._ready:
            cls._init()
        release_worker = cls._release_worker
        if release_worker is None:
            raise RuntimeError("SwapEngine release worker is not initialized")
        tensor_name = action.tensor
        action_type = str(action.type).upper()

        if action_type == "D2H":
            entry = SwapRegistry.get_entry_by_name(tensor_name)
            if entry is None:
                raise RuntimeError(f"Swap tensor is not available: {tensor_name}")
            previous_handles = cls._handles.get(tensor_name)
            if previous_handles is not None:
                if not all(handle.transfer == "H2D" and handle.device_waited for handle in previous_handles):
                    raise AssertionError(f"D2H requires a WAIT_DEVICE-completed H2D handle for {tensor_name!r}")
                if len(previous_handles) != len(entry.storage_tensors):
                    raise AssertionError(f"Swap handle group size changed for {tensor_name!r}")

            tensors = entry.storage_tensors

            detached_tensors = tuple(tensor.detach() for tensor in tensors)
            for tensor in detached_tensors:
                validate_tensor_for_swap(tensor)
            _reject_duplicate_storage(detached_tensors, tensor_name)

            handles = []
            try:
                for tensor in detached_tensors:
                    # Persistent state uses a fresh destination. WAIT_DEVICE is
                    # device-side only and does not make the old H2D source safe
                    # for immediate host-side reuse.
                    handles.append(d2h(tensor, cls._offload_stream, owner=tensor_name))
            except Exception:
                for handle in handles:
                    _wait(handle)
                    _release_cpu(handle)
                raise

            group_handles = tuple(handles)
            if previous_handles is not None:
                for handle in previous_handles:
                    release_worker.enqueue(tensor_name, handle, "cpu")
            cls._handles[tensor_name] = group_handles
            for handle in group_handles:
                release_worker.enqueue(tensor_name, handle, "npu")
            return

        if action_type == "H2D":
            handles = cls._handles.get(tensor_name)
            if handles is None:
                raise AssertionError(f"H2D lacks a live D2H handle for {tensor_name!r}")
            if all(handle.transfer == "H2D" for handle in handles):
                return
            if not all(handle.transfer == "D2H" for handle in handles):
                raise AssertionError(f"Mixed swap handle phases for {tensor_name!r}")
            for handle in handles:
                release_worker.cancel_pending_release(tensor_name, handle)
            release_worker.drain_completed()
            restored_handles = list(handles)
            for index, handle in enumerate(handles):
                try:
                    restored_handles[index] = h2d(handle, cls._prefetch_stream)
                except Exception:
                    cls._handles[tensor_name] = tuple(restored_handles)
                    raise
            cls._handles[tensor_name] = tuple(restored_handles)
            return

        if action_type == "WAIT_DEVICE":
            handles = cls._handles.get(tensor_name)
            if handles is None:
                raise AssertionError(f"WAIT_DEVICE lacks handles for {tensor_name}")
            if not all(handle.transfer == "H2D" for handle in handles):
                raise AssertionError(f"WAIT_DEVICE requires live H2D handles for {tensor_name!r}")
            if all(handle.device_waited for handle in handles):
                return
            current_stream = torch_npu.npu.current_stream()
            for handle in handles:
                if handle.swap_event is not None:
                    current_stream.wait_event(handle.swap_event)
                handle.device_waited = True
            return

        if action_type == "WAIT":
            handles = cls._handles.pop(tensor_name, None)
            if handles is None:
                return
            first_error: Exception | None = None
            for handle in handles:
                try:
                    _wait(handle)
                except Exception as exc:
                    first_error = first_error or exc
            for handle in handles:
                try:
                    _release_cpu(handle)
                except Exception as exc:
                    first_error = first_error or exc
            if first_error is not None:
                raise first_error
            return

        raise ValueError(f"Unknown swap action type: {action_type}")

    @classmethod
    def reset(cls) -> None:
        if not cls._ready:
            return
        release_worker = cls._release_worker
        if release_worker is None:
            raise RuntimeError("SwapEngine release worker is not initialized")
        cleanup_error: Exception | None = None
        for tensor_name in list(cls._handles):
            try:
                cls.remove(tensor_name)
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        release_worker.drain_completed()
        if cleanup_error is not None:
            raise RuntimeError("Failed to reset one or more swap handles") from cleanup_error

    @classmethod
    def handle_phase(cls, tensor_name: str) -> str | None:
        handles = cls._handles.get(tensor_name)
        if handles is None:
            return None
        phases = {handle.transfer for handle in handles}
        if len(phases) != 1:
            raise RuntimeError(f"Mixed swap handle phases for {tensor_name!r}")
        phase = phases.pop()
        if phase == "H2D" and all(handle.device_waited for handle in handles):
            return "H2D_WAITED"
        return phase

    @classmethod
    def remove(cls, tensor_name: str) -> None:
        handles = cls._handles.pop(tensor_name, None)
        first_error: Exception | None = None
        if handles is not None:
            for handle in handles:
                if cls._release_worker is not None:
                    cls._release_worker.cancel_pending_release(tensor_name, handle)
                try:
                    _wait(handle)
                    _release_cpu(handle)
                except Exception as error:
                    first_error = first_error or error
        if cls._release_worker is not None:
            cls._release_worker.wait_for_name(tensor_name)
        if first_error is not None:
            raise RuntimeError(f"Failed to remove swap tensor {tensor_name!r}") from first_error

    @classmethod
    def shutdown(cls) -> None:
        cleanup_error: Exception | None = None
        if cls._ready:
            try:
                cls.reset()
            except Exception as exc:
                cleanup_error = exc
        if cls._release_worker is not None:
            try:
                cls._release_worker.shutdown()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        try:
            PinnedCpuStorage.shutdown()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc

        cls._offload_stream = None
        cls._prefetch_stream = None
        cls._release_worker = None
        cls._handles = {}
        cls._ready = False
        if cleanup_error is not None:
            raise RuntimeError("Failed to shut down unified swap cleanly") from cleanup_error


def _wait(handle: SwapHandle) -> None:
    swap_event = handle.swap_event
    if swap_event is not None and not handle.is_completed:
        swap_event.synchronize()
    handle.is_completed = True


def _release_cpu(handle: SwapHandle) -> None:
    tensor_cpu = handle.tensor_cpu
    if tensor_cpu is not None:
        handle.tensor_cpu = None
        PinnedCpuStorage.free(tensor_cpu, owner=handle.owner)


def _release_npu(handle: SwapHandle) -> None:
    if handle.tensor_npu is not None:
        handle.tensor_npu.untyped_storage().resize_(0)


def _reject_duplicate_storage(tensors: tuple[torch.Tensor, ...], tensor_name: str) -> None:
    """Reject aliases because two release workers must not own one storage."""
    seen = {}
    for index, tensor in enumerate(tensors):
        storage = tensor.untyped_storage()
        storage_size = storage.size()
        key = (
            (str(tensor.device), storage.data_ptr(), storage_size)
            if storage_size > 0
            else (str(tensor.device), id(tensor), storage_size)
        )
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                f"Unified swap group '{tensor_name}' contains aliased storage at indices {previous} and {index}"
            )
        seen[key] = index
