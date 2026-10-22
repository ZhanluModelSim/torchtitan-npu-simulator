# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device/stream protocol emulation for storage-free Meta execution."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any

import torch


class MetaDeviceModule:
    """Minimal stand-in for `torch.cuda`/`torch_npu`, covering every method
    actually called on `device_module` by `torchtitan.trainer`,
    `torchtitan.components.metrics`, and `torchtitan.distributed.utils`
    (verified against the pinned torchtitan commit -- see design doc §5.1),
    plus the subset of the `torch.<device_type>` module API that
    `torch.distributed.device_mesh._get_device_handle`/FSDP2 resolve via
    `getattr(torch, device_type)` (`current_device`, `is_available`,
    `is_initialized`, `current_stream`, `stream`, `memory_summary`,
    `Event`, `Stream`)."""

    name = "Meta_Simulator"

    def __init__(self):
        self._stream_context: ContextVar[MetaDeviceModule.Stream | None] = ContextVar(
            "simulator_device_stream", default=None
        )

    def set_device(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def empty_cache(self) -> None:
        return None

    def reset_peak_memory_stats(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def get_device_name(self, *_args: Any, **_kwargs: Any) -> str:
        return self.name

    def get_device_properties(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
        # total_memory must be non-zero: torchtitan's own
        # DeviceMemoryMonitor._to_pct() unconditionally divides by it
        # (`100 * memory / self.device_capacity`) when computing display
        # percentages. Since `memory_stats()` below always reports 0 bytes
        # under meta simulation (no real memory is ever allocated), the
        # resulting percentage is always 0% regardless of this
        # denominator's exact value -- it exists purely to avoid
        # ZeroDivisionError, not to represent a real capacity.
        return SimpleNamespace(total_memory=1, name=self.name)

    def memory_stats(self, *_args: Any, **_kwargs: Any) -> dict[str, int]:
        return {
            "active_bytes.all.peak": 0,
            "reserved_bytes.all.peak": 0,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_rng_state(self, *_args: Any, **_kwargs: Any) -> torch.Tensor:
        # `torch.random.fork_rng` (used by pipeline schedule stage init)
        # calls `device_mod.get_rng_state(device)`.  Returning the host RNG
        # state is sufficient: meta simulation never relies on device RNG,
        # fork_rng only needs a storable/restorable byte buffer.
        return torch.get_rng_state()

    def set_rng_state(self, state: torch.Tensor, *_args: Any, **_kwargs: Any) -> None:
        # Pair with `get_rng_state`.  Ignore the restore under meta sim.
        return None

    def current_stream(self, *_args: Any, **_kwargs: Any) -> MetaDeviceModule.Stream:
        stream = self._stream_context.get()
        if stream is None:
            stream = self.Stream()
            self._stream_context.set(stream)
        return stream

    @contextmanager
    def stream(self, stream=None):
        token = self._stream_context.set(stream or self.current_stream())
        try:
            yield
        finally:
            self._stream_context.reset(token)

    def memory_summary(self, *_args: Any, **_kwargs: Any) -> str:
        return ""

    class Event:
        """Matches `torch.Event`'s generic API surface (`query`, `record`,
        `synchronize`, `wait`, `elapsed_time`) -- FSDP2's collectives code
        (`torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py`) calls
        these on the Event objects returned by `Stream.record_event()`."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def query(self) -> bool:
            return True

        def record(self, *_args: Any, **_kwargs: Any):
            return self

        def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def elapsed_time(self, *_args: Any, **_kwargs: Any) -> float:
            return 0.0

    class Stream:
        """Matches `torch.Stream`'s generic API surface (`query`,
        `record_event`, `synchronize`, `wait_event`, `wait_stream`) --
        FSDP2's `foreach_all_gather`/`foreach_reduce` call these directly
        on `device_module.Stream()`/`device_handle.current_stream()`
        instances during every forward/backward pass, independent of the
        `observe_collectives()` interception layer (Task 10), which
        only covers `torch.distributed`-level collective calls."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def query(self) -> bool:
            return True

        def record_event(self, event: MetaDeviceModule.Event | None = None) -> MetaDeviceModule.Event:
            event = event or MetaDeviceModule.Event()
            event.record(self)
            return event

        def synchronize(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait_event(self, *_args: Any, **_kwargs: Any) -> None:
            return None

        def wait_stream(self, *_args: Any, **_kwargs: Any) -> None:
            return None
