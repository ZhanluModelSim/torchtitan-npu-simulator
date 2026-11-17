# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flat pooled pinned host memory for FSDP CPU offload.

The torch_npu host pinned allocator rounds every allocation up to the
next power-of-two size class (measured on CANN 9.2.0: 12 MiB -> 16 MiB,
20 MiB -> 32 MiB, 600 MiB -> 1 GiB). FSDP's CPU offload pins one buffer
per parameter (inside to_empty()'s _apply -> reset_sharded_param) and
one per gradient (the non-blocking D2H offload) -- ~2500 requests per
rank for the 40-layer 32.6B crop model, which the rounding inflates by
~12.6 GiB per rank (~100 GB across 8 ranks).

A :class:`PinnedFlatPool` carves those buffers as views out of a small
number of power-of-two chunks, with 512-byte-aligned slice offsets.
Ownership follows the carved tensors:
every view holds its chunk alive through normal storage refcounting, so
the pool object itself is only an allocation cursor and may be dropped
once carving is done. There are no process-level singletons; rebuilding
a Trainer frees its chunks when the previous model's parameters and
gradient slots are released.

The ``Tensor.pin_memory`` interception is scoped, not just time-windowed:

- thread-local: only the thread that entered :meth:`pinned_pin_window`
  pools its calls; every other thread always runs the original
  implementation, even while a window is active;
- qualification-guarded: only CPU, contiguous, not-yet-pinned tensors
  are pooled; anything else is delegated to the original method with
  its arguments untouched (preserving stride/pinning/device semantics).

The guarded wrapper is installed once and never removed: without an
active window it delegates every call to the original implementation,
which also removes any install/restore ordering races.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_ORIG_PIN_MEMORY: Callable[..., torch.Tensor] | None = None

_MIN_CHUNK = 256 * 2**20
_MAX_CHUNK = 2 * 2**30
_ALIGNMENT = 512  # Match the NovaSwap pinned CPU pool alignment.

_state = threading.local()
_install_lock = threading.Lock()


def _pow2ceil(n: int) -> int:
    p = 1
    while p < n:
        p *= 2
    return p


def _install_guarded_wrapper() -> None:
    """Install the thread-local, qualification-guarded pin wrapper once."""
    global _ORIG_PIN_MEMORY
    with _install_lock:
        if _ORIG_PIN_MEMORY is not None:
            return
        original = torch.Tensor.pin_memory

        def _pooled_pin(self, *args, **kwargs):
            if args or kwargs:
                return original(self, *args, **kwargs)
            pool = getattr(_state, "pool", None)
            if pool is not None and self.device.type == "cpu" and self.is_contiguous() and not self.is_pinned():
                return pool.pin_copy(self)
            return original(self, *args, **kwargs)

        _ORIG_PIN_MEMORY = original
        torch.Tensor.pin_memory = _pooled_pin


class PinnedFlatPool:
    """Carves pinned CPU tensors from power-of-two chunks.

    The pool is an allocation cursor: chunks stay alive through the
    tensors carved from them, so the pool object may be dropped once
    carving is finished.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._chunks: list[torch.Tensor] = []
        self._next_chunk = _MIN_CHUNK
        self._cur: torch.Tensor | None = None
        self._offset = 0
        self.carved_bytes = 0

    def _grow(self, min_bytes: int) -> None:
        size = max(self._next_chunk, _pow2ceil(min_bytes))
        self._next_chunk = min(_MAX_CHUNK, max(_MIN_CHUNK, size * 2))
        if _ORIG_PIN_MEMORY is None:
            _install_guarded_wrapper()
        assert _ORIG_PIN_MEMORY is not None
        chunk = _ORIG_PIN_MEMORY(torch.empty(size, dtype=torch.uint8))
        self._chunks.append(chunk)
        self._cur = chunk
        self._offset = 0

    def allocate(self, shape, dtype: torch.dtype) -> torch.Tensor:
        """Return an uninitialized pinned tensor with the requested layout."""
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * dtype.itemsize
        padded = (nbytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
        if self._cur is None or self._offset + padded > self._cur.numel():
            self._grow(padded)
        assert self._cur is not None
        region = self._cur[self._offset : self._offset + nbytes]
        self._offset += padded
        self.carved_bytes += padded
        return region.view(dtype).view(shape)

    def pin_copy(self, src: torch.Tensor) -> torch.Tensor:
        """Pinned copy of a CPU, contiguous, not-yet-pinned tensor."""
        out = self.allocate(tuple(src.shape), src.dtype)
        out.copy_(src)
        return out

    @contextlib.contextmanager
    def pinned_pin_window(self) -> Iterator[None]:
        """Pool this thread's qualifying ``Tensor.pin_memory`` calls.

        Other threads and non-qualifying tensors always run the original
        implementation. Windows may nest; the previous window is restored
        on exit.
        """
        _install_guarded_wrapper()
        prev = getattr(_state, "pool", None)
        _state.pool = self
        try:
            yield
        finally:
            _state.pool = prev
