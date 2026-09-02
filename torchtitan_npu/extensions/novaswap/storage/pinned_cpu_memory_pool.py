# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

"""Pinned CPU slab pool for asynchronous swap operations."""

from __future__ import annotations

import atexit
import bisect
import logging
import threading
from dataclasses import dataclass

import torch
from torchtitan.tools.logging import logger

_SLAB_GROW_ALIGNMENT_BYTES = 1024**3

RangeKey = tuple[int, int]


@dataclass(frozen=True)
class _CpuMemoryPoolConfig:
    """Runtime configuration for process-wide pinned CPU storage."""

    enabled: bool
    max_cached_bytes: int
    slab_default_bytes: int
    alignment_bytes: int
    log_stats: bool
    log_all_ranks: bool

    def __post_init__(self) -> None:
        if self.slab_default_bytes <= 0:
            raise ValueError("slab_default_bytes must be positive")
        if self.alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be positive")

    @classmethod
    def defaults(cls) -> _CpuMemoryPoolConfig:
        """Return explicit backend defaults without a global argument singleton."""
        return cls(
            enabled=True,
            max_cached_bytes=0,
            slab_default_bytes=_SLAB_GROW_ALIGNMENT_BYTES,
            alignment_bytes=512,
            log_stats=False,
            log_all_ranks=False,
        )


class PinnedCpuStorage:
    """Process-wide pinned CPU storage used by the swap engine."""

    _lock = threading.RLock()
    _pool: CpuMemoryPool | None = None
    _config: _CpuMemoryPoolConfig | None = None
    _use_pool: bool | None = None
    _owners: dict[int, str] = {}
    _exit_hook_registered = False

    @classmethod
    def allocate(cls, size_bytes: int, *, owner: str | None = None) -> torch.Tensor:
        """Allocate a pinned uint8 buffer with exactly ``size_bytes`` elements."""
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")

        with cls._lock:
            if not cls._pool_enabled_unlocked():
                tensor = torch.empty(
                    size_bytes,
                    dtype=torch.uint8,
                    pin_memory=True,
                    device="cpu",
                )
            else:
                tensor = cls._get_pool_unlocked().allocate(size_bytes)
            if owner is not None:
                cls._owners[id(tensor)] = owner
            return tensor

    @classmethod
    def free(cls, tensor: torch.Tensor, *, owner: str | None = None) -> None:
        with cls._lock:
            recorded_owner = cls._owners.pop(id(tensor), None)
            if recorded_owner is not None and owner != recorded_owner:
                raise AssertionError("pinned CPU allocation released by a different tensor owner")
            if cls._pool_enabled_unlocked():
                cls._get_pool_unlocked().free(tensor)
            else:
                tensor.untyped_storage().resize_(0)

    @classmethod
    def shutdown(cls) -> None:
        with cls._lock:
            if cls._pool is not None:
                cls._log_stats_unlocked(context="shutdown")
                cls._pool.clear()
                cls._pool = None
            cls._config = None
            cls._use_pool = None
            cls._owners.clear()

    @classmethod
    def _pool_enabled_unlocked(cls) -> bool:
        if cls._use_pool is None:
            cls._use_pool = cls._get_config_unlocked().enabled
        return cls._use_pool

    @classmethod
    def _get_config_unlocked(cls) -> _CpuMemoryPoolConfig:
        if cls._config is None:
            # TORCHTITAN-NPU MOD: TorchTitan has no get_args() singleton.
            # Remove when: pool settings become a public plugin config surface.
            cls._config = _CpuMemoryPoolConfig.defaults()
        return cls._config

    @classmethod
    def _get_pool_unlocked(cls) -> CpuMemoryPool:
        if cls._pool is None:
            config = cls._get_config_unlocked()
            slab_default_bytes = config.slab_default_bytes
            if config.max_cached_bytes > 0:
                slab_default_bytes = min(slab_default_bytes, config.max_cached_bytes)
            cls._pool = CpuMemoryPool(
                max_cached_bytes=config.max_cached_bytes,
                slab_default_bytes=slab_default_bytes,
                alignment_bytes=config.alignment_bytes,
            )
            cls._register_exit_hook_unlocked()
        return cls._pool

    @classmethod
    def _log_stats_unlocked(cls, context: str) -> None:
        config = cls._config
        if cls._pool is None or config is None or not config.log_stats:
            return

        fields = " ".join(
            f"{name}={value:.6f}" if isinstance(value, float) else f"{name}={value}"
            for name, value in cls._pool.stats().items()
        )
        message = f"[PinnedMemoryPool] context={context} {fields}"
        logger.log(logging.INFO, message)

    @classmethod
    def _log_at_exit(cls) -> None:
        try:
            with cls._lock:
                cls._log_stats_unlocked(context="process_exit")
        except Exception:
            logger.exception("[PinnedMemoryPool] failed to log exit stats")

    @classmethod
    def _register_exit_hook_unlocked(cls) -> None:
        if not cls._exit_hook_registered:
            atexit.register(cls._log_at_exit)
            cls._exit_hook_registered = True


@dataclass
class _Slab:
    slab_id: int
    base: torch.Tensor
    size_bytes: int
    used_bytes: int = 0
    allocation_count: int = 0

    @property
    def empty(self) -> bool:
        return self.used_bytes == 0 and self.allocation_count == 0


@dataclass(frozen=True, order=True)
class _FreeRange:
    # Size is first so the natural order is usable for best-fit.
    size_bytes: int
    slab_id: int
    offset_bytes: int

    @property
    def key(self) -> RangeKey:
        return self.slab_id, self.offset_bytes

    @property
    def end(self) -> int:
        return self.offset_bytes + self.size_bytes


@dataclass(frozen=True)
class _Allocation:
    slab_id: int
    offset_bytes: int
    requested_bytes: int
    reserved_bytes: int


@dataclass
class _Counters:
    alloc_requests: int = 0
    reuse_hits: int = 0
    new_slabs: int = 0
    oom_retries: int = 0
    frees: int = 0
    evictions: int = 0
    active_bytes: int = 0
    active_bytes_peak: int = 0


def _align_up(value: int, alignment: int) -> int:
    return 0 if value <= 0 else ((value + alignment - 1) // alignment) * alignment


class CpuMemoryPool:
    """Thread-safe best-fit allocator for reusable pinned CPU byte slabs."""

    def __init__(
        self,
        max_cached_bytes: int,
        slab_default_bytes: int,
        alignment_bytes: int,
        device: str = "cpu",
        pin_memory: bool = True,
    ) -> None:
        if slab_default_bytes <= 0:
            raise ValueError("slab_default_bytes must be positive")
        if alignment_bytes <= 0:
            raise ValueError("alignment_bytes must be positive")

        self.max_cached_bytes = max_cached_bytes
        self.slab_default_bytes = slab_default_bytes
        self.alignment_bytes = alignment_bytes
        self.device = torch.device(device)  # pyrefly: ignore[read-only]
        self.pin_memory = pin_memory

        self._reset_slab_state()
        self._allocations: dict[int, _Allocation] = {}
        self._counters = _Counters()
        self._lock = threading.RLock()

    def allocate(self, size_bytes: int) -> torch.Tensor:
        """Allocate a uint8 buffer from a slab using a byte-count request."""
        if size_bytes < 0:
            raise ValueError(f"size_bytes must be non-negative, got {size_bytes}")

        requested_bytes = size_bytes
        reserved_bytes = _align_up(requested_bytes, self.alignment_bytes)

        with self._lock:
            self._counters.alloc_requests += 1

            if requested_bytes == 0:
                tensor = torch.empty(
                    size_bytes,
                    dtype=torch.uint8,
                    device=self.device,
                    pin_memory=self.pin_memory,
                )
                allocation = _Allocation(-1, 0, 0, 0)
                self._track_allocation(tensor, allocation, reused=False)
                return tensor

            free_range = self._best_fit(reserved_bytes, self.alignment_bytes)
            reused = free_range is not None

            if free_range is None:
                slab_bytes = self._next_slab_bytes(reserved_bytes)
                self._grow_with_retry(slab_bytes, reserved_bytes)
                free_range = self._best_fit(reserved_bytes, self.alignment_bytes)
                if free_range is None:
                    raise RuntimeError("Slab grow succeeded but no fitting range was found")

            tensor, allocation = self._allocate_from_free_range(
                free_range,
                requested_bytes,
                reserved_bytes,
                self.alignment_bytes,
            )
            self._track_allocation(tensor, allocation, reused)
            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        with self._lock:
            allocation = self._allocations.pop(id(tensor), None)
            if allocation is None:
                raise ValueError("Cannot free a tensor that was not allocated by this pool")

            self._counters.frees += 1
            self._counters.active_bytes -= allocation.requested_bytes
            if allocation.reserved_bytes == 0:
                return

            self._release(allocation)
            self._trim_cache_unlocked()

    def clear(self) -> None:
        with self._lock:
            if self._allocations:
                raise RuntimeError(f"Cannot clear pinned pool with {len(self._allocations)} active allocation(s)")
            self._reset_slab_state()
            self._counters = _Counters()

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total_slab_bytes = self._total_slab_bytes_unlocked()
            reserved_bytes = sum(slab.used_bytes for slab in self._slabs.values())
            free_bytes = total_slab_bytes - reserved_bytes
            largest_free = self._free_sizes[-1] if self._free_sizes else 0
            fragmented_free = max(free_bytes - largest_free, 0)
            requests = self._counters.alloc_requests

            return {
                "max_cached_bytes": self.max_cached_bytes,
                "alloc_requests": requests,
                "reuse_hits": self._counters.reuse_hits,
                "reuse_hit_rate": self._counters.reuse_hits / requests if requests else 0.0,
                "new_slabs": self._counters.new_slabs,
                "oom_retries": self._counters.oom_retries,
                "frees": self._counters.frees,
                "evictions": self._counters.evictions,
                "active_bytes": self._counters.active_bytes,
                "active_bytes_peak": self._counters.active_bytes_peak,
                "reserved_bytes": reserved_bytes,
                "total_slab_bytes": total_slab_bytes,
                "free_bytes": free_bytes,
                "largest_free_range_bytes": largest_free,
                "fragmentation": fragmented_free / free_bytes if free_bytes else 0.0,
            }

    # Allocation ownership and statistics.
    def _track_allocation(
        self,
        tensor: torch.Tensor,
        allocation: _Allocation,
        reused: bool,
    ) -> None:
        tensor_id = id(tensor)
        self._allocations[tensor_id] = allocation
        if reused:
            self._counters.reuse_hits += 1

        self._counters.active_bytes += allocation.requested_bytes
        self._counters.active_bytes_peak = max(
            self._counters.active_bytes_peak,
            self._counters.active_bytes,
        )

    # Byte slab and free-range indexes.
    def _reset_slab_state(self) -> None:
        self._slabs: dict[int, _Slab] = {}
        self._free_sizes: list[int] = []
        self._free_keys_by_size: dict[int, set[RangeKey]] = {}
        self._free_by_key: dict[RangeKey, _FreeRange] = {}
        self._free_offsets_by_slab: dict[int, list[int]] = {}
        self._next_slab_id = 0

    def _allocate_slab(self, size_bytes: int) -> torch.Tensor:
        return torch.empty(
            size_bytes,
            dtype=torch.uint8,
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _add_slab(self, base: torch.Tensor) -> None:
        slab_id = self._next_slab_id
        actual_bytes = base.numel()
        self._slabs[slab_id] = _Slab(slab_id, base, actual_bytes)
        try:
            self._add_free(_FreeRange(actual_bytes, slab_id, 0))
        except Exception:
            self._slabs.pop(slab_id)
            raise
        self._next_slab_id += 1

    def _best_fit(self, size_bytes: int, alignment_bytes: int) -> _FreeRange | None:
        index = bisect.bisect_left(self._free_sizes, size_bytes)
        while index < len(self._free_sizes):
            size = self._free_sizes[index]
            for key in self._free_keys_by_size[size]:
                free_range = self._free_by_key[key]
                offset_bytes = _align_up(free_range.offset_bytes, alignment_bytes)
                if offset_bytes + size_bytes <= free_range.end:
                    return free_range
            index += 1
        return None

    def _allocate_from_free_range(
        self,
        free_range: _FreeRange,
        requested_bytes: int,
        reserved_bytes: int,
        alignment_bytes: int,
    ) -> tuple[torch.Tensor, _Allocation]:
        slab = self._slabs[free_range.slab_id]
        offset_bytes = _align_up(free_range.offset_bytes, alignment_bytes)
        prefix_bytes = offset_bytes - free_range.offset_bytes
        tensor = slab.base.narrow(0, offset_bytes, requested_bytes)

        self._remove_free(free_range)
        if prefix_bytes:
            self._add_free(_FreeRange(prefix_bytes, free_range.slab_id, free_range.offset_bytes))

        remaining_bytes = free_range.end - (offset_bytes + reserved_bytes)
        if remaining_bytes:
            self._add_free(
                _FreeRange(
                    remaining_bytes,
                    free_range.slab_id,
                    offset_bytes + reserved_bytes,
                )
            )

        slab.used_bytes += reserved_bytes
        slab.allocation_count += 1
        allocation = _Allocation(
            slab_id=free_range.slab_id,
            offset_bytes=offset_bytes,
            requested_bytes=requested_bytes,
            reserved_bytes=reserved_bytes,
        )
        return tensor, allocation

    def _release(self, allocation: _Allocation) -> None:
        slab = self._slabs[allocation.slab_id]
        released = _FreeRange(
            allocation.reserved_bytes,
            allocation.slab_id,
            allocation.offset_bytes,
        )
        previous, following = self._neighbors(allocation.slab_id, allocation.offset_bytes)

        if previous is not None and previous.end == released.offset_bytes:
            self._remove_free(previous)
            released = _FreeRange(
                previous.size_bytes + released.size_bytes,
                released.slab_id,
                previous.offset_bytes,
            )

        if following is not None and released.end == following.offset_bytes:
            self._remove_free(following)
            released = _FreeRange(
                released.size_bytes + following.size_bytes,
                released.slab_id,
                released.offset_bytes,
            )

        self._add_free(released)
        slab.used_bytes -= allocation.reserved_bytes
        slab.allocation_count -= 1

    def _evict_empty_slab(self) -> int:
        victim = next((slab for slab in self._slabs.values() if slab.empty), None)
        if victim is None:
            return 0

        whole_slab = self._free_by_key[(victim.slab_id, 0)]
        self._remove_free(whole_slab)
        self._slabs.pop(victim.slab_id)
        return victim.size_bytes

    def _neighbors(
        self,
        slab_id: int,
        offset_bytes: int,
    ) -> tuple[_FreeRange | None, _FreeRange | None]:
        offsets = self._free_offsets_by_slab.get(slab_id)
        if not offsets:
            return None, None

        index = bisect.bisect_left(offsets, offset_bytes)
        previous = self._free_by_key[(slab_id, offsets[index - 1])] if index > 0 else None
        following = self._free_by_key[(slab_id, offsets[index])] if index < len(offsets) else None
        return previous, following

    def _add_free(self, free_range: _FreeRange) -> None:
        self._free_by_key[free_range.key] = free_range
        keys = self._free_keys_by_size.get(free_range.size_bytes)
        if keys is None:
            keys = set()
            self._free_keys_by_size[free_range.size_bytes] = keys
            bisect.insort(self._free_sizes, free_range.size_bytes)
        keys.add(free_range.key)

        offsets = self._free_offsets_by_slab.setdefault(free_range.slab_id, [])
        bisect.insort(offsets, free_range.offset_bytes)

    def _remove_free(self, free_range: _FreeRange) -> None:
        keys = self._free_keys_by_size[free_range.size_bytes]
        offsets = self._free_offsets_by_slab[free_range.slab_id]
        size_index = bisect.bisect_left(self._free_sizes, free_range.size_bytes)
        offset_index = bisect.bisect_left(offsets, free_range.offset_bytes)

        self._free_by_key.pop(free_range.key)
        keys.remove(free_range.key)
        if not keys:
            self._free_keys_by_size.pop(free_range.size_bytes)
            self._free_sizes.pop(size_index)

        offsets.pop(offset_index)
        if not offsets:
            self._free_offsets_by_slab.pop(free_range.slab_id)

    # Slab growth and cache trimming.
    def _next_slab_bytes(self, reserved_bytes: int) -> int:
        if reserved_bytes <= self.slab_default_bytes:
            return self.slab_default_bytes
        return _align_up(reserved_bytes, _SLAB_GROW_ALIGNMENT_BYTES)

    def _grow_with_retry(
        self,
        slab_bytes: int,
        required_bytes: int,
    ) -> None:
        try:
            slab = self._allocate_slab(slab_bytes)
        except RuntimeError:
            self._counters.oom_retries += 1
            self._evict_empty_slabs_unlocked(target_bytes=0)

            try:
                slab = self._allocate_slab(slab_bytes)
            except RuntimeError as error:
                raise RuntimeError(
                    "Pinned pool failed to allocate slab: "
                    f"requested={required_bytes}, slab={slab_bytes}, "
                    f"max_cached={self.max_cached_bytes}"
                ) from error

        self._add_slab(slab)
        self._counters.new_slabs += 1

    def _trim_cache_unlocked(self) -> None:
        if self.max_cached_bytes >= 0:
            self._evict_empty_slabs_unlocked(self.max_cached_bytes)

    def _evict_empty_slabs_unlocked(self, target_bytes: int) -> None:
        while self._total_slab_bytes_unlocked() > target_bytes:
            if not self._evict_empty_slab():
                return
            self._counters.evictions += 1

    def _total_slab_bytes_unlocked(self) -> int:
        return sum(slab.size_bytes for slab in self._slabs.values())
