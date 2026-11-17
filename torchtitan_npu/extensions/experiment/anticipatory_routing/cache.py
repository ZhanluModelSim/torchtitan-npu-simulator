# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import enum
from typing import Literal

import torch

# One microbatch's worth of routing indices, keyed by router FQN.
RoutingSlot = dict[str, torch.Tensor]

IndexStoreDtype = Literal["auto", "int16", "int32", "int64"]

_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


class RoutingMode(enum.Enum):
    """What the routers do with their top-k index decision this forward."""

    OFF = "off"
    """Behave exactly like the unmodified router."""

    CAPTURE = "capture"
    """Compute indices normally and store them in the current slot."""

    REPLAY = "replay"
    """Ignore the scores and use the indices already in the current slot."""


class RoutingOperation(enum.Enum):
    """Router execution role, independent of autograd and compilation state."""

    OFF = "off"
    INITIALIZE = "initialize"
    CAPTURE = "capture"
    TRAIN = "train"


class RoutingReplayBuffer:
    """Router-local tensors shared by eager and compiled execution.

    These are runtime tensors, not registered model buffers or checkpoint state.
    An isolated eager forward discovers the actual local routing shape.
    Training reads ``indices`` populated outside the graph; recording writes
    separate ``capture_indices`` snapshotted into queue entries outside the graph.
    """

    def __init__(self, enabled: torch.Tensor):
        self.enabled = enabled
        self.operation = RoutingOperation.OFF
        self.indices: torch.Tensor | None = None
        self.capture_indices: torch.Tensor | None = None

    def select_experts(self, live_ids: torch.Tensor) -> torch.Tensor:
        """Initialize, record or select IDs according to the explicit operation."""
        if self.operation is RoutingOperation.OFF:
            return live_ids
        if self.operation is RoutingOperation.INITIALIZE:
            self.initialize(live_ids)
            return live_ids

        if self.indices is None or self.capture_indices is None:
            raise RuntimeError("Routing buffers must be initialized before capture or training")
        if self.indices.shape != live_ids.shape:
            raise ValueError(f"Cached route shape {self.indices.shape} does not match live IDs {live_ids.shape}")

        if self.operation is RoutingOperation.CAPTURE:
            self.capture_indices.copy_(live_ids)
            return live_ids
        # TRAIN stays constant across NORMAL and REPLAY. Only tensor contents
        # change, so these two training modes can reuse the same compiled graph.
        return torch.where(self.enabled, self.indices, live_ids)

    def initialize(self, indices: torch.Tensor) -> None:
        if self.indices is None:
            self.indices = indices.detach().clone()
            self.capture_indices = indices.detach().clone()

    def load(self, indices: torch.Tensor) -> None:
        if self.indices is None:
            raise RuntimeError("Replay buffer was not initialized by a forward pass")
        if indices.shape != self.indices.shape:
            raise ValueError(f"Cached route shape {indices.shape} does not match replay buffer {self.indices.shape}")
        self.indices.copy_(indices)


def resolve_store_dtype(name: IndexStoreDtype, num_experts: int) -> torch.dtype:
    """Pick the narrowest integer dtype that can hold an expert id.

    A slot holds one ``(T, K)`` index tensor per MoE layer, and a delay of
    ``dt`` steps keeps ``dt`` slots alive at once, so the dtype is worth
    narrowing: int16 is a quarter of the int64 the top-k returns.
    """
    if name != "auto":
        dtype = _DTYPE_BY_NAME[name]
        if num_experts - 1 > torch.iinfo(dtype).max:
            raise ValueError(f"index_store_dtype={name} cannot represent expert ids up to {num_experts - 1}.")
        return dtype
    if num_experts - 1 <= torch.iinfo(torch.int16).max:
        return torch.int16
    if num_experts - 1 <= torch.iinfo(torch.int32).max:
        return torch.int32
    if num_experts - 1 <= torch.iinfo(torch.int64).max:
        return torch.int64
    raise ValueError("Expert ids exceed int64 storage capacity")


class RoutingIndexCache:
    """Store the anticipatory routers read from and write to.

    One instance is shared by every router in the model. Both eager and compiled
    routers use local replay tensors populated by :meth:`select`. Dictionary
    access and execution-role changes remain outside the model forward.
    Initialization, capture and training do not overlap; callers always reset
    in a finally block before starting the next operation.
    """

    def __init__(
        self,
        *,
        store_dtype: torch.dtype,
    ) -> None:
        self.mode = RoutingMode.OFF
        self.slot: RoutingSlot | None = None
        self.step_slots: list[RoutingSlot] | None = None
        self.store_dtype = store_dtype
        self.replay_buffers: dict[str, RoutingReplayBuffer] = {}
        self.replay_enabled: torch.Tensor | None = None

    def create_replay_buffers(self, keys: list[str], device: torch.device) -> None:
        self.replay_enabled = torch.zeros((), dtype=torch.bool, device=device)
        self.replay_buffers = {key: RoutingReplayBuffer(self.replay_enabled) for key in keys}

    def _set_operation(self, operation: RoutingOperation) -> None:
        for buffer in self.replay_buffers.values():
            buffer.operation = operation

    def begin_initialization(self) -> None:
        """Discover local ID shapes during an explicitly eager isolated forward."""
        self._set_operation(RoutingOperation.INITIALIZE)

    @property
    def needs_materialization(self) -> bool:
        """Whether routing still needs to discover local tensor shapes."""
        return any(buffer.indices is None for buffer in self.replay_buffers.values())

    def begin_capture(self, slot: RoutingSlot) -> None:
        """Start recording one microbatch; the caller resets after the forward."""
        self.mode = RoutingMode.CAPTURE
        self.slot = slot
        self._set_operation(RoutingOperation.CAPTURE)

    def save_captured_routes(self) -> None:
        """Snapshot a successful forward into the current slot, outside the graph."""
        for key, buffer in self.replay_buffers.items():
            if buffer.capture_indices is None:
                raise RuntimeError("Capture buffer was not initialized by a forward pass")
            self.capture(key, buffer.capture_indices)

    def begin_training(self, step_slots: list[RoutingSlot] | None) -> None:
        """Select live or cached routing for one complete optimizer step.

        ``None`` runs the step with the cache off, which is what ordinary
        training does. ACTIVE and DRAIN both supply cached slots.
        """
        self.mode = RoutingMode.REPLAY if step_slots is not None else RoutingMode.OFF
        self.step_slots = step_slots
        if self.replay_enabled is not None:
            self.replay_enabled.fill_(step_slots is not None)
        self._set_operation(RoutingOperation.TRAIN)

    def reset(self) -> None:
        """End initialization, capture or training, keeping allocated ID tensors."""
        self.mode = RoutingMode.OFF
        self.slot = self.step_slots = None
        self._set_operation(RoutingOperation.OFF)
        if self.replay_enabled is not None:
            self.replay_enabled.zero_()

    def select(self, accumulation_index: int) -> None:
        """Point ``slot`` at the microbatch about to run. No-op when off."""
        if self.step_slots is None:
            return
        slot = self.step_slots[accumulation_index]
        self.slot = slot
        # FQN lookup, shape validation and dtype conversion happen outside the
        # graph. Keep these tensors unchanged until this microbatch's backward
        # (including activation-checkpoint recomputation) has completed.
        for key, buffer in self.replay_buffers.items():
            buffer.load(self.replay(key))

    def capture(self, key: str, topk_expert_ids_TK: torch.Tensor) -> None:
        """Store one router's indices for the microbatch being executed."""
        if self.slot is None:
            raise RuntimeError(
                "RoutingIndexCache is in CAPTURE mode with no slot set. The "
                "trainer must assign a slot before each microbatch forward."
            )
        # copy=True even when the dtype already matches: without it ``to``
        # returns the caller's tensor, which is still attached to the forward
        # that produced it.
        self.slot[key] = topk_expert_ids_TK.detach().to(dtype=self.store_dtype, copy=True)

    def replay(self, key: str) -> torch.Tensor:
        """Return the indices cached for this router in the current slot."""
        if self.slot is None:
            raise RuntimeError(
                "RoutingIndexCache is in REPLAY mode with no slot set. The "
                "trainer must assign a slot before each microbatch forward."
            )
        cached = self.slot.get(key)
        if cached is None:
            # Never fall back to freshly computed indices: that would silently
            # train some layers on current-parameter routing and some on stale
            # routing, which is not the technique.
            raise KeyError(
                f"No cached routing indices for router '{key}'. The capture "
                "pass must visit every router the training forward visits."
            )
        return cached


def slot_nbytes(slot: RoutingSlot) -> int:
    """Bytes held by one microbatch's cached indices."""
    return sum(t.numel() * t.element_size() for t in slot.values())
