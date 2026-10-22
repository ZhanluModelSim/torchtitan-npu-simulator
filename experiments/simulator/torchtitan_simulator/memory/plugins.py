# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Extension points for static memory modeling.

The core estimator owns generic use-def liveness. Plugins add narrow,
framework-specific residency models without mixing those policies into the
main event scan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from torchtitan_simulator.memory.records import (
        AutogradSavedTensorEvent,
        CheckpointBoundaryEvent,
        CheckpointTensorRecord,
        FSDPResidencyEvent,
        RawMemoryEvent,
        TensorLifetime,
    )


@dataclass(frozen=True, slots=True)
class MissingParameterGradient:
    """A trainable local parameter that did not receive an autograd tensor."""

    name: str
    num_bytes: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class ParameterTensorMetadata:
    """Persistent local parameter storage and its logical metadata."""

    name: str
    num_bytes: int
    shape: tuple[int, ...]
    dtype: str
    storage_key: str = ""


@dataclass(slots=True)
class MemoryModelContext:
    events: list[RawMemoryEvent]
    comm_by_op: dict[int, Any]
    lifetimes_by_tensor_id: dict[int, TensorLifetime]
    param_ids: set[int]
    fsdp_residency_events: list[FSDPResidencyEvent] = field(default_factory=list)
    missing_parameter_gradients: list[MissingParameterGradient] = field(default_factory=list)
    parameter_tensors: list[ParameterTensorMetadata] = field(default_factory=list)
    checkpoint_boundary_events: list[CheckpointBoundaryEvent] = field(default_factory=list)
    checkpoint_tensors: list[CheckpointTensorRecord] = field(default_factory=list)
    activation_offload_tensors: list[CheckpointTensorRecord] = field(default_factory=list)
    autograd_saved_tensors: list[AutogradSavedTensorEvent] = field(default_factory=list)
    alias_base_by_tensor_id: dict[int, int] = field(default_factory=dict)
    storage_roots: dict[str, int] = field(default_factory=dict)
    offload_ac_saved_tensors: bool = False
    notes: list[str] = field(default_factory=list)

    def tensor_root(self, tensor_id: int, storage_key: str = "") -> int:
        """Resolve storage provenance before falling back to dispatch aliases."""
        if storage_key and storage_key in self.storage_roots:
            return self.storage_roots[storage_key]
        seen: set[int] = set()
        while tensor_id in self.alias_base_by_tensor_id:
            if tensor_id in seen:
                raise ValueError("Tensor alias provenance contains a cycle")
            seen.add(tensor_id)
            parent = self.alias_base_by_tensor_id[tensor_id]
            if parent == tensor_id:
                break
            tensor_id = parent
        return tensor_id


class MemoryModelPlugin(Protocol):
    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        """Return extra lifetimes synthesized by this model."""
