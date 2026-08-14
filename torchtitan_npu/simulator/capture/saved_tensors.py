# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observe the tensors autograd actually saves for backward."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture


@dataclass(frozen=True, slots=True)
class _SavedTensorHandle:
    tensor: torch.Tensor
    slot_id: int | None


class AutogradSavedTensorCapture:
    """Saved-tensor hook that preserves data while recording slot metadata."""

    def __init__(self) -> None:
        self._context: object | None = None

    def __enter__(self) -> "AutogradSavedTensorCapture":
        self._context = torch.autograd.graph.saved_tensors_hooks(
            self._pack,
            self._unpack,
        )
        self._context.__enter__()  # type: ignore[union-attr]
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> bool:
        assert self._context is not None
        return bool(self._context.__exit__(exc_type, exc_value, traceback))  # type: ignore[union-attr]

    @staticmethod
    def _pack(tensor: torch.Tensor) -> _SavedTensorHandle:
        capture = get_active_capture()
        slot_id = (
            capture.record_autograd_saved_tensor_pack(tensor)
            if capture is not None
            else None
        )
        # Hooks must not retain the original tensor. ``detach`` shares its
        # storage and is the documented identity-preserving hook pattern.
        if capture is None:
            packed = tensor.detach()
        else:
            with capture.suspend_recording():
                packed = tensor.detach()
        return _SavedTensorHandle(packed, slot_id)

    @staticmethod
    def _unpack(handle: _SavedTensorHandle) -> torch.Tensor:
        capture = get_active_capture()
        if capture is not None:
            capture.record_autograd_saved_tensor_unpack(handle.slot_id)
        return handle.tensor
