# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Activation-checkpoint policy extensions shared by MoE models."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch
import torch.nn as nn
import torchtitan.distributed.activation_checkpoint as activation_checkpoint


_EXPENSIVE_MATMUL_OP_PATHS = (
    "aten._grouped_mm.default",
    "npu.npu_quant_matmul.default",
    "npu.npu_grouped_matmul.default",
)
_SAVE_OPS_PATCH_LOCK = threading.Lock()


def resolve_expensive_matmul_ops() -> set[Any]:
    """Resolve optional dispatcher ops after their backend libraries load."""
    save_ops: set[Any] = set()
    for path in _EXPENSIVE_MATMUL_OP_PATHS:
        obj: Any = torch.ops
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        save_ops.add(obj)
    return save_ops


@contextmanager
def extend_upstream_save_ops(extra_save_ops: set[Any]) -> Iterator[None]:
    """Temporarily extend TorchTitan's private selective-AC save-op factory."""
    get_save_ops = getattr(activation_checkpoint, "_get_save_ops", None)
    if get_save_ops is None:
        raise RuntimeError(
            "TorchTitan selective AC no longer exposes _get_save_ops; "
            "the MoE selective-AC integration must be updated."
        )

    def _get_extended_save_ops() -> set[Any]:
        return set(get_save_ops()) | extra_save_ops

    with _SAVE_OPS_PATCH_LOCK:
        activation_checkpoint._get_save_ops = _get_extended_save_ops
        try:
            yield
        finally:
            activation_checkpoint._get_save_ops = get_save_ops


def apply_moe_ac(
    model: nn.Module,
    ac_config: Any,
    *,
    model_compile_enabled: bool,
    base_folder: str,
) -> None:
    """Apply AC while caching expensive grouped/quantized matmuls under SAC."""
    if ac_config.mode != "selective":
        activation_checkpoint.apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=base_folder,
        )
        return

    with extend_upstream_save_ops(resolve_expensive_matmul_ops()):
        activation_checkpoint.apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=base_folder,
        )
