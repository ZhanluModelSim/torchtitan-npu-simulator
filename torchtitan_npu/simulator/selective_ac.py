# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator-scoped selective activation-checkpoint policy overrides."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Literal

import torch
import torchtitan.distributed.activation_checkpoint as activation_checkpoint


SelectiveACSaveOp = Literal[
    "full",
    "none",
    "default",
    "compute-intensive",
    "attention",
    "linear",
    "mm",
    "gmm",
    "quant-mm",
    "comm",
    "max",
]

_FULL_CHOICES = frozenset(
    {
        "default",
        "compute-intensive",
        "attention",
        "linear",
        "mm",
        "gmm",
        "quant-mm",
        "comm",
        "max",
    }
)
_OP_PATHS: dict[str, tuple[str, ...]] = {
    "attention": (
        "aten._scaled_dot_product_cudnn_attention.default",
        "aten._scaled_dot_product_attention_math.default",
        "aten._scaled_dot_product_fused_attention_overrideable.default",
    ),
    # These are intentionally delegated to the upstream policy. Its every
    # second mm/linear recompute rule remains in effect.
    "linear": ("aten.linear.default",),
    "mm": ("aten.mm.default",),
    "gmm": (
        "aten._grouped_mm.default",
        "npu.npu_grouped_matmul.default",
    ),
    "quant-mm": ("npu.npu_quant_matmul.default",),
    "comm": (
        "_c10d_functional.reduce_scatter_tensor.default",
        "_c10d_functional.all_to_all_single.default",
    ),
    "max": ("aten.max.default",),
}
_PATCH_LOCK = threading.RLock()
_explicit_selection_active: ContextVar[bool] = ContextVar(
    "simulator_explicit_selective_ac_save_ops", default=False
)


def has_explicit_selective_ac_save_ops() -> bool:
    """Whether model-specific SAC extensions must defer to a CLI selection."""
    return _explicit_selection_active.get()


def _resolve_ops(paths: tuple[str, ...]) -> set[Any]:
    resolved: set[Any] = set()
    for path in paths:
        op: Any = torch.ops
        try:
            for part in path.split("."):
                op = getattr(op, part)
        except AttributeError:
            continue
        resolved.add(op)
    return resolved


def _compute_intensive_ops() -> set[Any]:
    from torch._functorch.partitioners import get_default_op_list

    return {
        op.default
        for op in get_default_op_list().compute_intensive_ops
    }


def _validate_selection(save_ops: list[SelectiveACSaveOp]) -> None:
    selected = set(save_ops)
    if "none" in selected and len(selected) != 1:
        raise ValueError(
            "simulation.selective_ac_save_ops 'none' must be used alone"
        )


def _normalized_selection(save_ops: list[SelectiveACSaveOp]) -> set[SelectiveACSaveOp]:
    selected = set(save_ops)
    if "full" in selected:
        selected.remove("full")
        selected.update(_FULL_CHOICES)
    return selected


def _selected_save_ops(
    save_ops: list[SelectiveACSaveOp],
    upstream_save_ops: set[Any],
) -> set[Any]:
    selected = _normalized_selection(save_ops)
    if "none" in selected:
        return set()

    resolved = set(upstream_save_ops) if "default" in selected else set()
    if "compute-intensive" in selected:
        resolved.update(_compute_intensive_ops())
    for choice, paths in _OP_PATHS.items():
        if choice in selected:
            resolved.update(_resolve_ops(paths))
    return resolved


@contextmanager
def selective_ac_save_ops_context(
    save_ops: list[SelectiveACSaveOp] | None,
) -> Iterator[None]:
    """Apply an explicit SAC save-op selection while models are wrapped.

    ``mm`` and ``linear`` stay in TorchTitan's original policy, including its
    alternating recompute rule. The other selected op groups are ordinary
    save-ops and therefore resolve to ``MUST_SAVE`` in that policy.
    """
    if save_ops is None:
        yield
        return

    _validate_selection(save_ops)
    token = _explicit_selection_active.set(True)
    original_get_save_ops = getattr(activation_checkpoint, "_get_save_ops", None)
    if original_get_save_ops is None:
        _explicit_selection_active.reset(token)
        raise RuntimeError(
            "TorchTitan selective AC no longer exposes _get_save_ops; "
            "the simulator selective-AC integration must be updated."
        )

    with _PATCH_LOCK:
        def _get_selected_save_ops() -> set[Any]:
            return _selected_save_ops(save_ops, set(original_get_save_ops()))

        activation_checkpoint._get_save_ops = _get_selected_save_ops
        try:
            yield
        finally:
            activation_checkpoint._get_save_ops = original_get_save_ops
            _explicit_selection_active.reset(token)
