# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulator-scoped selective activation-checkpoint policy overrides."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torchtitan.distributed.activation_checkpoint import SelectiveAC

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
    "all-to-all",
    "max",
]

# Extra synthetic cache extensions are separate from native collective SAC.
_FULL_CHOICES: frozenset[SelectiveACSaveOp] = frozenset(
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
        "cann_ops_transformer.sparse_flash_mla.default",
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

    return {getattr(op, "default", op) for op in get_default_op_list().compute_intensive_ops}


def _validate_selection(save_ops: list[SelectiveACSaveOp]) -> None:
    selected = set(save_ops)
    unknown = selected - {"full", "none", "default", "compute-intensive", *_OP_PATHS, "all-to-all"}
    if unknown:
        raise ValueError(f"Unknown selective AC save groups: {sorted(unknown)}")
    if "all-to-all" in selected:
        raise ValueError("Synthetic all-to-all SAC extension is deferred; use native comm policy")
    if "none" in selected and len(selected) != 1:
        raise ValueError("simulation.selective_ac_save_ops 'none' must be used alone")


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
            operators = _resolve_ops(paths)
            if not operators:
                raise ValueError(f"Selected SAC group {choice!r} has no registered operators in this environment")
            resolved.update(operators)
    return resolved


class SimulatorSelectiveAC(SelectiveAC):
    """Use native SAC wrapping and alternating-mm behavior with explicit groups."""

    @dataclass(kw_only=True, slots=True)
    class Config(SelectiveAC.Config):
        save_ops: list[SelectiveACSaveOp] | None = None

    def get_save_ops(self) -> set:
        selected = cast("SimulatorSelectiveAC.Config", self.config).save_ops
        upstream = super().get_save_ops()
        if selected is None:
            return upstream
        _validate_selection(selected)
        return _selected_save_ops(selected, upstream)
