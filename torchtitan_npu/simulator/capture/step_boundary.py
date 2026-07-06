# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Detects forward/backward/optimizer step boundaries (see
spec/L1-StepGraph.md: "框架通过 autograd.backward hook + Optimizer.step
wrapper 自动识别边界") and buckets already-captured OpNodes into per-phase
StepGraphs."""

from __future__ import annotations

import uuid
from typing import Callable

import torch

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph

_PHASES = ("forward", "backward", "optimizer")


def _collect_optimizer_classes() -> list[type]:
    """Every currently-imported subclass of torch.optim.Optimizer (AdamW,
    Muon, swap/virtual optimizer wrappers, etc.)."""
    result: list[type] = []

    def _recurse(base: type) -> None:
        for sub in base.__subclasses__():
            result.append(sub)
            _recurse(sub)

    _recurse(torch.optim.Optimizer)
    return result


class StepBoundaryTracker:
    """Context manager that monkeypatches `torch.Tensor.backward` and every
    currently-loaded `Optimizer.step` to flip `self.current_phase`, plus
    exposes `.mark()` for callers that want to set the phase explicitly
    (e.g. before/after a pipeline-parallel schedule step)."""

    def __init__(self) -> None:
        self.current_phase = "forward"
        self._original_backward: Callable | None = None
        self._original_optimizer_steps: dict[type, Callable] = {}

    def __enter__(self) -> "StepBoundaryTracker":
        self.current_phase = "forward"
        self._original_backward = torch.Tensor.backward
        tracker = self

        def hooked_backward(self_tensor, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            tracker.current_phase = "backward"
            return tracker._original_backward(self_tensor, *args, **kwargs)  # type: ignore[misc]

        torch.Tensor.backward = hooked_backward  # type: ignore[method-assign]

        for optimizer_cls in _collect_optimizer_classes():
            if "step" not in optimizer_cls.__dict__:
                continue
            original_step = optimizer_cls.step
            self._original_optimizer_steps[optimizer_cls] = original_step

            def make_hooked_step(orig: Callable) -> Callable:
                def hooked_step(self_opt, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                    tracker.current_phase = "optimizer"
                    return orig(self_opt, *args, **kwargs)

                return hooked_step

            optimizer_cls.step = make_hooked_step(original_step)  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._original_backward is not None:
            torch.Tensor.backward = self._original_backward  # type: ignore[method-assign]
        for cls, original_step in self._original_optimizer_steps.items():
            cls.step = original_step  # type: ignore[method-assign]
        self._original_optimizer_steps.clear()

    def mark(self, phase: str) -> None:
        """Explicitly set the current phase."""
        self.current_phase = phase


def build_step_graphs(nodes: dict[int, OpNode]) -> dict[str, StepGraph]:
    """Bucket OpNodes into forward/backward/optimizer StepGraphs using each
    node's `annotations["phase"]` (defaults to `"forward"` if the tag is
    missing, e.g. a node captured without a `phase_provider`)."""
    buckets: dict[str, dict[str, OpNode]] = {phase: {} for phase in _PHASES}
    for op_id, node in nodes.items():
        phase = node.annotations.get("phase", "forward")
        buckets.setdefault(phase, {})[op_id] = node

    graphs: dict[str, StepGraph] = {}
    for phase, phase_nodes in buckets.items():
        if not phase_nodes:
            continue
        graphs[phase] = StepGraph(step_id=uuid.uuid4().hex[:12], step_type=phase, nodes=phase_nodes)
    return graphs
