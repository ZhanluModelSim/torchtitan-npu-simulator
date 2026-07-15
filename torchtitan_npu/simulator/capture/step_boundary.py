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
    """Bucket OpNodes into per-``(pp_stage, comp_type)`` StepGraphs.

    ``comp_type`` (set by ``dispatch_capture._record_event`` from
    ``_pp_context["comp_type"]``) is the fine-grained compute-graph class:
    ``"F"`` forward, ``"B"`` full backward, ``"I"`` input-grad only,
    ``"W"`` weight-grad only, ``"F_RECOMPUTE"`` recompute forward,
    ``"OPTIMIZER"``. Bucketing by ``(stage, comp_type)`` — instead of the
    coarse ``forward``/``backward``/``optimizer`` triple — restores the
    distinct compute graphs that complex pipeline strategies (ZBV /
    DualPipe / Interleaved zero-bubble) produce: an input-grad ("I") pass
    and a weight-grad ("W") pass are two topologically different DAGs and
    must NOT be merged into one ``"backward"`` StepGraph. Different PP
    stages' forwards are likewise kept separate.

    A node missing the ``comp_type`` tag (e.g. captured without a
    ``phase_provider``) falls back to its ``phase`` annotation. The
    template id is ``f"s{stage}_{comp_type}"`` (e.g. ``s0_I``, ``s2_W``)."""
    buckets: dict[str, dict[str, OpNode]] = {}
    for op_id, node in nodes.items():
        ann = node.annotations
        comp_type = ann.get("comp_type") or ann.get("phase", "forward")
        # Map legacy phase values to comp_type for backward compatibility.
        if comp_type == "forward":
            comp_type = "F"
        elif comp_type == "backward":
            comp_type = "B"
        elif comp_type == "optimizer":
            comp_type = "OPTIMIZER"
        stage = ann.get("pp_stage", -1)
        try:
            stage = int(stage)
        except (TypeError, ValueError):
            stage = -1
        template_id = f"s{stage}_{comp_type}"
        buckets.setdefault(template_id, {})[op_id] = node

    graphs: dict[str, StepGraph] = {}
    for template_id, phase_nodes in buckets.items():
        # step_type is the comp_type (strip the "s{stage}_" prefix) so
        # viz/exporters can group by compute-graph class.
        step_type = template_id.split("_", 1)[1] if "_" in template_id else template_id
        graphs[template_id] = StepGraph(
            step_id=uuid.uuid4().hex[:12], step_type=step_type, nodes=phase_nodes
        )
    return graphs
