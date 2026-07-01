# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""L0 op-level capture via TorchDispatchMode. Captures every dispatched
operator (aten or NPU custom op) during a training step, building a
producer/consumer dependency graph keyed by `id(tensor)` (meta tensors have
no storage to alias-track, matching spec/L0-OpNode.md's "Meta tensor
环境下关闭存储级追踪，退化到纯 id(tensor) 级" rule)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchtitan_npu.simulator.capture.module_path import ModulePathTracker
from torchtitan_npu.simulator.capture.op_mapping import to_canonical_op_type
from torchtitan_npu.simulator.capture.tensor_utils import to_tensor_meta
from torchtitan_npu.simulator.cost.op_cost_model import OpCostModel
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta

_id_counter = itertools.count()


def _next_op_id() -> str:
    return f"op_{next(_id_counter)}"


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_flatten_tensors(item))
    return tensors


@dataclass
class _RawEvent:
    op_id: str
    raw_op_type: str
    op_type: str
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    predecessors: list[str]
    module_path: str = ""
    phase: str = "forward"
    repeat_count: int = 1


def _shape_signature(event: _RawEvent) -> tuple:
    return (
        event.raw_op_type,
        event.module_path,
        event.phase,
        tuple(tuple(i.shape) for i in event.inputs),
        tuple(tuple(o.shape) for o in event.outputs),
    )


class OpDispatchCapture(TorchDispatchMode):
    """Records one L0 op stream. Usage::

        capture = OpDispatchCapture()
        with capture:
            out = model(x)
            out.sum().backward()
        nodes = capture.build_nodes()
    """

    def __init__(
        self,
        cost_model: OpCostModel | None = None,
        module_path_tracker: ModulePathTracker | None = None,
        phase_provider: Callable[[], str] | None = None,
    ) -> None:
        super().__init__()
        self.cost_model = cost_model or OpCostModel()
        self.module_path_tracker = module_path_tracker
        self.phase_provider = phase_provider
        self._events: list[_RawEvent] = []
        self._producer: dict[int, str] = {}
        self._last_signature: tuple | None = None
        self._previous_active_capture: OpDispatchCapture | None = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN201
        kwargs = kwargs or {}
        result = func(*args, **kwargs)

        flat_inputs = _flatten_tensors(args) + _flatten_tensors(tuple(kwargs.values()))
        flat_outputs = _flatten_tensors(result if isinstance(result, (tuple, list)) else (result,))
        module_path = self.module_path_tracker.current_path() if self.module_path_tracker else ""
        self._record_event(str(func), flat_inputs, flat_outputs, module_path)

        return result

    def record_synthetic_op(
        self,
        raw_op_type: str,
        inputs: list[torch.Tensor],
        outputs: list[torch.Tensor],
        module_path: str = "",
    ) -> None:
        """Manually register one synthetic L0 event, as if `raw_op_type` had
        gone through __torch_dispatch__ normally. Used by
        torchtitan_npu.simulator.hardware_shims for ops that cannot execute
        for real (raw Triton kernels / JIT-compiled extensions) but whose
        real op name + output shape are known analytically. Participates in
        the same producer/consumer id(tensor) wiring, repeat_count dedup,
        and phase tagging as real dispatched events."""
        self._record_event(raw_op_type, inputs, outputs, module_path)

    def _record_event(
        self,
        raw_op_type: str,
        flat_inputs: list[torch.Tensor],
        flat_outputs: list[torch.Tensor],
        module_path: str,
    ) -> None:
        predecessors = sorted({self._producer[id(t)] for t in flat_inputs if id(t) in self._producer})
        input_metas = [to_tensor_meta(t, name=f"in_{i}") for i, t in enumerate(flat_inputs)]
        output_metas = [to_tensor_meta(t, name=f"out_{i}") for i, t in enumerate(flat_outputs)]

        op_type = to_canonical_op_type(raw_op_type)
        phase = self.phase_provider() if self.phase_provider else "forward"

        candidate = _RawEvent(
            op_id="",
            raw_op_type=raw_op_type,
            op_type=op_type,
            inputs=input_metas,
            outputs=output_metas,
            predecessors=predecessors,
            module_path=module_path,
            phase=phase,
        )
        signature = _shape_signature(candidate)

        if self._events and signature == self._last_signature:
            retained = self._events[-1]
            retained.repeat_count += 1
            op_id = retained.op_id
        else:
            op_id = _next_op_id()
            candidate.op_id = op_id
            self._events.append(candidate)
            self._last_signature = signature

        for t in flat_outputs:
            self._producer[id(t)] = op_id

    def __enter__(self) -> "OpDispatchCapture":
        super().__enter__()
        global _active_capture
        self._previous_active_capture = _active_capture
        _active_capture = self
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        global _active_capture
        _active_capture = self._previous_active_capture
        super().__exit__(exc_type, exc_val, exc_tb)

    def build_nodes(self) -> dict[str, OpNode]:
        """Assemble captured events into OpNode objects with cost annotations."""
        nodes: dict[str, OpNode] = {}
        for event in self._events:
            cost = self.cost_model.compute(event.op_type, event.inputs, event.outputs, {})
            annotations: dict[str, Any] = {"raw_op_type": event.raw_op_type, "phase": event.phase}
            if event.module_path:
                annotations["module_path"] = event.module_path
            if event.repeat_count > 1:
                annotations["repeat_count"] = event.repeat_count
            if cost.unknown:
                annotations["cost_unknown"] = True
            nodes[event.op_id] = OpNode(
                op_id=event.op_id,
                op_type=event.op_type,
                inputs=event.inputs,
                outputs=event.outputs,
                attrs={},
                predecessors=list(event.predecessors),
                successors=[],
                flops=cost.flops,
                peak_mem=cost.peak_mem,
                param_mem=cost.param_mem,
                comm_bytes=cost.comm_bytes,
                annotations=annotations,
            )
        for op_id, node in nodes.items():
            for pred_id in node.predecessors:
                if pred_id in nodes:
                    nodes[pred_id].successors.append(op_id)
        return nodes


_active_capture: "OpDispatchCapture | None" = None


def get_active_capture() -> "OpDispatchCapture | None":
    """Returns the `OpDispatchCapture` instance currently inside its `with`
    block (there is at most one active at a time -- one step is captured at
    a time), or `None` if no capture is active. Lets code that has no
    direct reference to the capture instance (e.g. hardware_shims'
    nn.Module replacements, which run deep inside a model's forward/backward
    with no capture parameter threaded through) reach it to call
    `record_synthetic_op`."""
    return _active_capture
