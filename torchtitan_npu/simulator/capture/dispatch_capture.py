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
import weakref
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from torchtitan_npu.simulator.capture.checkpoint_execution import current_execution_kind
from torchtitan_npu.simulator.capture.module_path import ModulePathTracker
from torchtitan_npu.simulator.capture.op_mapping import to_canonical_op_type
from torchtitan_npu.simulator.capture.tensor_utils import dtype_to_str, tensor_volume_bytes, to_tensor_meta
from torchtitan_npu.simulator.cost.op_cost_model import OpCostModel
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta
from torchtitan_npu.simulator.memory.records import RawMemoryEvent, TensorRef

_id_counter = itertools.count()
_seq_counter = itertools.count()
_memory_event_counter = itertools.count()


def _next_op_id() -> int:
    return next(_id_counter)


def _flatten_tensors(value: Any, *, localize_dtensor: bool = True) -> list[torch.Tensor]:
    """Recursively extract all tensors from nested lists/tuples/dicts.

    PyTorch operator arguments can nest tensors inside dicts (e.g. some
    custom ops pass ``{"mask": tensor}``), so we recurse into dict values
    too -- otherwise those tensors are missed, breaking producer/consumer
    edge construction and producing incomplete IR dependencies. DTensors are
    localized by default because dependency and memory tracking are per-rank;
    callers may retain them to read logical global metadata instead.
    """
    tensors: list[torch.Tensor] = []

    from torch.distributed.tensor import DTensor
    if isinstance(value, DTensor):
        if localize_dtensor:
            local_tensor = getattr(value, "_local_tensor", None)
            tensors.append(local_tensor if isinstance(local_tensor, torch.Tensor) else value.to_local())
        else:
            tensors.append(value)
    elif isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(_flatten_tensors(item, localize_dtensor=localize_dtensor))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(_flatten_tensors(item, localize_dtensor=localize_dtensor))
    return tensors


@dataclass
class _RawEvent:
    op_id: int
    raw_op_type: str
    op_type: str
    inputs: list[TensorMeta]
    outputs: list[TensorMeta]
    predecessors: list[str]
    module_path: str = ""
    phase: str = "forward"
    execution_kind: str = "original_forward"
    repeat_count: int = 1
    comm_dim: str = ""
    comm_ranks_str: str = ""
    seq_idx: int = 0
    pp_stage: int = -1
    pp_mb_idx: int = -1
    # Fine-grained compute-graph class (mirrors
    # torch.distributed.pipelining.schedules._ComputationType):
    # "F" forward, "B" full backward, "I" input-grad only, "W" weight-grad
    # only, "F_RECOMPUTE" recompute forward, "OPTIMIZER". For non-PP steps
    # this derives from `phase` (F/B/OPTIMIZER). Used by build_step_graphs
    # to bucket L0 ops into per-(stage, comp_type) StepGraphs instead of
    # the coarse forward/backward/optimizer triple.
    comp_type: str = "F"
    # FSDP sharding state active when this op ran ("SHARDED"/"UNSHARDED"/"NA").
    fsdp_state: str = "NA"
    tensor_shape_scope: str = "local"


def _shape_signature(event: _RawEvent) -> tuple:
    # `comp_type` is part of the signature so that a full-backward ("B") op
    # and an input-grad ("I") / weight-grad ("W") op with identical shapes do
    # NOT collapse into one repeat_count'd entry — they belong to different
    # compute-graph templates and must stay distinct in the L0 IR.
    return (
        event.raw_op_type,
        event.module_path,
        event.phase,
        event.execution_kind,
        event.comp_type,
        event.pp_stage,
        event.tensor_shape_scope,
        tuple(tuple(i.shape) for i in event.inputs),
        tuple(tuple(o.shape) for o in event.outputs),
    )


def _to_tensor_ref(tensor: torch.Tensor, name: str, tensor_id: int) -> TensorRef:
    dtype = dtype_to_str(tensor.dtype)
    shape = tuple(int(d) for d in tensor.shape)
    return TensorRef(
        tensor_id=tensor_id,
        name=name,
        shape=shape,
        dtype=dtype,
        device=str(tensor.device),
        num_bytes=tensor_volume_bytes(shape, dtype),
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
        record_memory: bool = True,
    ) -> None:
        super().__init__()
        self.cost_model = cost_model or OpCostModel()
        self.module_path_tracker = module_path_tracker
        self.phase_provider = phase_provider
        self.record_memory = record_memory
        self._events: list[_RawEvent] = []
        self._memory_events: list[RawMemoryEvent] = []
        self._producer: dict[int, int] = {}
        self._tensor_identities: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int]] = {}
        self._reused_tensor_ids = itertools.count(1)
        self._last_signature: tuple | None = None
        self._previous_active_capture: OpDispatchCapture | None = None
        self._capture_l0: bool = True  # pass-through when False (duplicate class)
        self._pending_comm_links: dict[int, object] = {}  # id(tensor) → CommEvent for dst_entry_op resolution
        # Per-(stage, comp_type) class dedup: the FIRST occurrence of each
        # class is captured in full (becomes a StepGraph template), every later
        # occurrence is a pass-through that only bumps the instance count. This
        # captures every distinct compute graph once while keeping capture cost
        # proportional to the number of distinct classes (not num_microbatches).
        self._captured_classes: set[tuple[int, str]] = set()
        self._class_instance_counts: dict[tuple[int, str], int] = {}
        self._chunk_class_key: tuple[int, str] | None = None

    def begin_chunk(self, class_key: tuple[int, str]) -> None:
        """Mark the start of one pipeline compute chunk (one
        forward_one_chunk / backward_one_chunk / backward_weight_one_chunk
        call). `class_key = (pp_stage, comp_type)`. If this class has not
        been captured yet, enable full L0 capture for this chunk; otherwise
        disable it (pass-through) and bump the class's instance count so
        the L2 schedule can still instantiate the matching template for
        this microbatch. Pairs with `end_chunk`."""
        self._chunk_class_key = class_key
        if class_key in self._captured_classes:
            self._capture_l0 = False
            self._class_instance_counts[class_key] = self._class_instance_counts.get(class_key, 1) + 1
        else:
            self._captured_classes.add(class_key)
            self._capture_l0 = True
            self._class_instance_counts[class_key] = 1

    def end_chunk(self) -> None:
        """Mark the end of a compute chunk. Restores L0 capture so that
        inter-chunk communication / the optimizer phase (which run outside
        any chunk) are still recorded, matching the pre-existing behavior."""
        self._capture_l0 = True
        self._chunk_class_key = None

    @property
    def class_instance_counts(self) -> dict[tuple[int, str], int]:
        """Per-class instance counts (number of microbatches that executed
        each captured template). Consumed by the L2 schedule builder to
        instantiate StepInstances for every microbatch."""
        return self._class_instance_counts

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
        *,
        logical_dtensor_shapes: bool = False,
        memory_inputs: list[torch.Tensor] | None = None,
        memory_outputs: list[torch.Tensor] | None = None,
    ) -> None:
        """Manually register one synthetic L0 event, as if `raw_op_type` had
        gone through __torch_dispatch__ normally. Used by
        torchtitan_npu.simulator.hardware_shims for ops that cannot execute
        for real (raw Triton kernels / JIT-compiled extensions) but whose
        real op name + output shape are known analytically. Participates in
        the same producer/consumer id(tensor) wiring, repeat_count dedup,
        and phase tagging as real dispatched events.

        ``logical_dtensor_shapes`` keeps local tensors for dependency and
        memory tracking, but uses DTensor global shapes in the OpNode metadata.
        This is intended for logical fused ops such as optimizer updates.

        ``memory_inputs`` and ``memory_outputs`` may expose additional operands
        to tensor-lifetime tracking without adding them to the logical OpNode.
        """
        logical_input_metas = None
        logical_output_metas = None
        if logical_dtensor_shapes:
            logical_input_metas = [
                to_tensor_meta(tensor, name=f"in_{idx}")
                for idx, tensor in enumerate(
                    _flatten_tensors(inputs, localize_dtensor=False)
                )
            ]
            logical_output_metas = [
                to_tensor_meta(tensor, name=f"out_{idx}")
                for idx, tensor in enumerate(
                    _flatten_tensors(outputs, localize_dtensor=False)
                )
            ]
        self._record_event(
            raw_op_type,
            _flatten_tensors(inputs),
            _flatten_tensors(outputs),
            module_path,
            input_metas=logical_input_metas,
            output_metas=logical_output_metas,
            tensor_shape_scope="global" if logical_dtensor_shapes else "local",
            memory_flat_inputs=(
                _flatten_tensors(memory_inputs)
                if memory_inputs is not None
                else None
            ),
            memory_flat_outputs=(
                _flatten_tensors(memory_outputs)
                if memory_outputs is not None
                else None
            ),
        )

    def _record_event(
        self,
        raw_op_type: str,
        flat_inputs: list[torch.Tensor],
        flat_outputs: list[torch.Tensor],
        module_path: str,
        *,
        input_metas: list[TensorMeta] | None = None,
        output_metas: list[TensorMeta] | None = None,
        tensor_shape_scope: str = "local",
        memory_flat_inputs: list[torch.Tensor] | None = None,
        memory_flat_outputs: list[torch.Tensor] | None = None,
    ) -> None:
        if not self._capture_l0:
            return  # pass-through: duplicate (stage, comp_type) class skips L0 capture
        # Skip the framework metadata-inference forward (DYNAMIC-mode
        # `_compute_outputs` and the FSDP _lazy_init/unshard setup it triggers),
        # which is a shape-inference artifact, not a training microbatch — see
        # meta_env._in_metadata_inference (set around _prepare_forward_infra /
        # _prepare_backward_infra / _compute_outputs).
        try:
            from torchtitan_npu.simulator.meta_env import _in_metadata_inference
            if _in_metadata_inference:
                return
        except Exception:
            pass
        input_ids = [self.tensor_id(tensor) for tensor in flat_inputs]
        output_ids = [self.tensor_id(tensor) for tensor in flat_outputs]
        memory_flat_inputs = flat_inputs if memory_flat_inputs is None else memory_flat_inputs
        memory_flat_outputs = flat_outputs if memory_flat_outputs is None else memory_flat_outputs
        memory_input_ids = [self.tensor_id(tensor) for tensor in memory_flat_inputs]
        memory_output_ids = [self.tensor_id(tensor) for tensor in memory_flat_outputs]
        predecessors = sorted({self._producer[tensor_id] for tensor_id in input_ids if tensor_id in self._producer})
        if input_metas is None:
            input_metas = [to_tensor_meta(t, name=f"in_{i}") for i, t in enumerate(flat_inputs)]
        if output_metas is None:
            output_metas = [to_tensor_meta(t, name=f"out_{i}") for i, t in enumerate(flat_outputs)]
        if len(input_metas) != len(flat_inputs) or len(output_metas) != len(flat_outputs):
            raise ValueError("Logical and local synthetic-op tensor counts must match")

        op_type = to_canonical_op_type(raw_op_type)
        phase = self.phase_provider() if self.phase_provider else "forward"
        execution_kind = current_execution_kind(phase)

        # Read PP context for this op's stage/mb/comp_type/fsdp_state attribution.
        # `comp_type` is set per-chunk by the patched forward_one_chunk /
        # backward_one_chunk / backward_weight_one_chunk (see meta_env.py).
        # For non-PP steps (no chunk patches run), derive comp_type from phase.
        pp_stage = -1
        pp_mb_idx = -1
        comp_type = ""
        fsdp_state = "NA"
        try:
            from torchtitan_npu.simulator.meta_env import _pp_context
            pp_stage = int(_pp_context.get("stage", -1))
            pp_mb_idx = int(_pp_context.get("mb_idx", -1))
            comp_type = str(_pp_context.get("comp_type", ""))
            fsdp_state = str(_pp_context.get("fsdp_state", "NA"))
        except Exception:
            pass
        if phase == "optimizer":
            comp_type = "OPTIMIZER"
        elif not comp_type or comp_type == "F":
            # Only honor an explicit "F" during forward; otherwise derive from
            # phase so non-PP backward ops are not mislabeled "F" (the default).
            if phase == "backward":
                comp_type = "B"
            else:
                comp_type = "F"

        candidate = _RawEvent(
            op_id=0,
            raw_op_type=raw_op_type,
            op_type=op_type,
            inputs=input_metas,
            outputs=output_metas,
            predecessors=predecessors,
            module_path=module_path,
            phase=phase,
            execution_kind=execution_kind,
            seq_idx=next(_seq_counter),
            pp_stage=pp_stage,
            pp_mb_idx=pp_mb_idx,
            comp_type=comp_type,
            fsdp_state=fsdp_state,
            tensor_shape_scope=tensor_shape_scope,
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

        if self.record_memory:
            self._memory_events.append(
                RawMemoryEvent(
                    event_id=next(_memory_event_counter),
                    op_id=op_id,
                    seq_idx=candidate.seq_idx,
                    raw_op_type=raw_op_type,
                    op_type=op_type,
                    phase=phase,
                    execution_kind=execution_kind,
                    module_path=module_path,
                    inputs=tuple(
                        _to_tensor_ref(tensor, name=f"in_{idx}", tensor_id=memory_input_ids[idx])
                        for idx, tensor in enumerate(memory_flat_inputs)
                    ),
                    outputs=tuple(
                        _to_tensor_ref(tensor, name=f"out_{idx}", tensor_id=memory_output_ids[idx])
                        for idx, tensor in enumerate(memory_flat_outputs)
                    ),
                    pp_stage=pp_stage,
                    pp_mb_idx=pp_mb_idx,
                    comp_type=comp_type,
                )
            )

        # Resolve pending comm links: if any input tensor was produced by a
        # comm op (e.g. recv/unshard), this op is the dst_entry_op consumer.
        for tid in input_ids:
            if tid in self._pending_comm_links:
                event = self._pending_comm_links.pop(tid)
                event.dst_entry_op = op_id
        self._last_signature = signature

        for tid in output_ids:
            self._producer[tid] = op_id

    def tensor_id(self, tensor: torch.Tensor) -> int:
        raw_id = id(tensor)
        identity = self._tensor_identities.get(raw_id)
        if identity is not None and identity[0]() is tensor:
            return identity[1]

        stable_id = raw_id if identity is None else -next(self._reused_tensor_ids)
        self._tensor_identities[raw_id] = (weakref.ref(tensor), stable_id)
        return stable_id

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
            annotations: dict[str, Any] = {
                "raw_op_type": event.raw_op_type,
                "phase": event.phase,
                "execution_kind": event.execution_kind,
                "is_recompute": event.execution_kind == "recompute",
                # Always stamp comp_type/fsdp_state so build_step_graphs can
                # bucket by (stage, comp_type) regardless of capture order.
                "comp_type": event.comp_type,
                "fsdp_state": event.fsdp_state,
            }
            if event.module_path:
                annotations["module_path"] = event.module_path
            if event.tensor_shape_scope != "local":
                annotations["tensor_shape_scope"] = event.tensor_shape_scope
            if event.repeat_count > 1:
                annotations["repeat_count"] = event.repeat_count
            if cost.unknown:
                annotations["cost_unknown"] = True
            if event.comm_dim:
                annotations["comm_dim"] = event.comm_dim
            if event.comm_ranks_str:
                annotations["comm_ranks"] = event.comm_ranks_str
            if event.pp_stage >= 0:
                annotations["pp_stage"] = event.pp_stage
            if event.pp_mb_idx >= 0:
                annotations["pp_mb_idx"] = event.pp_mb_idx
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
                seq_idx=event.seq_idx,
            )
        for op_id, node in nodes.items():
            for pred_id in node.predecessors:
                if pred_id in nodes:
                    nodes[pred_id].successors.append(op_id)
        return nodes

    def memory_events(self) -> list[RawMemoryEvent]:
        """Return the uncollapsed op stream used by static memory planning."""
        return list(self._memory_events)


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
