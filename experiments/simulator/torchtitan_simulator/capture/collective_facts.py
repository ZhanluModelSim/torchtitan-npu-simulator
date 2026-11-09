# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Versioned semantic facts captured while native collective objects are live."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from collections.abc import Iterator

_SUPPORTED = {
    "c10d._allgather_base_.default",
    "c10d._reduce_scatter_base_.default",
    "c10d.allreduce_.default",
    "_c10d_functional.all_to_all_single.default",
}
_A2A = "_c10d_functional.all_to_all_single.default"
_PROOF_METADATA_KEY = "torchtitan_simulator.collective_facts.v1"
_exchange_facts: ContextVar[dict[str, Any] | None] = ContextVar("simulator_collective_facts", default=None)


def reduction_name(value: object) -> str | None:
    """Decode installed named enums before TorchBind objects are stringified."""
    if isinstance(value, torch.ScriptObject):
        try:
            value = value.op()
        except (AttributeError, RuntimeError):
            return None
    elif isinstance(value, dist.ReduceOp):
        value = value.op
    # TorchBind op() returns an integer in some builds. Compare against this
    # installation's enum, never a hardcoded integer or an opaque print value.
    for name in ("SUM", "AVG", "MAX", "MIN", "PRODUCT", "PREMUL_SUM", "BAND", "BOR", "BXOR"):
        named = getattr(dist.ReduceOp, name, None)
        if named is not None and not isinstance(value, (str, bool)) and value == named:
            return name
    return None


def _schema_argument(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
    for index, argument in enumerate(func._schema.arguments):
        if argument.name == name:
            return args[index] if index < len(args) else kwargs.get(name, argument.default_value)
    return None


@contextmanager
def collective_fact_scope(facts: dict[str, Any]) -> Iterator[None]:
    """Bind producer-proven full-group geometry for one native exchange."""
    token = _exchange_facts.set(deepcopy(facts))
    try:
        yield
    finally:
        _exchange_facts.reset(token)


def retain_backward_facts(result: torch.Tensor, facts: dict[str, Any]) -> None:
    """Keep evidence on the exact native autograd node without patching it."""
    pending = [result.grad_fn]
    seen = set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if node.name() == "GeneratedBackwardFor__c10d_functional_all_to_all_single_defaultBackward":
            node.metadata[_PROOF_METADATA_KEY] = deepcopy(facts)
            return
        # wait_tensor / wrapping nodes can sit above the real collective node.
        # Stop at the first native collective; do not attach to an older input.
        pending.extend(function for function, _index in node.next_functions if function is not None)


def _active_exchange_facts() -> dict[str, Any] | None:
    facts = _exchange_facts.get()
    if facts is not None:
        return deepcopy(facts)
    current_node = getattr(torch._C, "_current_autograd_node", None)
    node = current_node() if current_node is not None else None
    if node is None or _PROOF_METADATA_KEY not in node.metadata:
        return None
    facts = deepcopy(node.metadata[_PROOF_METADATA_KEY])
    if "a2a_proof" in facts:
        proof = facts["a2a_proof"]
        proof["direction"] = "combine" if proof["direction"] == "dispatch" else "dispatch"
    if "send_rows" in facts:
        rows = facts["send_rows"]
        facts["send_rows"] = [list(column) for column in zip(*rows, strict=True)]
    if "recv_rows" in facts:
        rows = facts["recv_rows"]
        facts["recv_rows"] = [list(column) for column in zip(*rows, strict=True)]
    input_shapes = facts.pop("rank_input_shapes", None)
    output_shapes = facts.pop("rank_output_shapes", None)
    if output_shapes is not None:
        facts["rank_input_shapes"] = output_shapes
    if input_shapes is not None:
        facts["rank_output_shapes"] = input_shapes
    return facts


def _buffer_relation(primitive, inputs, outputs, group, capture_rank):
    if primitive == "c10d.allreduce_.default":
        return "inplace"
    # This field describes every participant. A single rank's alias observation
    # cannot prove that other ranks also placed their shard in the output.
    if group is None or len(group) != 1 or len(inputs) != 1 or len(outputs) != 1:
        return "distinct"
    source, target = inputs[0], outputs[0]
    if source.dtype != target.dtype or source.untyped_storage()._cdata != target.untyped_storage()._cdata:
        return "distinct"
    if primitive == "c10d._allgather_base_.default":
        expected_offset = target.storage_offset() + group.index(capture_rank) * source.numel()
        if source.storage_offset() == expected_offset:
            return "same_region"
    elif len(group) == 1 and source.storage_offset() == target.storage_offset():
        return "same_region"
    return "distinct"


def native_collective_facts(func, args, kwargs, operands, result) -> dict[str, Any] | None:
    """Deliver facts, with missing inputs explicit rather than inferred defaults."""
    primitive = str(func)
    if primitive not in _SUPPORTED:
        return None
    from torchtitan_simulator.capture.comm_events import _resolve_comm_ranks

    capture_rank = dist.get_rank() if dist.is_initialized() else None
    groups = _resolve_comm_ranks(operands.group)
    matching = [group for group in groups if capture_rank in group]
    group = matching[0] if len(matching) == 1 else None
    outputs = (result,) if operands.outputs is None else operands.outputs
    dense = all(tensor.layout == torch.strided and tensor.is_contiguous() for tensor in (*operands.inputs, *outputs))
    facts: dict[str, Any] = {
        "schema_version": 1,
        "primitive": primitive,
        "capture_rank": capture_rank,
        "ordered_group": group,
        "source": "native_collective_contract_v1",
        "tensor_contract": {
            "dense": dense,
            "uniform_across_ranks": primitive != _A2A,
            "source": "native_collective_contract_v1",
        },
        "buffer_relation": _buffer_relation(primitive, operands.inputs, outputs, group, capture_rank),
    }
    missing = []
    if capture_rank is None:
        missing.append("capture_rank")
    if group is None:
        missing.append("ordered_group")
    if primitive in {"c10d.allreduce_.default", "c10d._reduce_scatter_base_.default"}:
        operation = reduction_name(_schema_argument(func, args, kwargs, "reduce_op"))
        facts["reduction"] = {
            "operation": operation,
            "pre_scale": 1.0,
            "post_scale": 1.0,
            "scale_implemented_by_nodes": False,
            "source": "installed_torch_named_reduceop_v1",
        }
        if operation is None:
            missing.append("reduction.operation")
    if primitive == _A2A:
        exchange = _active_exchange_facts()
        if exchange is None:
            missing.append("a2a_full_group_split_facts")
        else:
            uniform = exchange.pop("uniform_across_ranks", False)
            facts.update(exchange)
            facts["tensor_contract"] = {
                **facts["tensor_contract"],
                "uniform_across_ranks": uniform,
                "source": exchange.get("source", facts["source"]),
            }
    if missing:
        facts["unresolved_fields"] = missing
    return facts
