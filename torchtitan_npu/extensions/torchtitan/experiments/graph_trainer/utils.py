# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared graph and profiling utilities for NPU auto-overlap."""

from __future__ import annotations

import logging
import operator
import os
import tempfile
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.fx as fx
from torch.utils import _pytree
from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_COUNT_EXCHANGE,
    _EP_TOKEN_EXCHANGE,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import is_c10d_functional_node

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_ASCEND_WORK_PATH = "ASCEND_WORK_PATH"
_NODE_ID_META = "npu_auto_overlap_node_id"

_ALWAYS_METADATA_ONLY_TARGETS = {
    operator.getitem,
    torch.ops.aten.alias.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.view.default,
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.expand.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.unbind.int,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.split_with_sizes.default,
}


def call_function_target(node: torch.fx.Node) -> Callable[..., Any]:
    """Return the callable target of an FX ``call_function`` node."""
    if node.op != "call_function" or not callable(node.target):
        raise AssertionError(f"expected an FX call_function with a callable target, got {node.op}: {node.target}")
    return node.target


def resolve_collective_process_group(node: torch.fx.Node) -> Any:
    """Resolve the live process group used by a functional collective."""
    from torch._inductor.fx_passes.bucketing import _resolve_group_name
    from torch.distributed.distributed_c10d import _resolve_process_group
    from torch.fx.operator_schemas import normalize_function

    normalized = normalize_function(
        call_function_target(node),
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("normalize_function returned None for collective node")
    return _resolve_process_group(_resolve_group_name(normalized[1]["group_name"]))


def _custom_meta(node: torch.fx.Node) -> dict[str, Any]:
    custom = node.meta.get("custom")
    return dict(custom) if isinstance(custom, dict) else {}


def assign_stable_node_tags(gm: torch.fx.GraphModule) -> None:
    """Assign pre-scheduling ordinals once and validate later calls."""
    nodes = tuple(gm.graph.nodes)
    tagged = [node for node in nodes if _NODE_ID_META in _custom_meta(node)]
    if not tagged:
        for ordinal, node in enumerate(nodes):
            custom = _custom_meta(node)
            custom[_NODE_ID_META] = ordinal
            node.meta["custom"] = custom
        return
    if len(tagged) != len(nodes):
        raise RuntimeError("FX graph is only partially tagged with auto-overlap node IDs")

    stable_ids = [node_id(node) for node in nodes]
    if any(stable_id is None for stable_id in stable_ids):
        raise RuntimeError("FX graph contains an invalid auto-overlap node ID")
    if len(set(stable_ids)) != len(stable_ids):
        raise RuntimeError("FX graph contains duplicate auto-overlap node IDs")


def node_id(node: torch.fx.Node) -> int | None:
    value = _custom_meta(node).get(_NODE_ID_META)
    return value if type(value) is int else None


def canonical_order(gm: torch.fx.GraphModule) -> dict[torch.fx.Node, int]:
    result: dict[torch.fx.Node, int] = {}
    for node in gm.graph.nodes:
        stable_id = node_id(node)
        if stable_id is None:
            raise RuntimeError(f"FX node {node.name} has no auto-overlap node ID")
        result[node] = stable_id
    return result


def _metadata_tensors(value: Any) -> tuple[torch.Tensor, ...]:
    return tuple(leaf for leaf in _pytree.tree_leaves(value) if isinstance(leaf, torch.Tensor))


def _reshape_shares_input_storage(node: torch.fx.Node) -> bool:
    """Return whether FakeTensor metadata proves ``reshape`` is view-only."""
    if not node.args or not isinstance(node.args[0], torch.fx.Node):
        return False
    inputs = _metadata_tensors(node.args[0].meta.get("val"))
    outputs = _metadata_tensors(node.meta.get("val"))
    if len(inputs) != 1:
        return False
    if len(outputs) == 1:
        try:
            if inputs[0].untyped_storage()._cdata == outputs[0].untyped_storage()._cdata:
                return True
        except (AttributeError, NotImplementedError, RuntimeError):
            pass
    from torch.fx.experimental.symbolic_shapes import statically_known_true

    value = inputs[0]
    if value.layout != torch.strided:
        return False
    # Keep symbolic checks guard-free; unknown contiguity may require a copy.
    if any(statically_known_true(size == 0) for size in value.shape):
        return True
    expected_stride = 1
    for size, stride in reversed(tuple(zip(value.shape, value.stride(), strict=True))):
        if statically_known_true(size == 1):
            continue
        if not statically_known_true(stride == expected_stride):
            return False
        expected_stride *= size
    return True


def is_metadata_only_compute_node(node: torch.fx.Node) -> bool:
    """Return whether an FX call only changes tensor metadata."""
    if node.op != "call_function":
        return False
    if node.target in _ALWAYS_METADATA_ONLY_TARGETS:
        return True
    return node.target == torch.ops.aten.reshape.default and _reshape_shares_input_storage(node)


@contextmanager
def isolated_cann_profiler_work_path(
    prefix: str,
    *,
    retain: bool = False,
) -> Iterator[Path]:
    """Use an isolated ``ASCEND_WORK_PATH`` for internal CANN profiling.

    Collection and export must finish inside the context. The directory is
    The directory is removed afterwards unless ``retain`` is true. The
    original environment value is always restored.
    """

    previous = os.environ.get(_ASCEND_WORK_PATH)
    with ExitStack() as stack:
        if retain:
            rank = os.getenv("RANK", "unknown")
            temp_dir = tempfile.mkdtemp(prefix=f"{prefix}rank{rank}_pid{os.getpid()}_")
        else:
            temp_dir = stack.enter_context(tempfile.TemporaryDirectory(prefix=prefix))
        work_path = Path(temp_dir).resolve()
        os.environ[_ASCEND_WORK_PATH] = str(work_path)
        try:
            if retain:
                logger.info("NPU auto-overlap benchmark profiling retained, path=%s", work_path)
            yield work_path
        finally:
            if previous is None:
                os.environ.pop(_ASCEND_WORK_PATH, None)
            else:
                os.environ[_ASCEND_WORK_PATH] = previous


def dump_fx_graph(
    gm: torch.fx.GraphModule,
    prefix: str,
) -> Path:
    """Save a readable FX graph as ``<prefix>_rank<rank>.py``."""
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    output_dir = Path.cwd() / "fx_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prefix}_rank{rank}.py"
    readable = gm.print_readable(
        print_output=False,
        include_stride=True,
        include_device=True,
        expanded_def=True,
    )
    output_path.write_text(readable, encoding="utf-8")
    logger.info("Saved FX graph to %s", output_path)
    return output_path


def _schema_arguments(node: fx.Node) -> dict[str, Any]:
    schema = getattr(node.target, "_schema", None)
    arguments = getattr(schema, "arguments", ())
    bound = {argument.name: value for argument, value in zip(arguments, node.args, strict=False)}
    bound.update(node.kwargs)
    return bound


def _collective_type(node: fx.Node) -> str:
    schema = getattr(node.target, "_schema", None)
    name = getattr(schema, "name", None)
    return str(name) if name is not None else str(node.target)


def collective_group(node: fx.Node) -> str:
    """Return the group identifier bound to a functional collective."""
    bound = _schema_arguments(node)
    for name in ("group_name", "group", "tag"):
        if name in bound:
            return str(bound[name])
    return f"unknown:{_collective_type(node)}"


def _collective_arguments_without_group(node: fx.Node) -> tuple[tuple[str, Any], ...]:
    return tuple(
        (name, _value_descriptor(value))
        for name, value in _schema_arguments(node).items()
        if name not in {"group_name", "group", "tag"}
    )


def is_collective_launch(node: fx.Node) -> bool:
    """Identify functional collective launches, excluding wait_tensor."""
    return is_c10d_functional_node(node) and node.target != torch.ops._c10d_functional.wait_tensor.default


def _collective_group_ordinals(gm: fx.GraphModule) -> dict[str, int]:
    """Name rank-local groups by first appearance, independent of group hashes."""
    ordinals: dict[str, int] = {}
    for node in sorted(gm.graph.nodes, key=canonical_order(gm).__getitem__):
        if is_collective_launch(node):
            group = collective_group(node)
            if group not in ordinals:
                ordinals[group] = len(ordinals)
    return ordinals


def _communication_role(node: fx.Node) -> str | None:
    if not is_c10d_functional_node(node):
        return None
    custom = node.meta.get("custom")
    if not isinstance(custom, dict):
        custom = {}
    if custom.get(_EP_TOKEN_COUNT_EXCHANGE) == "dispatch":
        return "count"
    role = custom.get(_EP_TOKEN_EXCHANGE)
    return str(role) if role is not None else "other"


def _dimension_descriptor(value: Any) -> int | tuple[str, int | None]:
    if isinstance(value, int):
        return value
    hint = getattr(value, "hint", None)
    if hint is None:
        hint = getattr(getattr(value, "node", None), "hint", None)
    return ("sym", int(hint) if hint is not None else None)


def _tensor_descriptor(value: torch.Tensor) -> tuple[Any, ...]:
    device = getattr(value, "device", None)
    device_type = getattr(device, "type", str(device))
    return (
        tuple(_dimension_descriptor(dim) for dim in value.shape),
        tuple(_dimension_descriptor(dim) for dim in value.stride()),
        str(value.dtype),
        str(device_type),
    )


def _value_descriptor(value: Any) -> Any:
    """Return a rank-independent, hashable FX value/argument description."""
    if isinstance(value, fx.Node):
        return ("fx", _value_descriptor(value.meta.get("val")))
    if isinstance(value, torch.Tensor):
        return ("tensor", *_tensor_descriptor(value))
    if isinstance(value, torch.SymInt):
        return _dimension_descriptor(value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_descriptor(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_value_descriptor(item) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                sorted(
                    ((str(key), _value_descriptor(item)) for key, item in value.items()),
                    key=lambda item: item[0],
                )
            ),
        )
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    # Use stable strings for immutable arguments; repr may contain addresses.
    return (type(value).__qualname__, str(value))


def _module_scope(node: fx.Node) -> tuple[str, ...]:
    stack = node.meta.get("nn_module_stack")
    if not isinstance(stack, dict):
        return ()
    result = []
    for key, value in stack.items():
        fqn = value[0] if isinstance(value, tuple) and value and isinstance(value[0], str) else key
        result.append(str(fqn))
    return tuple(result)


def _node_scope(node: fx.Node) -> tuple[Any, ...]:
    return (
        node.meta.get("chunked_region_fqn"),
        node.meta.get("chunked_region_is_backward"),
        node.meta.get("chunk_id"),
        _module_scope(node),
    )


def cross_rank_semantic_keys(
    gm: fx.GraphModule,
    node_ids: frozenset[int] | None = None,
) -> dict[int, tuple[Any, ...]]:
    """Map rank-local IDs to semantic keys, numbering duplicate signatures."""
    group_ordinals = _collective_group_ordinals(gm)
    communication_counts: dict[tuple[int, str], int] = defaultdict(int)
    compute_counts: dict[tuple[Any, ...], int] = defaultdict(int)
    result: dict[int, tuple[Any, ...]] = {}

    for node in sorted(gm.graph.nodes, key=canonical_order(gm).__getitem__):
        stable_id = node_id(node)
        if stable_id is None:
            continue

        if is_collective_launch(node):
            group = group_ordinals[collective_group(node)]
            collective = _collective_type(node)
            counter_key = (group, collective)
            occurrence = communication_counts[counter_key]
            communication_counts[counter_key] += 1
            if node_ids is None or stable_id in node_ids:
                result[stable_id] = (
                    "communication",
                    group,
                    collective,
                    occurrence,
                    _node_scope(node),
                    _communication_role(node),
                    _collective_arguments_without_group(node),
                    _value_descriptor(node.meta.get("val")),
                )
            continue

        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        if is_metadata_only_compute_node(node):
            continue

        signature = (
            _node_scope(node),
            node.op,
            str(node.target),
            _value_descriptor((node.args, node.kwargs)),
            _value_descriptor(node.meta.get("val")),
        )
        occurrence = compute_counts[signature]
        compute_counts[signature] += 1
        if node_ids is None or stable_id in node_ids:
            result[stable_id] = ("compute", signature, occurrence)
    return result
