# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU whole-graph profiling and per-node cost extraction."""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.fx as fx
import torch_npu
from torch.utils import _pytree
from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_COUNT_EXCHANGE,
    _EP_TOKEN_EXCHANGE,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    is_c10d_functional_node,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import run_traced
from torchtitan.tools.logging import logger

from .utils import (
    canonical_order,
    is_metadata_only_compute_node,
    isolated_cann_profiler_work_path,
    node_id,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_WHOLE_GRAPH_WARMUP_RUNS = 3
_CALIBRATION_GRAPH_SCOPE = "torchtitan_npu::auto_overlap_calibration_graph"

# Exclude device control tasks from compute costs. Collective costs retain their
# complete HCOM interval because its synchronization latency is overlapable.
_DEVICE_CONTROL_TASK_PREFIXES = (
    "EVENT_",
    "NOTIFY_",
    "PLACE_HOLDER",
    "PROFILING_",
    "STREAM_",
    "WRITE_VALUE",
)
_DEVICE_CONTROL_TASK_TYPES = {"COMMUNICATION"}


@dataclass(frozen=True)
class WholeGraphProfileCosts:
    """Cross-rank aligned costs keyed by stable FX node identity."""

    local_ms: dict[int, float]
    aligned_ms: dict[int, float]
    measured_ids: frozenset[int]


@dataclass(frozen=True)
class _BufferState:
    name: str
    buffer: torch.Tensor
    version: int
    snapshot: torch.Tensor


@contextmanager
def _preserve_model_state(model) -> Iterator[None]:
    """Restore mutated buffers and reject parameter or gradient changes."""
    parameter_states = tuple(
        (
            name,
            parameter,
            parameter._version,
            parameter.grad,
            None if parameter.grad is None else parameter.grad._version,
        )
        for name, parameter in model.named_parameters()
    )
    buffer_states = tuple(
        _BufferState(name, buffer, buffer._version, buffer.detach().clone()) for name, buffer in model.named_buffers()
    )
    try:
        yield
    finally:
        mutated_parameters = [
            name for name, parameter, version, _, _ in parameter_states if parameter._version != version
        ]
        mutated_gradients = [
            name
            for name, parameter, _, gradient, gradient_version in parameter_states
            if parameter.grad is not gradient
            or (gradient is not None and gradient_version is not None and gradient._version != gradient_version)
        ]
        mutated_buffers = [
            state
            for state in buffer_states
            if state.buffer._version != state.version and not torch.equal(state.buffer, state.snapshot)
        ]
        if mutated_buffers:
            with torch.no_grad():
                for state in mutated_buffers:
                    state.buffer.copy_(state.snapshot)
        restore_failed_buffers = [
            state.name for state in mutated_buffers if not torch.equal(state.buffer, state.snapshot)
        ]
        if mutated_parameters or mutated_gradients or restore_failed_buffers:
            raise RuntimeError(
                "NPU auto-overlap calibration mutated training state: "
                f"parameters={mutated_parameters[:8]}, "
                f"gradients={mutated_gradients[:8]}, "
                f"buffer_restore_failed={restore_failed_buffers[:8]}"
            )


def make_calibration_runner(traced_result, runtime_context):
    """Build a state-preserving runner for whole-graph calibration."""
    # Accept both the typed construction-time context and the legacy dict
    # supplied later by the compatibility patch.
    get_context_value = (
        runtime_context.__getitem__
        if isinstance(runtime_context, dict)
        else lambda name: getattr(runtime_context, name)
    )
    model = get_context_value("module")
    if traced_result is None:
        traced_result = get_context_value("traced_result")
    training_args = get_context_value("args")
    train_context = get_context_value("train_context")

    def run_candidate(candidate_gm):
        calibration_args = _pytree.tree_map(
            lambda value: value.clone() if isinstance(value, torch.Tensor) else value,
            training_args,
        )
        candidate = replace(traced_result, gm=candidate_gm)
        try:
            # Keep input/state preparation and restoration outside the scope
            # used to attribute profiler events to FX nodes.
            with (
                _preserve_model_state(model),
                train_context(),
                torch.profiler.record_function(_CALIBRATION_GRAPH_SCOPE),
            ):
                return run_traced(candidate, module=model)(*calibration_args)
        finally:
            del calibration_args

    return run_candidate


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


def _collective_group(node: fx.Node) -> str:
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


def _is_collective_launch(node: fx.Node) -> bool:
    return is_c10d_functional_node(node) and node.target != torch.ops._c10d_functional.wait_tensor.default


def _collective_group_ordinals(gm: fx.GraphModule) -> dict[str, int]:
    """Name rank-local groups by first appearance, independent of group hashes."""
    ordinals: dict[str, int] = {}
    for node in sorted(gm.graph.nodes, key=canonical_order(gm).__getitem__):
        if _is_collective_launch(node):
            group = _collective_group(node)
            if group not in ordinals:
                ordinals[group] = len(ordinals)
    return ordinals


def _collective_count_manifest(gm: fx.GraphModule) -> tuple[tuple[int, str, int], ...]:
    """Count FX collective launches by logical local group and collective type."""
    ordinals = _collective_group_ordinals(gm)
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for node in gm.graph.nodes:
        if _is_collective_launch(node):
            counts[ordinals[_collective_group(node)], _collective_type(node)] += 1
    return tuple(sorted((group, collective, count) for (group, collective), count in counts.items()))


def _validate_collective_counts_across_ranks(gm: fx.GraphModule) -> None:
    """Require only equal per-group/per-type FX communication stage counts."""
    import torch.distributed as dist

    manifest = _collective_count_manifest(gm)
    if not dist.is_available() or not dist.is_initialized():
        return
    gathered: list[tuple[tuple[int, str, int], ...] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, manifest)
    if any(item != manifest for item in gathered):
        raise RuntimeError(f"NPU auto-overlap FX collective counts differ across ranks: manifests={gathered}")


def _is_profile_cost_node(node: fx.Node) -> bool:
    return node.op in {"call_function", "call_method", "call_module"}


# Some composite operators emit the expanded CPU operation name instead.
_HOST_OP_ALIASES = {
    "torchtitan::deterministic_scatter_add": ("aten::scatter_add",),
    "torchtitan_npu::npu_moe_token_unpermute_grad_unweighted": ("npu::npu_moe_token_unpermute_grad",),
}


def _host_op_names(node: fx.Node) -> tuple[str, ...]:
    """Return native profiler CPU scope names expected for one FX call."""
    target = node.target
    schema = getattr(target, "_schema", None)
    schema_name = getattr(schema, "name", None)
    if isinstance(schema_name, str):
        return (schema_name, *_HOST_OP_ALIASES.get(schema_name, ()))
    if node.op == "call_method" and isinstance(target, str):
        return (f"aten::{target}",)
    return ()


def _match_fx_nodes_to_host_scopes(
    trace: dict[str, Any] | list[Any],
    gm: fx.GraphModule,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Match ordered FX calls inside the calibration graph scope.

    Match the full call sequence to disambiguate repeated operation names.
    Unsupported calls remain unmatched and fall back to prior costs.
    """
    raw_events = trace.get("traceEvents", ()) if isinstance(trace, dict) else trace
    events = [event for event in raw_events if isinstance(event, dict)]
    specs = [
        (node_id(node), _host_op_names(node))
        for node in gm.graph.nodes
        if node_id(node) is not None and _host_op_names(node)
    ]
    graph_scopes = [
        event
        for event in events
        if event.get("ph") == "X" and event.get("name") == _CALIBRATION_GRAPH_SCOPE and "ts" in event and "dur" in event
    ]
    if len(graph_scopes) != 1:
        return {}, {
            "expected": len(specs),
            "matched": 0,
            "graph_scopes": len(graph_scopes),
        }

    graph_scope = graph_scopes[0]
    scope_begin = float(graph_scope["ts"])
    scope_end = scope_begin + float(graph_scope["dur"])
    scope_lane = graph_scope.get("pid"), graph_scope.get("tid")
    host_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") == "cpu_op"
        and (event.get("pid"), event.get("tid")) == scope_lane
        and "ts" in event
        and "dur" in event
        and scope_begin <= float(event["ts"])
        and float(event["ts"]) + float(event["dur"]) <= scope_end
    ]

    def match_lane(host_events):
        host_events.sort(key=lambda event: (float(event["ts"]), -float(event["dur"])))
        positions: dict[str, list[int]] = defaultdict(list)
        for index, event in enumerate(host_events):
            positions[str(event.get("name"))].append(index)
        cursor = 0
        matched_indices = {}
        for stable_id, names in specs:
            indices = positions.get(names[0], ())
            offset = bisect_left(indices, cursor)
            if offset == len(indices):
                continue
            index = indices[offset]
            matched_indices[stable_id] = index
            cursor = index + 1

        # Expanded aliases are optional: a runtime branch may skip the native
        # op. Resolve them only when every alias-bearing FX call between two
        # primary scopes can be matched, avoiding a shift to a later call.
        matched_spec_indices = [
            index for index, (stable_id, _names) in enumerate(specs) if stable_id in matched_indices
        ]
        boundaries = [-1, *matched_spec_indices, len(specs)]
        for left, right in pairwise(boundaries):
            pending = [(stable_id, names[1:]) for stable_id, names in specs[left + 1 : right] if len(names) > 1]
            if not pending:
                continue
            lower = 0 if left < 0 else matched_indices[specs[left][0]] + 1
            upper = len(host_events) if right == len(specs) else matched_indices[specs[right][0]]
            tentative = {}
            alias_cursor = lower
            for stable_id, aliases in pending:
                choices = []
                for alias in aliases:
                    indices = positions.get(alias, ())
                    offset = bisect_left(indices, alias_cursor)
                    if offset < len(indices) and indices[offset] < upper:
                        choices.append(indices[offset])
                if not choices:
                    tentative.clear()
                    break
                index = min(choices)
                tentative[stable_id] = index
                alias_cursor = index + 1
            matched_indices.update(tentative)
        return {stable_id: host_events[index] for stable_id, index in matched_indices.items()}

    matched = match_lane(host_events)
    return matched, {
        "expected": len(specs),
        "matched": len(matched),
        "graph_scopes": 1,
    }


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


def _device_interval_union_us(events: list[dict[str, Any]]) -> float:
    intervals = sorted(
        (
            float(event["ts"]),
            float(event["ts"]) + float(event["dur"]),
        )
        for event in events
    )
    if not intervals:
        return 0.0
    total = 0.0
    begin, end = intervals[0]
    for next_begin, next_end in intervals[1:]:
        if next_begin <= end:
            end = max(end, next_end)
        else:
            total += end - begin
            begin, end = next_begin, next_end
    return total + end - begin


def _is_non_collective_device_work(event: dict[str, Any]) -> bool:
    """Return whether an event is device compute or copy work."""
    # Exclude host-side enqueue and dequeue ranges.
    if str(event.get("cat", "")).lower() in {"cpu_op", "python_function", "enqueue", "dequeue"}:
        return False
    name = str(event.get("name", "")).upper()
    args = event.get("args", {})
    task_type = ""
    if isinstance(args, dict):
        task_type = str(args.get("Task Type", args.get("task_type", ""))).upper()

    # Collective launches are measured separately through HCOM events.
    if name.startswith("HCOM_"):
        return False
    if task_type in _DEVICE_CONTROL_TASK_TYPES:
        return False
    return not any(
        value.startswith(prefix) for value in (task_type, name) for prefix in _DEVICE_CONTROL_TASK_PREFIXES if value
    )


def _is_host_only_scalar_node(node: fx.Node) -> bool:
    """Return whether ``_local_scalar_dense`` reads an existing CPU tensor."""
    if node.target is not torch.ops.aten._local_scalar_dense.default or not node.args:
        return False
    value = node.args[0]
    if isinstance(value, fx.Node):
        value = value.meta.get("val")
    return isinstance(value, torch.Tensor) and value.device.type == "cpu"


def _extract_cann_node_costs(
    trace: dict[str, Any] | list[Any],
    profile_nodes: dict[int, fx.Node],
    node_scopes: dict[int, dict[str, Any]],
    *,
    debug: bool = False,
) -> dict[int, float]:
    """Map CPU dispatcher scopes to device work through ``torch_to_npu`` flows."""
    raw_events = trace.get("traceEvents", ()) if isinstance(trace, dict) else trace
    events = [event for event in raw_events if isinstance(event, dict)]
    complete = [event for event in events if event.get("ph") == "X" and "ts" in event and "dur" in event]
    device_pids = {
        event.get("pid")
        for event in events
        if event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") in {"Ascend Hardware", "Communication"}
    }
    if not device_pids:
        logger.warning("NPU auto-overlap CANN device process metadata missing; no whole-graph costs extracted")
        return {}

    def flow_key(event):
        return event.get("cat"), event.get("name"), event.get("id")

    flows = [
        event
        for event in events
        if event.get("cat") == "async_npu" and event.get("name") == "torch_to_npu" and event.get("id") is not None
    ]
    flow_finishes = {flow_key(event): event for event in flows if event.get("ph") == "f"}
    starts_by_lane: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for event in flows:
        if event.get("ph") == "s" and "ts" in event:
            starts_by_lane.setdefault((event.get("pid"), event.get("tid")), []).append(event)
    start_times_by_lane: dict[tuple[Any, Any], list[float]] = {}
    for lane, starts in starts_by_lane.items():
        starts.sort(key=lambda event: float(event["ts"]))
        start_times_by_lane[lane] = [float(event["ts"]) for event in starts]
    # Match each flow finish to a device task starting at the same timestamp.
    # Fall back to interval containment when trace timestamp precision differs.
    device_by_start: dict[tuple[Any, Any, float], list[dict[str, Any]]] = {}
    device_by_lane: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for event in complete:
        category = str(event.get("cat", "")).lower()
        if event.get("pid") not in device_pids or category in {"cpu_op", "python_function", "enqueue", "dequeue"}:
            continue
        key = (
            event.get("pid"),
            event.get("tid"),
            round(float(event["ts"]), 3),
        )
        device_by_start.setdefault(key, []).append(event)
        device_by_lane.setdefault(key[:2], []).append(event)

    costs: dict[int, float] = {}
    missing_device_work: list[str] = []
    for stable_id, node in profile_nodes.items():
        assert stable_id == node_id(node)
        host_scope = node_scopes.get(stable_id)
        if is_metadata_only_compute_node(node) or _is_host_only_scalar_node(node):
            costs[stable_id] = 0.0
            continue
        if host_scope is None:
            # Python-only graph plumbing has no native scope or device work.
            # Warn only when a native scope was expected.
            if _host_op_names(node):
                logger.warning(
                    "NPU auto-overlap CANN host scope missing: node_id=%s node=%s target=%s",
                    stable_id,
                    node.name,
                    node.target,
                )
            continue
        begin = float(host_scope["ts"])
        end = begin + float(host_scope["dur"])
        associated: list[dict[str, Any]] = []
        lane = host_scope.get("pid"), host_scope.get("tid")
        start_times = start_times_by_lane.get(lane, ())
        starts = starts_by_lane.get(lane, ())
        # Restrict flow lookup to this host scope's lane and time range.
        for start in starts[bisect_left(start_times, begin) : bisect_right(start_times, end)]:
            finish = flow_finishes.get(flow_key(start))
            if finish is None:
                continue
            finish_ts = float(finish.get("ts", -1.0))
            key = (
                finish.get("pid"),
                finish.get("tid"),
                round(finish_ts, 3),
            )
            candidates = device_by_start.get(key, ())
            if not candidates:
                candidates = [
                    event
                    for event in device_by_lane.get(key[:2], ())
                    if float(event["ts"]) <= finish_ts <= float(event["ts"]) + float(event["dur"])
                ]
            associated.extend(candidates)
        associated = list({id(event): event for event in associated}.values())
        if _is_collective_launch(node):
            hcom = [event for event in associated if str(event.get("name", "")).lower().startswith("hcom_")]
            if not hcom:
                logger.warning(
                    "NPU auto-overlap CANN HCOM association missing: node_id=%s node=%s target=%s associated=%s",
                    stable_id,
                    node.name,
                    node.target,
                    [event.get("name") for event in associated],
                )
                continue
            # The longest HCOM event covers the collective's overlapable
            # launch-to-ready interval, including protocol synchronization.
            cost_us = max(float(event["dur"]) for event in hcom)
        else:
            device_work = [event for event in associated if _is_non_collective_device_work(event)]
            if not device_work:
                missing_device_work.append(f"{stable_id}:{node.name}")
                continue
            cost_us = _device_interval_union_us(device_work)
        costs[stable_id] = max(0.0, cost_us / 1000.0)
    if missing_device_work and debug:
        logger.info(
            "NPU auto-overlap CANN nodes without measured device work: count=%d fallback=previous_round examples=%s",
            len(missing_device_work),
            missing_device_work[:8],
        )
    return costs


def _local_profile_costs(
    gm: fx.GraphModule,
    run_candidate,
    profile_node_ids: frozenset[int] | None,
    calibration_round: int = 1,
    debug: bool = False,
) -> tuple[dict[int, float], Any]:
    import torch_npu

    profile_nodes = {
        stable_id: node
        for node in gm.graph.nodes
        if (stable_id := node_id(node)) is not None
        and _is_profile_cost_node(node)
        and (profile_node_ids is None or stable_id in profile_node_ids)
    }
    with isolated_cann_profiler_work_path(
        prefix=f"auto_overlap_whole_graph_bench_round{calibration_round}_",
        retain=debug,
    ) as work_path:
        trace_path = work_path / "trace.json"
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            record_op_args=False,
        )
        torch_npu.npu.synchronize()
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        ) as profiler:
            result = run_candidate(gm)
            torch_npu.npu.synchronize()
        profiler.export_chrome_trace(str(trace_path))
        if not trace_path.is_file():
            raise RuntimeError("CANN profiler did not export a calibration trace")
        with trace_path.open("r", encoding="utf-8") as trace_file:
            trace = json.load(trace_file)
        host_scopes, match_stats = _match_fx_nodes_to_host_scopes(trace, gm)
        costs = (
            _extract_cann_node_costs(
                trace,
                profile_nodes,
                host_scopes,
                debug=debug,
            )
            if match_stats["graph_scopes"] == 1
            else {}
        )
        requested_host_ids = {stable_id for stable_id, node in profile_nodes.items() if _host_op_names(node)}
        requested_matched_ids = requested_host_ids & host_scopes.keys()
        if match_stats["graph_scopes"] != 1:
            logger.warning(
                "NPU auto-overlap calibration graph scope count is invalid: "
                "calibration_round=%d count=%d expected=1; all ranks will "
                "use previous-round costs",
                calibration_round,
                match_stats["graph_scopes"],
            )
        logger.info(
            "NPU auto-overlap host-order mapping: calibration_round=%d "
            "fx_host_ops_matched=%d/%d requested_host_ops_matched=%d/%d "
            "extracted_costs=%d",
            calibration_round,
            match_stats["matched"],
            match_stats["expected"],
            len(requested_matched_ids),
            len(requested_host_ids),
            len(costs),
        )
        if requested_matched_ids != requested_host_ids:
            logger.warning(
                "NPU auto-overlap host-order mapping missing requested native scopes: "
                "calibration_round=%d missing=%d examples=%s; all ranks will "
                "use previous-round costs for incomplete node IDs",
                calibration_round,
                len(requested_host_ids - requested_matched_ids),
                sorted(requested_host_ids - requested_matched_ids)[:8],
            )
    return costs, result


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


def _cross_rank_semantic_keys(
    gm: fx.GraphModule,
    profile_node_ids: frozenset[int] | None,
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

        if _is_collective_launch(node):
            group = group_ordinals[_collective_group(node)]
            collective = _collective_type(node)
            counter_key = (group, collective)
            occurrence = communication_counts[counter_key]
            communication_counts[counter_key] += 1
            if profile_node_ids is None or stable_id in profile_node_ids:
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

        if profile_node_ids is not None and stable_id not in profile_node_ids:
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
        result[stable_id] = ("compute", signature, occurrence)
    return result


def _align_costs(
    gm: fx.GraphModule,
    local: dict[int, float],
    profile_node_ids: frozenset[int] | None = None,
) -> dict[int, float]:
    import torch.distributed as dist

    semantic_keys = _cross_rank_semantic_keys(gm, profile_node_ids)
    # Keep global semantic numbering unchanged, but gather only one MoE layer
    # at a time. Forward/backward and both chunks share the same region FQN.
    # Untagged nodes form a separate batch for callers outside chunked MoE.
    batches: dict[str, list[int]] = defaultdict(list)
    for node in gm.graph.nodes:
        stable_id = node_id(node)
        if stable_id is not None and stable_id in semantic_keys:
            batches[str(node.meta.get("chunked_region_fqn") or "")].append(stable_id)
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    batch_names = sorted(batches)
    if distributed:
        # Agree on a union, not local graph order: a rank missing a layer must
        # still participate with an empty dictionary to preserve fallback.
        manifests: list[list[str] | None] = [None] * world_size
        dist.all_gather_object(manifests, batch_names)
        if any(manifest is None for manifest in manifests):
            raise RuntimeError("NPU auto-overlap failed to gather profile layer manifest")
        batch_names = sorted({name for manifest in manifests if manifest is not None for name in manifest})
        del manifests
    aligned: dict[int, float] = {}
    for batch_name in batch_names:
        batch_ids = batches.get(batch_name, ())
        local_by_key = {
            semantic_keys[stable_id]: float(local[stable_id]) for stable_id in batch_ids if stable_id in local
        }
        gathered: list[dict[tuple[Any, ...], float] | None] = [None] * world_size
        if distributed:
            dist.all_gather_object(gathered, local_by_key)
        else:
            gathered[0] = local_by_key
        if any(item is None for item in gathered):
            raise RuntimeError("NPU auto-overlap failed to gather profile costs")
        concrete_costs = cast("list[dict[tuple[Any, ...], float]]", gathered)
        for stable_id in batch_ids:
            semantic_key = semantic_keys[stable_id]
            values = [float(item[semantic_key]) for item in concrete_costs if semantic_key in item]
            # Incomplete measurements must fall back identically on every rank.
            if len(values) == world_size:
                aligned[stable_id] = min(values)
        # Release all remote layer dictionaries before the next gather.
        del concrete_costs, gathered, local_by_key
    return aligned


def _gather_profile_errors(local_error: str | None) -> list[str | None]:
    """Share rank-local profiling failures before cost alignment."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return [local_error]
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    return errors


def profile_whole_graph_costs(
    gm: fx.GraphModule,
    run_candidate,
    *,
    profile_node_ids: frozenset[int] | None = None,
    calibration_round: int = 1,
    debug: bool = False,
) -> WholeGraphProfileCosts | None:
    """Profile an isolated full-graph execution and align costs by rank minimum.

    Extract costs only for ``profile_node_ids`` by associating native dispatcher
    scopes with their device work.
    """
    local: dict[int, float] = {}
    result = None
    local_error = None
    try:
        _validate_collective_counts_across_ranks(gm)
        cpu_rng_before = torch.random.get_rng_state().clone()
        npu_rng_before = torch_npu.npu.get_rng_state().clone()
        with torch.random.fork_rng(
            devices=[torch_npu.npu.current_device()],
            device_type="npu",
        ):
            # Warm up before the measured run. Replay the same fork-local RNG
            # state for stable routed workloads.
            warmup_cpu_rng = torch.random.get_rng_state().clone()
            warmup_npu_rng = torch_npu.npu.get_rng_state().clone()
            for _ in range(_WHOLE_GRAPH_WARMUP_RUNS):
                torch.random.set_rng_state(warmup_cpu_rng)
                torch_npu.npu.set_rng_state(warmup_npu_rng)
                warmup_result = run_candidate(gm)
                torch_npu.npu.synchronize()
                del warmup_result
            torch.random.set_rng_state(warmup_cpu_rng)
            torch_npu.npu.set_rng_state(warmup_npu_rng)
            local, result = _local_profile_costs(
                gm,
                run_candidate,
                profile_node_ids,
                calibration_round=calibration_round,
                debug=debug,
            )
        if not torch.equal(cpu_rng_before, torch.random.get_rng_state()) or not torch.equal(
            npu_rng_before, torch_npu.npu.get_rng_state()
        ):
            raise RuntimeError("whole-graph calibration changed CPU or NPU RNG state")
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    finally:
        # Release calibration outputs before rank-status coordination.
        del result

    errors = _gather_profile_errors(local_error)
    if any(error is not None for error in errors):
        failures = {rank: error for rank, error in enumerate(errors) if error is not None}
        logger.warning(
            "NPU auto-overlap calibration round failed; keeping the previous "
            "schedule: calibration_round=%d failures=%s",
            calibration_round,
            failures,
        )
        return None

    aligned = _align_costs(
        gm,
        local,
        profile_node_ids,
    )
    return WholeGraphProfileCosts(
        local_ms=local,
        aligned_ms=aligned,
        measured_ids=frozenset(aligned),
    )
