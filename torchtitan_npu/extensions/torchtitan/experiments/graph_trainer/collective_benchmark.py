# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU runtime benchmarks for distributed collectives."""

from __future__ import annotations

import itertools
import json
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch_npu
from torch.utils import _python_dispatch, _pytree

from .utils import (
    call_function_target,
    isolated_cann_profiler_work_path,
    resolve_collective_process_group,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

_NPU_PROFILER_CACHE_TAG = "torch_profiler_npu"
_NPU_CANN_CACHE_TAG = "torch_npu_cann_hcom_benchmark"
_CANN_MARKER_PREFIX = "NPU_COMM_BENCH"
_CANN_WARMUP_RUNS = 20


@dataclass(frozen=True)
class _PreparedCollectiveBenchmark:
    node: torch.fx.Node
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    process_group: Any
    cache_key: str


@dataclass(frozen=True)
class _CollectiveBenchmarkSpec:
    """Metadata-only description of one unique collective benchmark."""

    node: torch.fx.Node
    process_group: Any
    cache_key: str
    materialize: Any


def _balanced_splits(total: int, parts: int) -> list[int]:
    """Split ``total`` elements as evenly as possible across ``parts``."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if parts <= 0:
        raise ValueError(f"parts must be positive, got {parts}")

    quotient, remainder = divmod(total, parts)
    return [quotient + int(index < remainder) for index in range(parts)]


def _build_a2a_split_matrix(rows_per_rank: Sequence[int]) -> list[list[int]]:
    """Build ``source x destination`` splits for a valid synthetic all-to-all."""
    world_size = len(rows_per_rank)
    if world_size == 0:
        raise ValueError("rows_per_rank must not be empty")
    return [_balanced_splits(int(rows), world_size) for rows in rows_per_rank]


def _a2a_splits_for_rank(
    split_matrix: Sequence[Sequence[int]],
    rank: int,
) -> tuple[list[int], list[int]]:
    """Return input/output split sizes for one rank from a split matrix."""
    world_size = len(split_matrix)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if any(len(row) != world_size for row in split_matrix):
        raise ValueError("split_matrix must be square")

    input_splits = list(split_matrix[rank])
    output_splits = [int(split_matrix[source][rank]) for source in range(world_size)]
    return input_splits, output_splits


def _contiguous_strides(shape: Sequence[int]) -> list[int]:
    """Return row-major strides for ``shape``."""
    strides = [0] * len(shape)
    running_stride = 1
    for index in range(len(shape) - 1, -1, -1):
        strides[index] = running_stride
        running_stride *= max(1, int(shape[index]))
    return strides


def _benchmark_a2a_with_npu_events(
    node: torch.fx.Node,
    nruns: int,
    *,
    uniform_dispatch_rows: int | None = None,
    reverse_uniform_splits: bool = False,
) -> tuple[float | None, str]:
    """Benchmark A2A using balanced synthetic splits.

    Resolve local routed rows from the dispatch assumption or FakeTensor
    metadata, then gather them across the communication group to construct
    compatible splits. Reverse the dispatch splits for a paired combine.
    """
    import torch.distributed as dist
    from torch._dynamo.testing import rand_strided
    from torch._inductor import config, fx_utils
    from torch._inductor.fx_passes import node_runtime_estimation
    from torch._inductor.fx_passes.bucketing import _resolve_group_name
    from torch.distributed.distributed_c10d import _resolve_process_group
    from torch.fx.experimental.symbolic_shapes import optimization_hint
    from torch.fx.operator_schemas import normalize_function

    if not node_runtime_estimation.can_benchmark_collective():
        return None, ""

    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success:
        return None, ""

    target = call_function_target(node)
    normalized = normalize_function(
        target,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("normalize_function returned None for all_to_all_single")
    group_name = _resolve_group_name(normalized[1]["group_name"])
    process_group = _resolve_process_group(group_name)
    world_size = dist.get_world_size(process_group)
    group_rank = dist.get_rank(process_group)

    args, kwargs = _pytree.tree_map_only(
        torch.SymInt,
        lambda value: optimization_hint(value, fallback=config.unbacked_symint_fallback),
        (args, kwargs),
    )

    if len(args) < 4 or not isinstance(args[0], torch.Tensor):
        raise AssertionError("unexpected all_to_all_single arguments")

    fake_input = args[0]
    shape = [node_runtime_estimation.get_hint(dim) for dim in fake_input.shape]
    if shape[0] is None:
        shape[0] = sum(int(value) for value in args[2])
    shape = [int(value) if value is not None else config.unbacked_symint_fallback for value in shape]
    stride_hints = [node_runtime_estimation.get_hint(value) for value in fake_input.stride()]
    stride = (
        _contiguous_strides(shape) if any(value is None for value in stride_hints) else cast("list[int]", stride_hints)
    )
    if uniform_dispatch_rows is not None:
        if uniform_dispatch_rows < 0:
            raise ValueError("uniform dispatch rows must be non-negative")
        local_rows = uniform_dispatch_rows
    else:
        local_rows = int(shape[0])
    rows_per_rank: list[int | None] = [None] * world_size
    dist.all_gather_object(rows_per_rank, local_rows, group=process_group)
    if any(rows is None for rows in rows_per_rank):
        raise AssertionError("failed to gather A2A input rows")

    split_matrix = _build_a2a_split_matrix(cast("list[int]", rows_per_rank))
    dispatch_input_splits, dispatch_output_splits = _a2a_splits_for_rank(split_matrix, group_rank)
    if reverse_uniform_splits:
        input_splits = dispatch_output_splits
        output_splits = dispatch_input_splits
    else:
        input_splits = dispatch_input_splits
        output_splits = dispatch_output_splits
    # For a combine, the local input contains everything received by the
    # paired dispatch (the column sum).
    shape[0] = sum(input_splits)
    input_tensor = rand_strided(
        shape,
        stride,
        device=fake_input.device,
        dtype=fake_input.dtype,
    )
    benchmark_args = (input_tensor, output_splits, input_splits, group_name)

    input_bytes = input_tensor.numel() * input_tensor.element_size()
    cache_key = (
        f"{node.target}: ({world_size} group size, {input_bytes} bytes, "
        f"shape={tuple(shape)}, dtype={input_tensor.dtype}, "
        f"input_splits={tuple(input_splits)}, "
        f"output_splits={tuple(output_splits)}, npu balanced splits)"
    )
    cached = node_runtime_estimation.get_cached_runtime(cache_key)
    cache_hits: list[bool | None] = [None] * world_size
    dist.all_gather_object(cache_hits, cached is not None, group=process_group)
    if all(cache_hits):
        if cached is None:
            raise AssertionError("collective cache agreement is inconsistent")
        return cached, cache_key

    dist.barrier(group=process_group)
    torch_npu.npu.synchronize()
    result = target(*benchmark_args, **kwargs)
    torch.ops._c10d_functional.wait_tensor(result)
    torch_npu.npu.synchronize()

    elapsed_times = []
    for _ in range(nruns):
        dist.barrier(group=process_group)
        start_event = torch_npu.npu.Event(enable_timing=True)
        end_event = torch_npu.npu.Event(enable_timing=True)
        start_event.record()
        result = target(*benchmark_args, **kwargs)
        torch.ops._c10d_functional.wait_tensor(result)
        end_event.record()
        end_event.synchronize()
        elapsed_times.append(float(start_event.elapsed_time(end_event)))

    runtime_ms = float(statistics.median(elapsed_times))
    node_runtime_estimation.set_cached_runtime(cache_key, runtime_ms)
    return runtime_ms, cache_key


def _materialize_collective_args(
    node: torch.fx.Node,
) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Materialize generic functional-collective inputs from FakeTensor metadata."""
    from torch._dynamo.testing import rand_strided
    from torch._inductor import config, fx_utils
    from torch._inductor.fx_passes.node_runtime_estimation import get_hint
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success:
        return None

    def to_real(tensor: torch.Tensor) -> torch.Tensor:
        shape = [get_hint(dim) for dim in tensor.shape]
        stride = [get_hint(value) for value in tensor.stride()]
        if any(value is None for value in itertools.chain(shape, stride)):
            raise ValueError("cannot materialize collective with unbacked dimensions")
        concrete_shape = cast("list[int]", shape)
        concrete_stride = cast("list[int]", stride)
        return rand_strided(
            concrete_shape,
            concrete_stride,
            device=tensor.device,
            dtype=tensor.dtype,
        )

    args, kwargs = _pytree.tree_map_only(
        torch.Tensor,
        to_real,
        (args, kwargs),
    )
    args, kwargs = _pytree.tree_map_only(
        torch.SymInt,
        lambda value: optimization_hint(
            value,
            fallback=config.unbacked_symint_fallback,
        ),
        (args, kwargs),
    )
    return args, kwargs


def _a2a_profiler_preliminary_key(
    node: torch.fx.Node,
    *,
    uniform_dispatch_rows: int | None = None,
    reverse_uniform_splits: bool = False,
) -> tuple[Any, ...] | None:
    """Build a metadata-only preliminary key for an A2A collective."""
    from torch._inductor import config, fx_utils
    from torch._inductor.fx_passes import node_runtime_estimation
    from torch._inductor.fx_passes.bucketing import _resolve_group_name
    from torch.fx.operator_schemas import normalize_function

    success, args, _kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success or len(args) < 4 or not isinstance(args[0], torch.Tensor):
        return None
    normalized = normalize_function(
        call_function_target(node),
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("normalize_function returned None for all_to_all_single")
    group_name = _resolve_group_name(normalized[1]["group_name"])
    fake_input = args[0]
    shape = [node_runtime_estimation.get_hint(dim) for dim in fake_input.shape]
    if uniform_dispatch_rows is not None:
        if uniform_dispatch_rows < 0:
            raise ValueError("uniform dispatch rows must be non-negative")
        shape[0] = uniform_dispatch_rows
    elif shape[0] is None:
        shape[0] = sum(int(value) for value in args[2]) if args[2] else config.unbacked_symint_fallback
    shape = tuple(int(value) if value is not None else config.unbacked_symint_fallback for value in shape)
    stride_hints = [node_runtime_estimation.get_hint(value) for value in fake_input.stride()]
    stride = tuple(
        _contiguous_strides(shape) if any(value is None for value in stride_hints) else cast("list[int]", stride_hints)
    )
    return (
        str(node.target),
        str(group_name),
        shape,
        stride,
        str(fake_input.dtype),
        str(fake_input.device),
        reverse_uniform_splits,
    )


def _describe_a2a_profiler_benchmark(
    node: torch.fx.Node,
    *,
    uniform_dispatch_rows: int | None = None,
    reverse_uniform_splits: bool = False,
    cache_tag: str = _NPU_PROFILER_CACHE_TAG,
) -> _CollectiveBenchmarkSpec | None:
    """Describe self-consistent A2A splits without allocating a real input."""
    import torch.distributed as dist
    from torch._dynamo.testing import rand_strided
    from torch._inductor import config, fx_utils
    from torch._inductor.fx_passes import node_runtime_estimation
    from torch._inductor.fx_passes.bucketing import _resolve_group_name
    from torch.distributed.distributed_c10d import _resolve_process_group
    from torch.fx.experimental.symbolic_shapes import optimization_hint
    from torch.fx.operator_schemas import normalize_function

    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success or len(args) < 4 or not isinstance(args[0], torch.Tensor):
        return None

    normalized = normalize_function(
        call_function_target(node),
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise AssertionError("normalize_function returned None for all_to_all_single")
    group_name = _resolve_group_name(normalized[1]["group_name"])
    process_group = _resolve_process_group(group_name)
    world_size = dist.get_world_size(process_group)
    group_rank = dist.get_rank(process_group)
    args, kwargs = _pytree.tree_map_only(
        torch.SymInt,
        lambda value: optimization_hint(
            value,
            fallback=config.unbacked_symint_fallback,
        ),
        (args, kwargs),
    )
    fake_input = args[0]
    shape = [node_runtime_estimation.get_hint(dim) for dim in fake_input.shape]
    if shape[0] is None:
        shape[0] = sum(int(value) for value in args[2])
    shape = [int(value) if value is not None else config.unbacked_symint_fallback for value in shape]
    stride_hints = [node_runtime_estimation.get_hint(value) for value in fake_input.stride()]
    stride = (
        _contiguous_strides(shape) if any(value is None for value in stride_hints) else cast("list[int]", stride_hints)
    )
    if uniform_dispatch_rows is not None:
        if uniform_dispatch_rows < 0:
            raise ValueError("uniform dispatch rows must be non-negative")
        local_rows = uniform_dispatch_rows
    else:
        local_rows = int(shape[0])
    rows_per_rank: list[int | None] = [None] * world_size
    dist.all_gather_object(
        rows_per_rank,
        local_rows,
        group=process_group,
    )
    if any(rows is None for rows in rows_per_rank):
        raise AssertionError("failed to gather A2A input rows")
    split_matrix = _build_a2a_split_matrix(cast("list[int]", rows_per_rank))
    dispatch_input_splits, dispatch_output_splits = _a2a_splits_for_rank(split_matrix, group_rank)
    if reverse_uniform_splits:
        input_splits = dispatch_output_splits
        output_splits = dispatch_input_splits
    else:
        input_splits = dispatch_input_splits
        output_splits = dispatch_output_splits
    shape[0] = sum(input_splits)
    concrete_shape = shape
    concrete_stride = stride
    input_bytes = int(fake_input.element_size())
    for dim in concrete_shape:
        input_bytes *= int(dim)
    cache_key = (
        f"{node.target}: ({world_size} group size, {input_bytes} bytes, "
        f"shape={tuple(concrete_shape)}, dtype={fake_input.dtype}, "
        f"input_splits={tuple(input_splits)}, "
        f"output_splits={tuple(output_splits)}, "
        f"npu balanced splits, {cache_tag})"
    )

    def materialize():
        input_tensor = rand_strided(
            concrete_shape,
            concrete_stride,
            device=fake_input.device,
            dtype=fake_input.dtype,
        )
        return (input_tensor, output_splits, input_splits, group_name), dict(kwargs)

    return _CollectiveBenchmarkSpec(
        node=node,
        process_group=process_group,
        cache_key=cache_key,
        materialize=materialize,
    )


def _generic_collective_preliminary_key(node: torch.fx.Node) -> tuple[Any, ...] | None:
    """Build a metadata-only preliminary key for a non-A2A collective."""
    from torch._inductor import fx_utils
    from torch._inductor.fx_passes.node_runtime_estimation import get_hint

    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success:
        return None
    tensor_descriptors = []
    for value in _pytree.tree_leaves((args, kwargs)):
        if not isinstance(value, torch.Tensor):
            continue
        shape = tuple(get_hint(dim) for dim in value.shape)
        stride = tuple(get_hint(dim) for dim in value.stride())
        if any(dim is None for dim in itertools.chain(shape, stride)):
            return None
        tensor_descriptors.append((shape, stride, str(value.dtype), str(value.device)))
    return str(node.target), tuple(tensor_descriptors), repr((node.args[1:], node.kwargs))


def _collective_preliminary_key(
    node: torch.fx.Node,
    *,
    uniform_dispatch_rows: int | None = None,
    reverse_uniform_splits: bool = False,
) -> tuple[Any, ...] | None:
    if node.target is torch.ops._c10d_functional.all_to_all_single.default:
        return _a2a_profiler_preliminary_key(
            node,
            uniform_dispatch_rows=uniform_dispatch_rows,
            reverse_uniform_splits=reverse_uniform_splits,
        )
    if uniform_dispatch_rows is not None or reverse_uniform_splits:
        raise ValueError("uniform A2A options were provided for a non-A2A node")
    return _generic_collective_preliminary_key(node)


def _describe_collective_for_cann(
    node: torch.fx.Node,
    *,
    uniform_dispatch_rows: int | None = None,
    reverse_uniform_splits: bool = False,
) -> _CollectiveBenchmarkSpec | None:
    """Build a persistent CANN cache key before materializing inputs."""
    import torch.distributed as dist

    if node.target is torch.ops._c10d_functional.all_to_all_single.default:
        return _describe_a2a_profiler_benchmark(
            node,
            uniform_dispatch_rows=uniform_dispatch_rows,
            reverse_uniform_splits=reverse_uniform_splits,
            cache_tag=_NPU_CANN_CACHE_TAG,
        )
    if uniform_dispatch_rows is not None or reverse_uniform_splits:
        raise ValueError("uniform A2A options were provided for a non-A2A node")
    preliminary_key = _generic_collective_preliminary_key(node)
    if preliminary_key is None:
        return None
    process_group = resolve_collective_process_group(node)
    world_size = dist.get_world_size(process_group)
    cache_key = (
        f"{node.target}: ({world_size} group size, tensors={preliminary_key[1]}, "
        f"non_tensor_args={preliminary_key[2]}, {_NPU_CANN_CACHE_TAG})"
    )

    def materialize():
        materialized = _materialize_collective_args(node)
        if materialized is None:
            raise ValueError(f"cannot materialize collective {node.name}")
        return materialized

    return _CollectiveBenchmarkSpec(
        node=node,
        process_group=process_group,
        cache_key=cache_key,
        materialize=materialize,
    )


def _materialize_collective_spec(spec: _CollectiveBenchmarkSpec) -> _PreparedCollectiveBenchmark:
    args, kwargs = spec.materialize()
    return _PreparedCollectiveBenchmark(
        node=spec.node,
        args=tuple(args),
        kwargs=dict(kwargs),
        process_group=spec.process_group,
        cache_key=spec.cache_key,
    )


def _extract_hcom_durations_ms(
    trace: dict[str, Any] | list[Any],
    marker_names: Sequence[str],
    hcom_name_filters: Sequence[str] | None = None,
) -> dict[str, float]:
    """Associate launch markers with top-level CANN HCOM events.

    Collectives remain asynchronous inside CPU markers to avoid measuring
    rank-arrival waits, so an HCOM event may start after its marker ends. Match
    one event to each marker by issue order when counts and collective types
    agree; otherwise fall back to marker containment for synchronized traces.
    """
    events = trace.get("traceEvents", ()) if isinstance(trace, dict) else trace
    complete_events = [
        event
        for event in events
        if isinstance(event, dict) and event.get("ph") == "X" and "ts" in event and "dur" in event
    ]
    markers = {str(event.get("name")): event for event in complete_events if str(event.get("name")) in marker_names}
    if hcom_name_filters is not None and len(hcom_name_filters) != len(marker_names):
        raise ValueError("hcom_name_filters must align with marker_names")
    filter_by_marker = dict(zip(marker_names, hcom_name_filters, strict=True)) if hcom_name_filters is not None else {}
    accepted_filters = set(filter_by_marker.values())
    hcom_events = [
        event
        for event in complete_events
        if str(event.get("name", "")).lower().startswith("hcom_")
        and (
            not accepted_filters
            or any(name_filter in str(event.get("name", "")).lower() for name_filter in accepted_filters)
        )
    ]
    ordered_markers = [markers[name] for name in marker_names if name in markers]
    if len(ordered_markers) == len(marker_names) == len(hcom_events):
        ordered_markers.sort(key=lambda event: float(event["ts"]))
        hcom_events.sort(key=lambda event: float(event["ts"]))
        marker_name_by_id = {id(event): str(event["name"]) for event in ordered_markers}
        ordered_pairs = list(zip(ordered_markers, hcom_events, strict=True))
        if all(
            not filter_by_marker or filter_by_marker[marker_name_by_id[id(marker)]] in str(hcom.get("name", "")).lower()
            for marker, hcom in ordered_pairs
        ):
            return {marker_name_by_id[id(marker)]: float(hcom["dur"]) / 1000.0 for marker, hcom in ordered_pairs}

    result: dict[str, float] = {}
    for marker_name in marker_names:
        marker = markers.get(marker_name)
        if marker is None:
            continue
        marker_start = float(marker["ts"])
        marker_end = marker_start + float(marker["dur"])
        candidates = []
        for event in hcom_events:
            name_filter = filter_by_marker.get(marker_name)
            if name_filter and name_filter not in str(event.get("name", "")).lower():
                continue
            event_start = float(event["ts"])
            event_end = event_start + float(event["dur"])
            if event_start >= marker_start and event_end <= marker_end:
                candidates.append(float(event["dur"]))
        if candidates:
            # HCCL may expose nested/link activities. The longest hcom event is
            # the collective's top-level communication-stream occupancy; never
            # sum parallel links or nested children.
            result[marker_name] = max(candidates) / 1000.0
    return result


def _profile_one_collective_spec(
    spec: _CollectiveBenchmarkSpec,
    *,
    nruns: int,
    signature_index: int,
    retain_profile: bool,
) -> float:
    """Materialize, profile and release one unique collective signature."""
    import torch.distributed as dist
    from torch._inductor.fx_passes import node_runtime_estimation

    group_size = dist.get_world_size(spec.process_group)
    prepared: _PreparedCollectiveBenchmark | None = None
    materialize_error: str | None = None
    try:
        with _python_dispatch._disable_current_modes():
            prepared = _materialize_collective_spec(spec)
    except Exception as error:
        materialize_error = f"{type(error).__name__}: {error}"
    materialize_errors: list[str | None] = [None] * group_size
    dist.all_gather_object(materialize_errors, materialize_error, group=spec.process_group)
    if any(error is not None for error in materialize_errors):
        del prepared
        raise RuntimeError(f"collective materialization failed on at least one rank: {materialize_errors}")
    assert prepared is not None

    marker_names: list[str] = []
    marker_hcom_filters: list[str] = []
    measured: dict[str, float] = {}
    profile_error: str | None = None
    try:
        dist.barrier(group=spec.process_group)
        target = call_function_target(prepared.node)
        for _ in range(_CANN_WARMUP_RUNS):
            warmup_result = target(*prepared.args, **prepared.kwargs)
            torch.ops._c10d_functional.wait_tensor(warmup_result)
        torch_npu.npu.synchronize()

        with isolated_cann_profiler_work_path(
            prefix="auto_overlap_comm_bench_",
            retain=retain_profile,
        ) as work_path:
            trace_path = work_path / "trace.json"
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
                record_op_args=False,
            )
            dist.barrier(group=spec.process_group)
            torch_npu.npu.synchronize()
            target_name = str(prepared.node.target).lower().replace("_", "")
            hcom_name_filter = next(
                (kind for kind in ("alltoall", "allgather", "reducescatter", "allreduce") if kind in target_name),
                target_name,
            )
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
                issued_results = []
                for issue_index in range(nruns + 2):
                    marker_name = f"{_CANN_MARKER_PREFIX}::{signature_index}::{issue_index}"
                    marker_names.append(marker_name)
                    marker_hcom_filters.append(hcom_name_filter)
                    with torch.profiler.record_function(marker_name):
                        issued_results.append(target(*prepared.args, **prepared.kwargs))
                for result in issued_results:
                    torch.ops._c10d_functional.wait_tensor(result)
                torch_npu.npu.synchronize()
                del result
                issued_results.clear()
                del issued_results
            # The input is no longer referenced by device work. Release it
            # before exporting/parsing this signature's trace.
            del prepared
            prepared = None
            profiler.export_chrome_trace(str(trace_path))
            if not trace_path.is_file():
                raise RuntimeError("CANN profiler did not export a trace")
            with trace_path.open("r", encoding="utf-8") as trace_file:
                trace = json.load(trace_file)
            measured = _extract_hcom_durations_ms(trace, marker_names, marker_hcom_filters)
    except Exception as error:
        profile_error = f"{type(error).__name__}: {error}"
    finally:
        del prepared

    errors: list[str | None] = [None] * group_size
    dist.all_gather_object(errors, profile_error, group=spec.process_group)
    if any(error is not None for error in errors):
        raise RuntimeError(f"CANN collective profiling failed on at least one rank: {errors}")

    sample_markers = marker_names[1:-1]
    local_samples = [measured[marker] for marker in sample_markers if marker in measured]
    sample_error = None
    if len(local_samples) != nruns:
        sample_error = (
            f"rank {dist.get_rank(spec.process_group)} associated {len(local_samples)} "
            f"of {nruns} HCOM events for {spec.node.name}"
        )
    sample_errors: list[str | None] = [None] * group_size
    dist.all_gather_object(sample_errors, sample_error, group=spec.process_group)
    if any(error is not None for error in sample_errors):
        raise RuntimeError(f"CANN profiler could not associate every HCOM event: {sample_errors}")
    gathered: list[list[float] | None] = [None] * group_size
    dist.all_gather_object(gathered, local_samples, group=spec.process_group)
    if any(samples is None or len(samples) != nruns for samples in gathered):
        raise RuntimeError("CANN collective samples differ across ranks")
    concrete_samples = cast("list[list[float]]", gathered)
    global_max_samples = [max(float(samples[index]) for samples in concrete_samples) for index in range(nruns)]
    runtime_ms = float(statistics.median(global_max_samples))
    node_runtime_estimation.set_cached_runtime(spec.cache_key, runtime_ms)
    return runtime_ms


def benchmark_collectives_with_cann_profiler(
    requests: Sequence[tuple[torch.fx.Node, int | None, bool]],
    *,
    nruns: int = 5,
    retain_profile: bool = False,
) -> dict[torch.fx.Node, float]:
    """Benchmark collective requests with CANN and return per-node costs."""
    import torch.distributed as dist
    from torch._inductor.fx_passes import node_runtime_estimation

    if nruns <= 0:
        raise ValueError(f"nruns must be positive, got {nruns}")
    if not node_runtime_estimation.can_benchmark_collective():
        return {}

    # Deduplicate requests before constructing benchmark inputs.
    request_by_preliminary_key: dict[tuple[Any, ...], tuple[torch.fx.Node, int | None, bool]] = {}
    preliminary_key_by_node: dict[torch.fx.Node, tuple[Any, ...]] = {}
    for node, uniform_rows, reverse_splits in requests:
        key = _collective_preliminary_key(
            node,
            uniform_dispatch_rows=uniform_rows,
            reverse_uniform_splits=reverse_splits,
        )
        if key is None:
            continue
        preliminary_key_by_node[node] = key
        request_by_preliminary_key.setdefault(key, (node, uniform_rows, reverse_splits))

    # Reuse one benchmark spec for requests that resolve to the same workload.
    spec_by_preliminary_key: dict[tuple[Any, ...], _CollectiveBenchmarkSpec] = {}
    unique_specs: dict[str, _CollectiveBenchmarkSpec] = {}
    for key, (node, uniform_rows, reverse_splits) in request_by_preliminary_key.items():
        spec = _describe_collective_for_cann(
            node,
            uniform_dispatch_rows=uniform_rows,
            reverse_uniform_splits=reverse_splits,
        )
        if spec is None:
            continue
        spec_by_preliminary_key[key] = spec
        unique_specs.setdefault(spec.cache_key, spec)

    costs_by_key: dict[str, float] = {}
    missing: list[_CollectiveBenchmarkSpec] = []
    for spec in unique_specs.values():
        cached = node_runtime_estimation.get_cached_runtime(spec.cache_key)
        group_size = dist.get_world_size(spec.process_group)
        cache_hits: list[bool | None] = [None] * group_size
        dist.all_gather_object(cache_hits, cached is not None, group=spec.process_group)
        if all(cache_hits):
            if cached is None:
                raise AssertionError("collective cache agreement is inconsistent")
            costs_by_key[spec.cache_key] = float(cached)
        else:
            missing.append(spec)

    for signature_index, spec in enumerate(missing):
        costs_by_key[spec.cache_key] = _profile_one_collective_spec(
            spec,
            nruns=nruns,
            signature_index=signature_index,
            retain_profile=retain_profile,
        )

    result = {}
    for node, preliminary_key in preliminary_key_by_node.items():
        spec = spec_by_preliminary_key.get(preliminary_key)
        if spec is not None and spec.cache_key in costs_by_key:
            result[node] = costs_by_key[spec.cache_key]
    return result


def benchmark_collective_with_npu_events(
    node: torch.fx.Node,
    nruns: int = 2,
    *,
    uniform_dispatch_rows: int | None = None,
    reverse_uniform_splits: bool = False,
    generic_benchmark: Callable[[torch.fx.Node, int], tuple[float | None, str]] | None = None,
) -> tuple[float | None, str]:
    """Benchmark a collective through completion using device events.

    Resolve dynamic A2A rows from the assumed token count and construct
    balanced, cross-rank-consistent splits. Other collectives use PyTorch's
    generic event benchmark.
    """
    from torch._inductor.fx_passes import node_runtime_estimation

    with _python_dispatch._disable_current_modes():
        if node.target is torch.ops._c10d_functional.all_to_all_single.default:
            return _benchmark_a2a_with_npu_events(
                node,
                nruns,
                uniform_dispatch_rows=uniform_dispatch_rows,
                reverse_uniform_splits=reverse_uniform_splits,
            )
        if uniform_dispatch_rows is not None or reverse_uniform_splits:
            raise ValueError("uniform A2A options were provided for a non-A2A node")
        benchmark = generic_benchmark
        if benchmark is None:
            benchmark = node_runtime_estimation.benchmark_collective_with_cuda_events
        return benchmark(node, nruns)
