# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU runtime benchmarks for compute operations."""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.utils import _pytree

from .utils import call_function_target

if TYPE_CHECKING:
    from collections.abc import Callable

_GROUPED_MM_TARGET = "aten._grouped_mm.default"
_NPU_PROFILER_CACHE_TAG = "torch_profiler_npu"
PERMUTATION_TARGETS = {
    "npu.npu_moe_token_permute.default",
    "npu.npu_moe_token_unpermute.default",
    "torchtitan_npu.npu_moe_token_unpermute.default",
}


def make_npu_do_bench() -> Callable[[Callable[[], Any]], float]:
    """Create the configured NPU callable benchmark backend."""
    from torch._inductor.runtime.benchmarking import TorchProfilerBenchmarker

    benchmarker = TorchProfilerBenchmarker()
    return functools.partial(
        benchmarker.benchmark_gpu,
        rep=5,
        estimation_iters=2,
        memory_warmup_iters=1,
        device_type="npu",
    )


@dataclass(frozen=True)
class _GroupedMmBenchmarkSpec:
    tokens: int
    experts: int
    counts: tuple[int, ...]
    boundaries: tuple[int, ...]


def _balanced_group_boundaries(
    tokens: int,
    experts: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build legal cumulative GMM offsets."""
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative, got {tokens}")
    if experts <= 0:
        raise ValueError(f"experts must be positive, got {experts}")
    base, remainder = divmod(tokens, experts)
    counts = tuple(base + int(index < remainder) for index in range(experts))
    boundaries = tuple(itertools.accumulate(counts))
    assert boundaries[-1] == tokens
    return counts, boundaries


def _hint_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    hint = getattr(value, "hint", None)
    return int(hint) if hint is not None else None


def _grouped_mm_benchmark_spec(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    offsets: torch.Tensor,
    *,
    assumed_tokens: int | None = None,
) -> _GroupedMmBenchmarkSpec | None:
    """Infer token/expert dimensions for forward/dgrad and wgrad GMM."""
    lhs_shape = tuple(_hint_int(dim) for dim in lhs.shape)
    rhs_shape = tuple(_hint_int(dim) for dim in rhs.shape)
    offsets_shape = tuple(_hint_int(dim) for dim in offsets.shape)
    if len(offsets_shape) != 1 or offsets_shape[0] is None:
        return None
    experts = offsets_shape[0]
    if experts <= 0:
        return None

    if len(lhs_shape) == 2 and len(rhs_shape) == 3:
        # [tokens, K] @ [experts, K, N]
        if any(dim is None for dim in (*lhs_shape[1:], *rhs_shape)):
            return None
        tokens = lhs_shape[0]
        if rhs_shape[0] != experts or lhs_shape[1] != rhs_shape[1]:
            return None
    elif len(lhs_shape) == 2 and len(rhs_shape) == 2:
        # Weight gradient: [K, tokens] @ [tokens, N].
        if lhs_shape[0] is None or rhs_shape[1] is None:
            return None
        lhs_tokens, rhs_tokens = lhs_shape[1], rhs_shape[0]
        if lhs_tokens is not None and rhs_tokens is not None and lhs_tokens != rhs_tokens:
            return None
        tokens = lhs_tokens if lhs_tokens is not None else rhs_tokens
    else:
        return None

    if tokens is None:
        tokens = assumed_tokens
    if tokens is None or tokens < 0:
        return None

    counts, boundaries = _balanced_group_boundaries(tokens, experts)
    return _GroupedMmBenchmarkSpec(tokens, experts, counts, boundaries)


def _tensor_descriptor(tensor: torch.Tensor) -> tuple:
    shape = tuple(_hint_int(dim) for dim in tensor.shape)
    stride = tuple(_hint_int(dim) for dim in tensor.stride())
    return shape, stride, str(tensor.dtype), str(tensor.device)


def benchmark_npu_grouped_mm_node(
    node: torch.fx.Node,
    do_bench: Callable[[Callable[[], Any]], float],
    *,
    cache_tag: str = "",
    assumed_tokens: int | None = None,
) -> tuple[float, str] | None:
    """Benchmark ``aten._grouped_mm`` using balanced synthetic group offsets.

    Resolve dynamic routed rows from the assumed token total and construct
    legal group offsets.
    """
    if node.op != "call_function" or str(node.target) != _GROUPED_MM_TARGET:
        return None

    from torch._dynamo.testing import rand_strided
    from torch._inductor import fx_utils
    from torch._inductor.fx_passes import overlap_scheduling

    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success:
        return None
    lhs = args[0] if len(args) > 0 else kwargs.get("self")
    rhs = args[1] if len(args) > 1 else kwargs.get("mat2")
    offsets = args[2] if len(args) > 2 else kwargs.get("offs")
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor) or not isinstance(offsets, torch.Tensor):
        return None
    spec = _grouped_mm_benchmark_spec(
        lhs,
        rhs,
        offsets,
        assumed_tokens=assumed_tokens,
    )
    if spec is None:
        return None
    spec_tokens = spec.tokens

    is_batched_weight = rhs.ndim == 3
    lhs_token_axis = 0 if is_batched_weight else 1
    rhs_token_axis = None if is_batched_weight else 0

    def resolved_descriptor(
        tensor: torch.Tensor,
        token_axis: int | None = None,
    ) -> tuple | None:
        shape = []
        for axis, dim in enumerate(tensor.shape):
            value = _hint_int(dim)
            if value is None and axis == token_axis:
                value = spec_tokens
            if value is None:
                return None
            shape.append(int(value))
        stride = tuple(_hint_int(dim) for dim in tensor.stride())
        if any(dim is None for dim in stride):
            return None
        concrete_stride = cast("tuple[int, ...]", stride)
        return (
            tuple(shape),
            concrete_stride,
            str(tensor.dtype),
            str(tensor.device),
        )

    lhs_descriptor = resolved_descriptor(lhs, lhs_token_axis)
    rhs_descriptor = resolved_descriptor(rhs, rhs_token_axis)
    offsets_descriptor = resolved_descriptor(offsets)
    if any(descriptor is None for descriptor in (lhs_descriptor, rhs_descriptor, offsets_descriptor)):
        return None
    key = (
        f"{_GROUPED_MM_TARGET}: tensors="
        f"{(lhs_descriptor, rhs_descriptor, offsets_descriptor)} "
        f"tokens={spec.tokens} experts={spec.experts} "
        f"distribution={spec.counts} synthetic=balanced "
        f"assumed_tokens={assumed_tokens} "
        f"benchmarker={cache_tag or 'unspecified'}"
    )
    if cached := overlap_scheduling.get_cached_node_time(key):
        return float(cached), key
    if spec.tokens == 0:
        overlap_scheduling.set_cached_node_time(key, 0.0)
        return 0.0, key

    def materialize(
        tensor: torch.Tensor,
        token_axis: int | None = None,
    ) -> torch.Tensor:
        descriptor = resolved_descriptor(tensor, token_axis)
        if descriptor is None:
            raise ValueError("GMM tensor has unresolved non-token metadata")
        shape, stride, _, _ = descriptor
        return rand_strided(shape, stride, device=tensor.device, dtype=tensor.dtype)

    with overlap_scheduling._disable_current_modes():
        real_args = list(args)
        for index in range(2, len(real_args)):
            if isinstance(real_args[index], torch.Tensor):
                real_args[index] = materialize(real_args[index])
        real_kwargs = {
            key: (
                value
                if key in {"self", "mat2", "offs"}
                else _pytree.tree_map_only(
                    torch.Tensor,
                    materialize,
                    value,
                )
            )
            for key, value in kwargs.items()
        }
        real_lhs = materialize(lhs, lhs_token_axis)
        real_rhs = materialize(rhs, rhs_token_axis)
        if len(real_args) > 0:
            real_args[0] = real_lhs
        else:
            real_kwargs["self"] = real_lhs
        if len(real_args) > 1:
            real_args[1] = real_rhs
        else:
            real_kwargs["mat2"] = real_rhs
        real_offsets = torch.tensor(
            spec.boundaries,
            device=offsets.device,
            dtype=offsets.dtype,
        )
        if len(real_args) > 2:
            real_args[2] = real_offsets
        else:
            real_kwargs["offs"] = real_offsets
        target = call_function_target(node)
        elapsed_ms = float(do_bench(lambda: target(*real_args, **real_kwargs)))
    overlap_scheduling.set_cached_node_time(key, elapsed_ms)
    return elapsed_ms, key


def benchmark_npu_compute_node(
    node: torch.fx.Node,
    do_bench: Callable[[Callable[[], Any]], float],
    *,
    cache_tag: str = "",
) -> tuple[float, str] | None:
    """Benchmark a compute node using randomly materialized tensor inputs."""
    from torch._dynamo.testing import rand_strided
    from torch._inductor import config, fx_utils
    from torch._inductor.fx_passes import overlap_scheduling
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success:
        return None

    def descriptor(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return ("tensor", *_tensor_descriptor(value))
        if isinstance(value, torch.SymInt):
            return (
                "symint",
                optimization_hint(
                    value,
                    fallback=config.unbacked_symint_fallback,
                ),
            )
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        return (type(value).__qualname__, repr(value))

    key_tree = _pytree.tree_map(descriptor, (args, kwargs))
    key = f"{node.target}: args_kwargs={key_tree!r} materializer=random benchmarker={cache_tag or 'unspecified'}"
    cached = overlap_scheduling.get_cached_node_time(key)
    if cached is not None:
        return float(cached), key

    def materialize(tensor: torch.Tensor) -> torch.Tensor:
        shape = tuple(_hint_int(dim) for dim in tensor.shape)
        stride = tuple(_hint_int(dim) for dim in tensor.stride())
        if any(dim is None for dim in (*shape, *stride)):
            raise ValueError("compute tensor has unresolved shape or stride")
        return rand_strided(
            cast("tuple[int, ...]", shape),
            cast("tuple[int, ...]", stride),
            device=tensor.device,
            dtype=tensor.dtype,
        )

    with overlap_scheduling._disable_current_modes():
        real_args, real_kwargs = _pytree.tree_map_only(
            torch.Tensor,
            materialize,
            (args, kwargs),
        )
        real_args, real_kwargs = _pytree.tree_map_only(
            torch.SymInt,
            lambda value: optimization_hint(
                value,
                fallback=config.unbacked_symint_fallback,
            ),
            (real_args, real_kwargs),
        )
        target = call_function_target(node)
        elapsed_ms = float(do_bench(lambda: target(*real_args, **real_kwargs)))
    overlap_scheduling.set_cached_node_time(key, elapsed_ms)
    return elapsed_ms, key


@dataclass(frozen=True)
class _PermutationBenchmarkSpec:
    is_permute: bool
    token_name: str
    index_name: str
    tokens: torch.Tensor
    indices: torch.Tensor
    probs: torch.Tensor | None
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    index_shape: tuple[int, ...]
    index_stride: tuple[int, ...]
    prob_shape: tuple[int, ...] | None
    prob_stride: tuple[int, ...]
    source_tokens: int
    routed_tokens: int
    topk: int


def balanced_routing_indices(tokens, topk, experts, *, device, dtype):
    """Balance assignments to within one token and avoid duplicate top-k IDs."""
    if experts <= 0 or not 0 < topk <= experts or tokens <= 0:
        raise ValueError("balanced routing requires positive tokens and 0 < topk <= experts")
    return (torch.arange(tokens * topk, device=device, dtype=dtype) % experts).reshape(tokens, topk)


def _resolved_shape(tensor: torch.Tensor, rows: int) -> tuple[int, ...] | None:
    shape = [_hint_int(dim) for dim in tensor.shape]
    if shape and shape[0] is None:
        shape[0] = rows
    if not shape or any(dim is None or dim <= 0 for dim in shape):
        return None
    return cast("tuple[int, ...]", tuple(shape))


def _permutation_benchmark_spec(
    target: str,
    bound: dict[str, Any],
    *,
    routed_tokens: int,
    experts: int,
) -> _PermutationBenchmarkSpec | None:
    """Resolve and validate one supported permutation workload."""
    if bound.get("padded_mode", False) or bound.get("restore_shape") is not None:
        return None

    is_permute = target == "npu.npu_moe_token_permute.default"
    token_name = "tokens" if is_permute else "permuted_tokens"
    index_name = "indices" if is_permute else "sorted_indices"
    tokens = bound[token_name]
    indices = bound[index_name]
    if not isinstance(tokens, torch.Tensor) or not isinstance(indices, torch.Tensor):
        return None

    stride = tuple(_hint_int(dim) for dim in tokens.stride())
    index_stride = tuple(_hint_int(dim) for dim in indices.stride())
    if any(dim is None for dim in (*stride, *index_stride)):
        return None

    probs = bound.get("probs")
    if probs is not None and not isinstance(probs, torch.Tensor):
        return None
    prob_shape = None
    topk = 1
    if is_permute:
        if tokens.ndim != 2 or indices.ndim != 2 or (topk := _hint_int(indices.shape[1])) is None or topk <= 0:
            return None
        source_hints = {hint for hint in (_hint_int(tokens.shape[0]), _hint_int(indices.shape[0])) if hint is not None}
        if len(source_hints) > 1:
            return None
        if source_hints:
            source_tokens = next(iter(source_hints))
        else:
            source_tokens, remainder = divmod(routed_tokens, topk)
            if remainder:
                return None
        resolved_routed_tokens = source_tokens * topk
        if resolved_routed_tokens != routed_tokens:
            return None
        shape = _resolved_shape(tokens, source_tokens)
        index_shape = _resolved_shape(indices, source_tokens)
        if shape is None or index_shape is None or len(shape) != 2:
            return None
        if len(index_shape) != 2 or index_shape[0] != shape[0]:
            return None
        num_out = bound.get("num_out_tokens")
        if num_out is not None and _hint_int(num_out) not in (0, resolved_routed_tokens):
            return None
    else:
        if tokens.ndim != 2 or indices.ndim != 1:
            return None
        routed_hints = {hint for hint in (_hint_int(tokens.shape[0]), _hint_int(indices.shape[0])) if hint is not None}
        if len(routed_hints) > 1:
            return None
        resolved_routed_tokens = next(iter(routed_hints), routed_tokens)
        if resolved_routed_tokens != routed_tokens:
            return None
        shape = _resolved_shape(tokens, resolved_routed_tokens)
        index_shape = _resolved_shape(indices, resolved_routed_tokens)
        if shape is None or index_shape is None or len(shape) != 2:
            return None
        if len(index_shape) != 1 or index_shape[0] != shape[0]:
            return None
        if probs is not None:
            if probs.ndim != 2 or (topk := _hint_int(probs.shape[1])) is None or topk <= 0:
                return None
            source_hint = _hint_int(probs.shape[0])
            if source_hint is None:
                source_tokens, remainder = divmod(resolved_routed_tokens, topk)
                if remainder:
                    return None
            else:
                source_tokens = source_hint
            prob_shape = _resolved_shape(probs, source_tokens)
            if prob_shape is None or len(prob_shape) != 2 or math.prod(prob_shape) != shape[0]:
                return None
        else:
            source_tokens = resolved_routed_tokens
    if not 0 < topk <= experts:
        return None

    prob_stride = tuple(_hint_int(dim) for dim in probs.stride()) if probs is not None else ()
    if any(dim is None for dim in prob_stride):
        return None
    concrete_stride = cast("tuple[int, ...]", stride)
    concrete_index_stride = cast("tuple[int, ...]", index_stride)
    concrete_prob_stride = cast("tuple[int, ...]", prob_stride)
    return _PermutationBenchmarkSpec(
        is_permute=is_permute,
        token_name=token_name,
        index_name=index_name,
        tokens=tokens,
        indices=indices,
        probs=probs,
        shape=shape,
        stride=concrete_stride,
        index_shape=index_shape,
        index_stride=concrete_index_stride,
        prob_shape=prob_shape,
        prob_stride=concrete_prob_stride,
        source_tokens=source_tokens,
        routed_tokens=resolved_routed_tokens,
        topk=topk,
    )


def benchmark_npu_permutation_node(
    node: torch.fx.Node,
    do_bench: Callable[[Callable[[], Any]], float],
    *,
    routed_tokens: int,
    experts: int,
    cache_tag: str = "",
) -> tuple[float, str] | None:
    """Benchmark an NPU MoE permutation using balanced synthetic routing.

    Resolve dynamic source and routed rows from the routed-token workload and
    construct legal routing indices. Padded routing, truncated permute output
    and explicit unpermute restore shapes are not materialized.
    """
    from torch._dynamo.testing import rand_strided
    from torch._inductor import fx_utils
    from torch._inductor.fx_passes import overlap_scheduling

    if str(node.target) not in PERMUTATION_TARGETS:
        return None
    success, args, kwargs = fx_utils.get_fake_args_kwargs(node)
    if not success:
        return None
    target = call_function_target(node)
    schema = getattr(target, "_schema", None)
    if schema is None:
        raise AssertionError(f"permutation target has no operator schema: {target}")
    arguments = list(schema.arguments)
    bound = {arg.name: arg.default_value for arg in arguments}
    bound.update({arg.name: value for arg, value in zip(arguments, args, strict=False)})
    bound.update(kwargs)
    spec = _permutation_benchmark_spec(
        str(node.target),
        bound,
        routed_tokens=routed_tokens,
        experts=experts,
    )
    if spec is None:
        return None
    key = (
        f"{node.target}: balanced_permutation shape={spec.shape} stride={spec.stride} "
        f"dtype={spec.tokens.dtype} device={spec.tokens.device} index_dtype={spec.indices.dtype} "
        f"index_shape={spec.index_shape} index_stride={spec.index_stride} "
        f"probs={spec.prob_shape} prob_stride={spec.prob_stride} "
        f"prob_dtype={getattr(spec.probs, 'dtype', None)} experts={experts} "
        f"source_tokens={spec.source_tokens} routed_tokens={spec.routed_tokens} "
        f"num_out={bound.get('num_out_tokens')} benchmarker={cache_tag}"
    )
    cached = overlap_scheduling.get_cached_node_time(key)
    if cached is not None:
        return float(cached), key

    with overlap_scheduling._disable_current_modes():
        bound[spec.token_name] = rand_strided(
            spec.shape,
            spec.stride,
            device=spec.tokens.device,
            dtype=spec.tokens.dtype,
        )
        routing = balanced_routing_indices(
            spec.source_tokens,
            spec.topk,
            experts,
            device=spec.tokens.device,
            dtype=torch.int32,
        )
        if spec.is_permute:
            bound[spec.index_name] = torch.empty_strided(
                spec.index_shape,
                spec.index_stride,
                device=spec.indices.device,
                dtype=spec.indices.dtype,
            ).copy_(routing)
            if bound.get("num_out_tokens") is not None:
                num_out_tokens = _hint_int(bound["num_out_tokens"])
                if num_out_tokens is None:
                    raise ValueError("num_out_tokens must be concrete")
                bound["num_out_tokens"] = num_out_tokens
        else:
            # Unpermute consumes the inverse indices returned by NPU permute,
            # not the expert-sorting order itself. Generate valid indices with
            # a width-one dummy tensor because feature width does not affect
            # the permutation.
            dummy = torch.zeros(
                (spec.source_tokens, 1),
                device=spec.tokens.device,
                dtype=spec.tokens.dtype,
            )
            _, inverse = torch.ops.npu.npu_moe_token_permute.default(dummy, routing)
            bound[spec.index_name] = torch.empty_strided(
                spec.index_shape,
                spec.index_stride,
                device=spec.indices.device,
                dtype=spec.indices.dtype,
            ).copy_(inverse)
            if spec.probs is not None:
                assert spec.prob_shape is not None
                bound["probs"] = torch.empty_strided(
                    spec.prob_shape,
                    spec.prob_stride,
                    device=spec.probs.device,
                    dtype=spec.probs.dtype,
                ).fill_(1.0 / spec.topk)
        real_args = [bound[arg.name] for arg in arguments if not arg.kwarg_only]
        real_kwargs = {arg.name: bound[arg.name] for arg in arguments if arg.kwarg_only}
        elapsed = float(do_bench(lambda: target(*real_args, **real_kwargs)))
    overlap_scheduling.set_cached_node_time(key, elapsed)
    return elapsed, key


class ComputeBenchmarker:
    """Benchmark supported NPU compute nodes with one lazy timing backend."""

    def __init__(self) -> None:
        self._do_bench: Callable[[Callable[[], Any]], float] | None = None

    def benchmark(
        self,
        node: torch.fx.Node,
        *,
        assumed_tokens: int | None = None,
        permutation_assumption: tuple[int, int] | None = None,
    ) -> tuple[float, str] | None:
        if self._do_bench is None:
            self._do_bench = make_npu_do_bench()

        target = str(node.target)
        if target in PERMUTATION_TARGETS:
            if permutation_assumption is None:
                return None
            routed_tokens, experts = permutation_assumption
            return benchmark_npu_permutation_node(
                node,
                self._do_bench,
                routed_tokens=routed_tokens,
                experts=experts,
                cache_tag=_NPU_PROFILER_CACHE_TAG,
            )
        if target == _GROUPED_MM_TARGET:
            return benchmark_npu_grouped_mm_node(
                node,
                self._do_bench,
                cache_tag=_NPU_PROFILER_CACHE_TAG,
                assumed_tokens=assumed_tokens,
            )
        return benchmark_npu_compute_node(
            node,
            self._do_bench,
            cache_tag=_NPU_PROFILER_CACHE_TAG,
        )


def _is_concrete_tensor(tensor: torch.Tensor) -> bool:
    """Return whether a fake tensor can be materialized for benchmarking."""
    for value in (*tensor.shape, *tensor.stride()):
        if isinstance(value, int):
            continue
        if getattr(value, "hint", None) is None:
            return False
    return True


def _tensor_inputs(node: torch.fx.Node) -> tuple[torch.Tensor, ...]:
    """Return FakeTensor inputs referenced by an FX node."""
    tensors: list[torch.Tensor] = []
    flat_args, _ = _pytree.tree_flatten((node.args, node.kwargs))
    for arg in flat_args:
        if not isinstance(arg, torch.fx.Node):
            continue
        flat_values, _ = _pytree.tree_flatten(arg.meta.get("val"))
        tensors.extend(value for value in flat_values if isinstance(value, torch.Tensor))
    return tuple(tensors)


def _has_concrete_tensor_inputs(node: torch.fx.Node) -> bool:
    """Check all tensor inputs used by ``node`` have benchmarkable metadata."""
    tensors = _tensor_inputs(node)
    return bool(tensors) and all(_is_concrete_tensor(tensor) for tensor in tensors)


def _has_value_sensitive_tensor_inputs(node: torch.fx.Node) -> bool:
    """Reject random materialization when tensor values constrain legality.

    Integer/bool tensors commonly carry indices, masks, offsets, or split
    metadata. Randomizing them may lead to invalid operations.
    """
    return any(not (tensor.dtype.is_floating_point or tensor.dtype.is_complex) for tensor in _tensor_inputs(node))


def _is_npu_benchmark_compute_node(node: torch.fx.Node) -> bool:
    """Identify pure nodes that are safe for generic NPU benchmarking."""
    if node.op != "call_function" or node.is_impure():
        return False
    if isinstance(node.target, torch._ops.OpOverload) and node.target.namespace == "_c10d_functional":
        return False
    if str(node.target) == _GROUPED_MM_TARGET:
        return _has_concrete_tensor_inputs(node)
    tensor_inputs = _tensor_inputs(node)
    return (
        bool(tensor_inputs)
        and any(tensor.device.type in {"npu", "meta"} for tensor in tensor_inputs)
        and _has_concrete_tensor_inputs(node)
        and not _has_value_sensitive_tensor_inputs(node)
    )
