# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4529
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keep independent DSV4 metadata out of EP chunk shape handling.

Own EP chunk symbols through the torchtitan_chunk_batch/seq shape ID,
preserve independent packed/varlen symbols during body copying and final
concretization, and use chunk-local ranges for copied unbacked symbols.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import sympy
import torch
import torchtitan.experiments.graph_trainer.ep_chunk_pass
import torchtitan.experiments.graph_trainer.ep_pass_utils
import torchtitan.experiments.graph_trainer.passes
import torchtitan.experiments.graph_trainer.registry
from torch.utils._sympy.numbers import int_oo
from torch.utils._sympy.value_ranges import ValueRanges
from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    import torch.fx as fx
    from torchtitan.experiments.graph_trainer.configs import EpOverlapChunkDim


def _owned_chunk_symbol_hints(gm: fx.GraphModule, mode: str) -> dict[object, int]:
    """Collect symbols only from dimensions carrying the EP chunk shape ID."""
    dim = {"batch": 0, "seq": 1}.get(mode)
    if dim is None:
        raise ValueError(f"Unknown chunk mode: {mode!r}")
    expected_shape_id = f"torchtitan_chunk_{mode}"
    hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        val = torchtitan.experiments.graph_trainer.ep_pass_utils.tensor_meta(node)
        if node.op != "placeholder" or val is None or dim >= len(val.shape):
            continue
        shape_ids = getattr(val, "_dynamo_shape_ids", None)
        if not isinstance(shape_ids, dict) or shape_ids.get(dim) != expected_shape_id:
            continue
        torchtitan.experiments.graph_trainer.ep_pass_utils._record_symbols_from_extent(
            hints,
            val.shape[dim],
            source=f"{node.name}.shape[{dim}]",
        )
    return hints


def _restore_placeholder_shape_ids(gm: fx.GraphModule, example_inputs: tuple[Any, ...] | None) -> None:
    """Restore input shape IDs that make_fx does not retain on placeholders."""
    if example_inputs is None:
        raise ValueError("populate_chunk_dim_metadata_pass requires example inputs.")
    placeholders = [node for node in gm.graph.nodes if node.op == "placeholder"]
    if len(placeholders) != len(example_inputs):
        raise ValueError(
            "EP chunk placeholder/example-input mismatch: "
            f"{len(placeholders)} placeholders vs {len(example_inputs)} inputs."
        )
    for node, example_input in zip(placeholders, example_inputs, strict=True):
        val = torchtitan.experiments.graph_trainer.ep_pass_utils.tensor_meta(node)
        if not isinstance(val, torch.Tensor) or not isinstance(example_input, torch.Tensor):
            continue
        shape_ids = getattr(example_input, "_dynamo_shape_ids", None)
        if isinstance(shape_ids, dict):
            setattr(val, "_dynamo_shape_ids", dict(shape_ids))  # noqa: B010


def _drop_attention_masks(value: object) -> object:
    if not isinstance(value, dict) or "attention_masks" not in value:
        return value
    copied = dict(value)
    copied.pop("attention_masks")
    return copied


def _is_infinite_bound(bound: Any) -> bool:
    return (
        getattr(bound, "is_finite", None) is False
        or getattr(bound, "is_infinite", None) is True
        or bound in (sympy.oo, int_oo)
    )


def _half_size_bound(bound: Any) -> int | sympy.Expr:
    if _is_infinite_bound(bound):
        return int_oo
    return max(1, (int(bound) + 1) // 2)


def apply() -> None:
    ep_chunk_pass = torchtitan.experiments.graph_trainer.ep_chunk_pass
    ep_pass_utils = torchtitan.experiments.graph_trainer.ep_pass_utils
    graph_passes = torchtitan.experiments.graph_trainer.passes
    registry = torchtitan.experiments.graph_trainer.registry

    required_helpers = (
        "_concretize_value",
        "_placeholder_symbol_hints",
        "_record_symbols_from_extent",
        "chunk_symbol_hints_for_mode",
        "free_symbols",
        "tensor_meta",
    )
    if not all(hasattr(ep_pass_utils, name) for name in required_helpers):
        logger.warning("Skipping GraphTrainer EP chunk shape patch: required symbolic-shape helpers are unavailable.")
        return
    if getattr(
        ep_chunk_pass.populate_chunk_dim_metadata_pass,
        "npu_owns_ep_chunk_symbols",
        False,
    ):
        return

    original_symbol_hints = ep_pass_utils.chunk_symbol_hints_for_mode
    original_populate = ep_chunk_pass.populate_chunk_dim_metadata_pass
    original_mark_dynamic = ep_chunk_pass.mark_chunk_dynamic_dims
    original_prepare_inputs = ep_chunk_pass.prepare_ep_overlap_trace_inputs
    original_fresh_symbol = ep_chunk_pass._fresh_unbacked_symbol_for_copy
    original_placeholder_hints = ep_pass_utils._placeholder_symbol_hints
    original_concretize_value = ep_pass_utils._concretize_value

    @wraps(original_symbol_hints)
    def chunk_symbol_hints_for_mode(gm: fx.GraphModule, mode: str | None = None) -> dict[object, int]:
        explicit = original_symbol_hints(gm)
        if explicit or mode is None:
            return explicit
        owned = _owned_chunk_symbol_hints(gm, mode)
        # Preserve the upstream fallback for hand-built test graphs that call
        # the chunk pass directly without the populate pass.
        return owned or original_symbol_hints(gm, mode)

    @wraps(original_populate)
    def populate_chunk_dim_metadata_pass(
        gm: fx.GraphModule,
        example_inputs: tuple[Any, ...] | None = None,
        *,
        mode: EpOverlapChunkDim,
    ) -> fx.GraphModule:
        _restore_placeholder_shape_ids(gm, example_inputs)
        if not _owned_chunk_symbol_hints(gm, mode):
            raise ValueError(
                "ep_overlap graph chunking expected at least one selected "
                "dynamic placeholder dimension, but none was found."
            )
        return original_populate(gm, example_inputs, mode=mode)

    @wraps(original_mark_dynamic)
    def mark_chunk_dynamic_dims(tensor: torch.Tensor, *, mode: EpOverlapChunkDim) -> None:
        from torch._dynamo.decorators import mark_unbacked

        dim = {"batch": 0, "seq": 1}.get(mode)
        if dim is None:
            raise ValueError(f"Unknown chunk mode: {mode!r}")
        if tensor.dim() <= dim:
            raise ValueError(f"Cannot mark {mode} dim {dim} for shape {tuple(tensor.shape)}.")
        extent = int(tensor.shape[dim])
        if extent < 2 or extent % 2:
            raise ValueError(
                "EP overlap graph chunking requires an even selected dimension, "
                f"got {mode} size {extent} for shape {tuple(tensor.shape)}."
            )
        lower, upper = extent // 2, extent
        if mode == "batch":
            lower = max(2, lower)
            upper = max(upper, lower + 1)
        mark_unbacked(
            tensor,
            dim,
            hint_override=extent,
            min=lower,
            max=upper,
            specialize_on=[lambda value, extent=extent: value == extent],
            shape_id=f"torchtitan_chunk_{mode}",
        )

    @wraps(original_prepare_inputs)
    def prepare_ep_overlap_trace_inputs(
        compile_config: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        filtered_args = tuple(_drop_attention_masks(value) for value in args)
        filtered_kwargs = _drop_attention_masks(kwargs)
        assert isinstance(filtered_kwargs, dict)
        original_prepare_inputs(compile_config, filtered_args, filtered_kwargs)

    @wraps(original_fresh_symbol)
    def fresh_unbacked_symbol_for_copy(
        shape_env: Any,
        symbol: sympy.Symbol,
        symbol_hints: dict[object, int],
    ) -> sympy.Symbol:
        new_symbol = original_fresh_symbol(shape_env, symbol, symbol_hints)
        if symbol not in shape_env.var_to_range:
            return new_symbol
        if not ep_chunk_pass._symbol_has_selected_runtime_assert(shape_env, symbol, symbol_hints):
            return new_symbol
        original_range = shape_env.var_to_range[symbol]
        shape_env.var_to_range[new_symbol] = ValueRanges(
            lower=_half_size_bound(original_range.lower),
            upper=_half_size_bound(original_range.upper),
        )
        return new_symbol

    @wraps(original_concretize_value)
    def concretize_chunk_value(value: object, symbol_hints: dict[object, int]) -> object:
        if not ep_pass_utils.free_symbols(value) & symbol_hints.keys():
            return value
        return original_concretize_value(value, symbol_hints)

    @wraps(original_placeholder_hints)
    def chunk_placeholder_symbol_hints(gm: fx.GraphModule) -> dict[object, int]:
        chunk_symbols = chunk_symbol_hints_for_mode(gm).keys()
        hints: dict[object, int] = {}
        for node in gm.graph.nodes:
            val = ep_pass_utils.tensor_meta(node)
            if node.op != "placeholder" or val is None:
                continue
            for dim, extent in enumerate(val.shape):
                if not ep_pass_utils.free_symbols(extent) & chunk_symbols:
                    continue
                ep_pass_utils._record_symbols_from_extent(
                    hints,
                    extent,
                    source=f"{node.name}.shape[{dim}]",
                )
        return hints

    # pyrefly: ignore [missing-attribute]
    populate_chunk_dim_metadata_pass.npu_owns_ep_chunk_symbols = True
    ep_pass_utils.chunk_symbol_hints_for_mode = chunk_symbol_hints_for_mode
    ep_pass_utils._placeholder_symbol_hints = chunk_placeholder_symbol_hints
    ep_pass_utils._concretize_value = concretize_chunk_value
    ep_chunk_pass.chunk_symbol_hints_for_mode = chunk_symbol_hints_for_mode
    ep_chunk_pass.populate_chunk_dim_metadata_pass = populate_chunk_dim_metadata_pass
    ep_chunk_pass.mark_chunk_dynamic_dims = mark_chunk_dynamic_dims
    ep_chunk_pass.prepare_ep_overlap_trace_inputs = prepare_ep_overlap_trace_inputs
    ep_chunk_pass._fresh_unbacked_symbol_for_copy = fresh_unbacked_symbol_for_copy
    graph_passes.populate_chunk_dim_metadata_pass = populate_chunk_dim_metadata_pass
    registry.TRACE_INPUT_PREPARERS["ep_overlap"] = prepare_ep_overlap_trace_inputs
    logger.info("Enabled GraphTrainer EP chunk dynamic-shape patch for packed metadata")


apply()
