# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Symbol analysis owned by the CV graph-chunk experiment.

These helpers retain the real graph-chunk metadata algorithm from TorchTitan
c04d1bd3b6fd9f6f829bca6be148d9872a563c7a, ep_pass_utils.py. Upstream removed
that producer and these helpers in 43f73649f195a7790752daf11548d5523e9306b3.
They do not patch or recreate upstream modules. In particular, the folded
batch/sequence input protocol intentionally examines leading dimension zero.
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.fx as fx

CHUNK_SYMBOL_HINTS_META = "torchtitan_chunk_symbol_hints"


def require_graph_chunk_runtime() -> None:
    """Fail before touching runtime state until CV owns a new chunk producer."""
    raise NotImplementedError(
        "CV parallel runtime migration is pending: this TorchTitan target "
        "removed graph-based EP chunking and its trainer hooks. "
        "CV configuration and dispatcher registration remain available; "
        "the eager chunk producer has a different metadata contract."
    )


def tensor_meta(node: fx.Node) -> torch.Tensor | None:
    """Return tensor fake value metadata, if this node has tensor metadata."""
    val = node.meta.get("val")
    return val if isinstance(val, torch.Tensor) else None


def free_symbols(value: object) -> frozenset[object]:
    """Return symbolic-shape free symbols for values accepted by PyTorch."""
    from torch.fx.experimental.symbolic_shapes import free_symbols

    # PyTorch accepts symbolic scalars, tensors and nested shape expressions.
    return frozenset(free_symbols(cast("Any", value)))


def _hint(value: object) -> int | None:
    """Return a concrete/optimization hint when PyTorch exposes one."""
    if isinstance(value, int):
        return value
    if hasattr(value, "node"):
        if value.node.hint is not None:
            return int(value.node.hint)
        shape_env = getattr(value.node, "shape_env", None)
        if shape_env is not None:
            try:
                return int(shape_env.optimization_hint(value.node.expr))
            except Exception:
                return None
    return None


def _derive_symbol_hint(expr: object, extent_hint: int, symbol: object) -> int | None:
    # The algorithm probes the SymPy substitution protocol and rejects other values.
    expr = cast("Any", expr.node.expr if isinstance(expr, torch.SymInt) else expr)
    try:
        zero = expr.subs(symbol, 0)
        one = expr.subs(symbol, 1)
        if free_symbols(zero):
            return None
        offset = int(zero)
        coeff = int(one - zero)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if offset != 0 or coeff <= 0 or extent_hint % coeff:
        return None
    return extent_hint // coeff


def _merge_symbol_hint(hints: dict[object, int], symbol: object, hint: int, *, source: str) -> None:
    existing = hints.get(symbol)
    if existing is not None and existing != hint:
        raise ValueError(f"Chunk symbol {symbol} has conflicting hints: {existing} and {hint} from {source}.")
    hints[symbol] = hint


def _record_symbols_from_extent(hints: dict[object, int], extent: object, *, source: str) -> None:
    symbols = free_symbols(extent)
    if not symbols:
        return
    extent_hint = _hint(extent)
    if extent_hint is None:
        raise ValueError(
            f"Chunk pass found selected dynamic extent {extent} from {source} without an optimization hint."
        )
    for symbol in symbols:
        hint = _derive_symbol_hint(extent, int(extent_hint), symbol)
        if hint is None:
            raise ValueError(
                f"Chunk pass could not derive a base hint for symbol {symbol} from {source} extent {extent}."
            )
        _merge_symbol_hint(hints, symbol, hint, source=source)


def _selected_symbol_hints(node: fx.Node) -> dict[object, int]:
    hints = node.meta.get(CHUNK_SYMBOL_HINTS_META, {})
    return hints if isinstance(hints, dict) else {}


def chunk_symbol_hints_for_mode(gm: fx.GraphModule, mode: str | None = None) -> dict[object, int]:
    """Collect the selected chunk symbols and their optimization hints."""
    hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        for symbol, hint in _selected_symbol_hints(node).items():
            _merge_symbol_hint(hints, symbol, int(hint), source=node.name)
        if mode is None or node.op != "placeholder" or (val := tensor_meta(node)) is None:
            continue
        if mode not in ("batch", "seq"):
            raise ValueError(f"Unknown chunk mode: {mode!r}")
        dim = 0
        if dim < len(val.shape):
            _record_symbols_from_extent(hints, val.shape[dim], source=f"{node.name}.shape[{dim}]")
    return hints
