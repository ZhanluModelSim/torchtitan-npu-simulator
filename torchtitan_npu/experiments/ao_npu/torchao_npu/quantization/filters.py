# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module and parameter-level filters for quantization on NPU."""

from typing import Protocol, TypeVar, cast

from torch import nn


class ModuleFilterFn(Protocol):
    """Filter function for ``quantize_``'s ``filter_fn`` parameter."""

    def __call__(self, mod: nn.Module, fqn: str) -> bool: ...


class ParameterFilterFn(Protocol):
    """Parameter-level filter passed to ``ParamSwapConfig`` via ``params_filter_fn``."""

    def __call__(self, param: nn.Parameter, fqn: str) -> bool: ...


# Constrained TypeVar: forces all inputs to a combinator to be the same filter
# kind, so module filters and parameter filters cannot be mixed.
FilterFnType = TypeVar("FilterFnType", ModuleFilterFn, ParameterFilterFn)


# ---------------------------------------------------------------------------
# Module-level filters
# ---------------------------------------------------------------------------


def _is_expert(module: nn.Module, fqn: str) -> bool:
    """Match modules whose names end with "experts" or "shared_experts"."""
    return fqn.split(".")[-1] in ("experts", "shared_experts")


def match_module_type(*module_types: type[nn.Module]) -> ModuleFilterFn:
    """Return a filter that matches when ``mod`` is an instance of any of ``module_types``."""

    def _filter(mod: nn.Module, fqn: str) -> bool:
        return isinstance(mod, module_types)

    return _filter


# ---------------------------------------------------------------------------
# Parameter-level filters
# ---------------------------------------------------------------------------


def _is_parameter(param: nn.Parameter, fqn: str) -> bool:
    """
    Default filter for parameter-level recursion: returns True for all
    ``nn.Parameter`` not already wrapped.
    """
    from ..wrapper_tensors.base_wrapper_tensor import BaseTrainingWeightWrapperTensor

    return isinstance(param, nn.Parameter) and not isinstance(param.data, BaseTrainingWeightWrapperTensor)


def _is_parameter_with_wrapped_data(param: nn.Parameter, fqn: str) -> bool:
    """Filter for the convert step, identifying Parameters with wrapped data."""
    from ..wrapper_tensors.base_wrapper_tensor import BaseTrainingWeightWrapperTensor

    return isinstance(param, nn.Parameter) and isinstance(param.data, BaseTrainingWeightWrapperTensor)


# ---------------------------------------------------------------------------
# Filter factories (apply to both module and parameter filters)
#
# Note: combinators below use the constrained TypeVar ``FilterFnType`` so that
# all inputs must be the same filter kind. Mixing module filters and parameter
# filters in a single call (e.g., ``all_filters(_is_expert, _is_parameter)``)
# is not designed for and not allowed — it will be rejected by the type checker.
# ---------------------------------------------------------------------------


def all_filters(*filters: FilterFnType) -> FilterFnType:
    """Return a filter that matches when all of ``filters`` match (logical AND)."""

    def _filter(obj, fqn: str) -> bool:
        return all(f(obj, fqn) for f in filters)

    return cast("FilterFnType", _filter)


def any_filter(*filters: FilterFnType) -> FilterFnType:
    """Return a filter that matches when any of ``filters`` match (logical OR)."""

    def _filter(obj, fqn: str) -> bool:
        return any(f(obj, fqn) for f in filters)

    return cast("FilterFnType", _filter)


def not_filter(filter_fn: FilterFnType) -> FilterFnType:
    """Return a filter that negates ``filter_fn`` (logical NOT)."""

    def _filter(obj, fqn: str) -> bool:
        return not filter_fn(obj, fqn)

    return cast("FilterFnType", _filter)


def match_fqn_suffix(*suffixes: str) -> FilterFnType:
    """Return a filter that matches when ``fqn`` ends with any of ``suffixes``."""

    def _filter(obj, fqn: str) -> bool:
        return any(fqn.endswith(s) for s in suffixes)

    return cast("FilterFnType", _filter)


def match_fqn_exact(*fqns: str) -> FilterFnType:
    """Return a filter that matches when ``fqn`` equals any of ``fqns`` exactly."""
    fqn_set = frozenset(fqns)

    def _filter(obj, fqn: str) -> bool:
        return fqn in fqn_set

    return cast("FilterFnType", _filter)


def match_fqn_regex(*patterns: str) -> FilterFnType:
    """Return a filter that matches when ``fqn`` fully matches any of ``patterns`` (regex)."""
    import re

    compiled = [re.compile(p) for p in patterns]

    def _filter(obj, fqn: str) -> bool:
        return any(p.fullmatch(fqn) is not None for p in compiled)

    return cast("FilterFnType", _filter)
