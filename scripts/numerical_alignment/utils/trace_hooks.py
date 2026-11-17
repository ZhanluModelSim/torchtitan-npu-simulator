# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reusable forward-trace hook infrastructure for numerical checks.

Model-specific alignment scripts should only describe which modules map to
which trace keys.  Hook installation, tensor detaching and handle cleanup stay
in this module so DSV3, DSV3.2 and future models can share the same behavior.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

__all__ = [
    "HookSpec",
    "Trace",
    "TraceHookRegistry",
    "record_trace",
    "tensor_hook",
    "tensor_pre_hook",
]


Trace = MutableMapping[str, torch.Tensor]
Selector = Callable[[Any], torch.Tensor]


def record_trace(trace: Trace | None, key: str, value: Any) -> None:
    """Save an immutable CPU snapshot detached from the model graph."""

    if trace is not None:
        if isinstance(value, tuple) and value:
            value = value[0]
        if not isinstance(value, torch.Tensor):
            warnings.warn(f"Skipping trace {key!r}: expected a tensor, got {type(value).__name__}", stacklevel=2)
            return
        trace[key] = value.detach().cpu().clone()


def _identity(value: Any) -> torch.Tensor:
    return value


def tensor_hook(
    trace: Trace | None,
    key: str,
    selector: Selector = _identity,
) -> Callable:
    """Build a standard ``register_forward_hook`` callback."""

    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        record_trace(trace, key, selector(output))

    return hook


def tensor_pre_hook(
    trace: Trace | None,
    key: str,
    selector: Selector = lambda args: args[0],
) -> Callable:
    """Build a standard ``register_forward_pre_hook`` callback."""

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        record_trace(trace, key, selector(args))

    return hook


@dataclass
class TraceHookRegistry:
    """Own hook handles for one model invocation and remove them together."""

    trace: Trace
    handles: list[Any] = field(default_factory=list)

    def forward(
        self,
        module: nn.Module,
        key: str,
        selector: Selector = _identity,
    ) -> None:
        self.handles.append(module.register_forward_hook(tensor_hook(self.trace, key, selector)))

    def pre(
        self,
        module: nn.Module,
        key: str,
        selector: Selector = lambda args: args[0],
    ) -> None:
        self.handles.append(module.register_forward_pre_hook(tensor_pre_hook(self.trace, key, selector)))

    def callback(
        self,
        module: nn.Module,
        callback: Callable,
        *,
        pre: bool = False,
    ) -> None:
        register = module.register_forward_pre_hook if pre else module.register_forward_hook
        self.handles.append(register(callback))

    def extend(self, specs: Iterable[HookSpec]) -> None:
        for spec in specs:
            if spec.pre:
                self.pre(spec.module, spec.key, spec.selector)
            else:
                self.forward(spec.module, spec.key, spec.selector)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> TraceHookRegistry:
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.remove()


@dataclass(frozen=True)
class HookSpec:
    """Declarative mapping from a module output to a trace key."""

    module: nn.Module
    key: str
    selector: Selector = _identity
    pre: bool = False
