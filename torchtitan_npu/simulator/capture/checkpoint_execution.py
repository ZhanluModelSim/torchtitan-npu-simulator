# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Execution context tagging for activation-checkpoint replay."""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Callable, Iterable, Iterator

import torch.nn as nn

ORIGINAL_FORWARD = "original_forward"
RECOMPUTE = "recompute"
BACKWARD = "backward"
OPTIMIZER = "optimizer"

_execution_kind: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "simulator_execution_kind",
    default=None,
)
_INSTALL_MARKER = "_simulator_checkpoint_execution_tracking"


@contextlib.contextmanager
def execution_kind_context(kind: str) -> Iterator[None]:
    token = _execution_kind.set(kind)
    try:
        yield
    finally:
        _execution_kind.reset(token)


def current_execution_kind(phase: str) -> str:
    """Return the precise checkpoint context, or derive a normal step kind."""
    explicit_kind = _execution_kind.get()
    if explicit_kind is not None:
        return explicit_kind
    return {
        "forward": ORIGINAL_FORWARD,
        "backward": BACKWARD,
        "optimizer": OPTIMIZER,
    }.get(phase, phase)


class _ExecutionKindContext(contextlib.AbstractContextManager):
    """Add execution-kind tagging without making a reusable context one-shot.

    Non-reentrant checkpoint creates its forward/recompute context pair once
    per forward. Schedules that split input- and weight-gradient calculation
    can enter that recompute context more than once. A generator-based wrapper
    is inherently one-shot, even when the wrapped context (normally
    ``nullcontext`` for full AC) is reusable.
    """

    def __init__(self, first: contextlib.AbstractContextManager, kind: str) -> None:
        self._first = first
        self._kind = kind
        self._kind_contexts: list[contextlib.AbstractContextManager] = []

    def __enter__(self):  # noqa: ANN204
        kind_context = execution_kind_context(self._kind)
        kind_context.__enter__()
        try:
            result = self._first.__enter__()
        except BaseException as error:
            kind_context.__exit__(type(error), error, error.__traceback__)
            raise
        self._kind_contexts.append(kind_context)
        return result

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        kind_context = self._kind_contexts.pop()
        try:
            return self._first.__exit__(exc_type, exc_value, traceback)
        finally:
            kind_context.__exit__(exc_type, exc_value, traceback)


def _compose_context_fn(
    context_fn: Callable[[], tuple[contextlib.AbstractContextManager, contextlib.AbstractContextManager]] | None,
) -> Callable[[], tuple[contextlib.AbstractContextManager, contextlib.AbstractContextManager]]:
    def tracked_contexts() -> tuple[contextlib.AbstractContextManager, contextlib.AbstractContextManager]:
        if context_fn is None:
            forward_context = contextlib.nullcontext()
            recompute_context = contextlib.nullcontext()
        else:
            forward_context, recompute_context = context_fn()
        return (
            _ExecutionKindContext(forward_context, ORIGINAL_FORWARD),
            _ExecutionKindContext(recompute_context, RECOMPUTE),
        )

    return tracked_contexts


def install_checkpoint_execution_tracking(model_parts: Iterable[nn.Module]) -> int:
    """Instrument non-reentrant CheckpointWrappers without replacing AC policy.

    Torchtitan has already applied activation checkpointing by the time the
    simulator receives ``model_parts``. PyTorch stores checkpoint arguments in
    ``CheckpointWrapper.checkpoint_fn`` (a ``functools.partial``), so the
    simulator can compose its marker with full or selective AC's context_fn.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        CheckpointWrapper,
    )

    installed = 0
    seen: set[int] = set()
    for model in model_parts:
        for module in model.modules():
            if id(module) in seen or not isinstance(module, CheckpointWrapper):
                continue
            seen.add(id(module))
            if getattr(module, _INSTALL_MARKER, False):
                continue
            if module.checkpoint_impl == CheckpointImpl.REENTRANT:
                raise RuntimeError(
                    "Simulator recompute tracking requires non-reentrant activation checkpointing"
                )
            checkpoint_fn = module.checkpoint_fn
            if not isinstance(checkpoint_fn, functools.partial):
                raise TypeError(
                    "Unsupported CheckpointWrapper.checkpoint_fn; expected functools.partial"
                )
            checkpoint_fn.keywords["context_fn"] = _compose_context_fn(
                checkpoint_fn.keywords.get("context_fn")
            )
            setattr(module, _INSTALL_MARKER, True)
            installed += 1
    return installed
