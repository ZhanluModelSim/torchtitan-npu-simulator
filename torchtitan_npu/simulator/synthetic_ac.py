# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Selective activation-checkpoint semantics for simulator synthetic ops."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from typing import Any, Callable, Iterator

import torch


_save_patterns: ContextVar[tuple[str, ...]] = ContextVar(
    "simulator_synthetic_ac_save_patterns",
    default=(),
)


@dataclass(slots=True)
class _SyntheticACSession:
    save_patterns: tuple[str, ...]
    cached_outputs: dict[tuple[str, str, int], Any] = field(default_factory=dict)
    occurrence_by_op: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )
    is_recompute: bool = False

    def begin_pass(self, *, is_recompute: bool) -> None:
        self.is_recompute = is_recompute
        self.occurrence_by_op.clear()

    def should_save(self, raw_op_type: str) -> bool:
        return any(fnmatchcase(raw_op_type, pattern) for pattern in self.save_patterns)

    def next_key(self, raw_op_type: str, module_path: str) -> tuple[str, str, int]:
        prefix = (raw_op_type, module_path)
        occurrence = self.occurrence_by_op[prefix]
        self.occurrence_by_op[prefix] += 1
        return raw_op_type, module_path, occurrence


_active_session: ContextVar[_SyntheticACSession | None] = ContextVar(
    "simulator_synthetic_ac_session",
    default=None,
)


@contextlib.contextmanager
def synthetic_ac_policy_context(patterns: tuple[str, ...]) -> Iterator[None]:
    """Select synthetic operators whose outputs cross AC recomputation."""
    token = _save_patterns.set(patterns)
    try:
        yield
    finally:
        _save_patterns.reset(token)


class SyntheticACPassContext(contextlib.AbstractContextManager):
    """One reusable forward or recompute side of a checkpoint context pair."""

    def __init__(self, session: _SyntheticACSession, *, is_recompute: bool) -> None:
        self._session = session
        self._is_recompute = is_recompute
        self._tokens: list[Any] = []

    def __enter__(self) -> "SyntheticACPassContext":
        self._session.begin_pass(is_recompute=self._is_recompute)
        self._tokens.append(_active_session.set(self._session))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        _active_session.reset(self._tokens.pop())


def synthetic_ac_contexts() -> tuple[
    contextlib.AbstractContextManager,
    contextlib.AbstractContextManager,
]:
    """Return contexts sharing one cache for one checkpoint invocation."""
    patterns = _save_patterns.get()
    if not patterns:
        return contextlib.nullcontext(), contextlib.nullcontext()
    session = _SyntheticACSession(patterns)
    return (
        SyntheticACPassContext(session, is_recompute=False),
        SyntheticACPassContext(session, is_recompute=True),
    )


def run_synthetic_op(
    raw_op_type: str,
    *,
    inputs: list[torch.Tensor],
    output_factory: Callable[[], Any],
    module_path: str = "",
    attrs: dict[str, Any] | None = None,
) -> Any:
    """Execute/record a synthetic op, or reuse its saved forward outputs.

    Returning the exact forward tensor objects is intentional: the memory
    model uses tensor identity to recognize values retained across activation
    checkpoint recomputation.
    """
    session = _active_session.get()
    should_save = session is not None and session.should_save(raw_op_type)
    key = session.next_key(raw_op_type, module_path) if should_save else None

    if should_save and session is not None and session.is_recompute:
        assert key is not None
        if key not in session.cached_outputs:
            raise RuntimeError(
                "Synthetic AC replay diverged from original forward for "
                f"{raw_op_type!r} at {module_path!r}, occurrence {key[2]}"
            )
        outputs = session.cached_outputs[key]
        from torchtitan_npu.simulator.capture.dispatch_capture import (
            get_active_capture,
        )

        capture = get_active_capture()
        if capture is not None:
            capture.record_synthetic_ac_cache_hit(
                raw_op_type,
                outputs,
                module_path=module_path,
            )
        return outputs

    outputs = output_factory()
    flat_outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    from torchtitan_npu.simulator.capture.dispatch_capture import get_active_capture

    capture = get_active_capture()
    if capture is not None:
        capture.record_synthetic_op(
            raw_op_type,
            inputs=inputs,
            outputs=flat_outputs,
            module_path=module_path,
            attrs=attrs,
        )
    if should_save and session is not None:
        assert key is not None
        session.cached_outputs[key] = outputs
    return outputs
