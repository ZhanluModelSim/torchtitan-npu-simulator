# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/4763
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pass live GraphTrainer state to graph passes that explicitly request it.

This compatibility patch supplies the model, traced result, real first batch,
and training context needed by whole-graph benchmarking. GraphTrainerEx
installs it only when an auto-overlap pipeline is selected. After the upstream
API is available, remove the private ``_requires_runtime_context`` marker and
the legacy-dict handling in ``whole_graph_benchmark.make_calibration_runner``.

Remove this module after the TorchTitan dependency includes the PR.
"""

from __future__ import annotations

import functools
import inspect
import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import torchtitan.experiments.graph_trainer.passes
import torchtitan.experiments.graph_trainer.trainer

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

original_apply_graph_passes = torchtitan.experiments.graph_trainer.passes.apply_graph_passes
original_make_fx_forward_backward_step = getattr(
    torchtitan.experiments.graph_trainer.trainer.GraphTrainer,
    "_make_fx_forward_backward_step",
    None,
)
_UPSTREAM_PIPELINE_ACCEPTS_RUNTIME_CONTEXT = (
    "runtime_context"
    in inspect.signature(torchtitan.experiments.graph_trainer.passes.construct_default_graph_passes).parameters
)

_GRAPH_PASS_RUNTIME_CONTEXT: ContextVar[Callable[[], dict[str, Any]] | None] = ContextVar(
    "npu_graph_pass_runtime_context",
    default=None,
)


def _runtime_context_pass(pass_fn, runtime_context):
    """Adapt one opted-in pass to the native two-argument pass protocol."""

    def wrapped(gm, example_inputs):
        return pass_fn(
            gm,
            example_inputs,
            runtime_context=runtime_context,
        )

    # Native filtering/debug output derives the pass name from ``__name__``.
    wrapped.__name__ = torchtitan.experiments.graph_trainer.passes._get_pass_name(pass_fn)
    wrapped.__qualname__ = getattr(pass_fn, "__qualname__", wrapped.__name__)
    return wrapped


@functools.wraps(original_apply_graph_passes)
def patched_apply_graph_passes(
    gm,
    example_inputs,
    passes,
    *,
    compile_config=None,
    respect_disable_passes=True,
    runtime_context=None,
):
    """Provide runtime state only to graph passes that explicitly request it.

    The native pass runner remains responsible for filtering, debugging and
    validation. Runtime-aware passes are adapted back to its ordinary
    ``(gm, example_inputs)`` callable contract.
    """
    if runtime_context is None:
        context_factory = _GRAPH_PASS_RUNTIME_CONTEXT.get()
        if context_factory is not None:
            runtime_context = context_factory()

    if runtime_context is not None:
        passes = [
            _runtime_context_pass(pass_fn, runtime_context)
            if getattr(pass_fn, "_requires_runtime_context", False)
            else pass_fn
            for pass_fn in passes
        ]

    return original_apply_graph_passes(
        gm,
        example_inputs,
        passes,
        compile_config=compile_config,
        respect_disable_passes=respect_disable_passes,
    )


def patched_make_fx_forward_backward_step(
    self,
    model,
    inputs,
    labels,
    global_valid_tokens,
    params,
    extra_kwargs,
):
    """Expose the active traced call to runtime-aware NPU graph passes."""

    def context_factory():
        return {
            "traced_result": self._traced_step,
            "module": model,
            "args": (inputs, labels, global_valid_tokens, extra_kwargs),
            "train_context": self.train_context,
        }

    token = _GRAPH_PASS_RUNTIME_CONTEXT.set(context_factory)
    try:
        original = original_make_fx_forward_backward_step
        if original is None:
            raise RuntimeError("TorchTitan has no legacy _make_fx_forward_backward_step entrypoint")
        return original(
            self,
            model,
            inputs,
            labels,
            global_valid_tokens,
            params,
            extra_kwargs,
        )
    finally:
        _GRAPH_PASS_RUNTIME_CONTEXT.reset(token)


if original_make_fx_forward_backward_step is not None:
    functools.update_wrapper(
        patched_make_fx_forward_backward_step,
        original_make_fx_forward_backward_step,
    )
patched_make_fx_forward_backward_step._npu_runtime_context_patch = True  # pyrefly: ignore [missing-attribute]


def apply() -> None:
    if _UPSTREAM_PIPELINE_ACCEPTS_RUNTIME_CONTEXT:
        return
    if original_make_fx_forward_backward_step is None:
        logger.warning(
            "[PATCH] Skipped GraphTrainer runtime-context adapter because "
            "TorchTitan exposes neither the upstream runtime-context API nor "
            "the legacy _make_fx_forward_backward_step entrypoint"
        )
        return
    trainer = torchtitan.experiments.graph_trainer.trainer.GraphTrainer
    current = getattr(trainer, "_make_fx_forward_backward_step", None)
    if getattr(current, "_npu_runtime_context_patch", False):
        return
    logger.info("[PATCH] GraphTrainer graph-pass runtime context -> NPU adapter")
    torchtitan.experiments.graph_trainer.passes.apply_graph_passes = patched_apply_graph_passes
    # GraphTrainer imports this symbol directly, so patch its module alias too.
    torchtitan.experiments.graph_trainer.trainer.apply_graph_passes = patched_apply_graph_passes
    trainer._make_fx_forward_backward_step = (  # pyrefly: ignore [missing-attribute]
        patched_make_fx_forward_backward_step
    )
