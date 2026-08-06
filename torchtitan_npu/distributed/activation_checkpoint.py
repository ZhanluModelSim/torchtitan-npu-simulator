# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NPU extensions for TorchTitan activation checkpointing."""

from collections.abc import Callable, Collection, Iterator
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from threading import RLock
from typing import Any, TypeVar, cast

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import create_selective_checkpoint_contexts
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed import activation_checkpoint as titan_ac

BMM_SAC_SAVE_OPS = frozenset({torch.ops.aten.bmm.default})
_C10D_FUNCTIONAL_ATTR = "_c10d_functional"
_C10D_FUNCTIONAL_OPS = getattr(torch.ops, _C10D_FUNCTIONAL_ATTR)
ALL_TO_ALL_SAC_SAVE_OPS = frozenset({_C10D_FUNCTIONAL_OPS.all_to_all_single.default})

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class _ScopedSACState:
    enabled_ops: frozenset[object]
    mode: AbstractContextManager[Any]


_ACTIVE_SCOPED_SAC: ContextVar[_ScopedSACState | None] = ContextVar(
    "active_scoped_sac",
    default=None,
)

# TorchTitan constructs selective AC policies through a process-global helper.
_SAC_SAVE_OPS_LOCK = RLock()
_GET_SAVE_OPS_ATTR = "_get_save_ops"


@contextmanager
def extend_selective_ac_save_ops(additional_ops: Collection[object]) -> Iterator[None]:
    """Add save ops while TorchTitan constructs a selective AC policy."""
    if not additional_ops:
        yield
        return

    additional_ops = set(additional_ops)
    with _SAC_SAVE_OPS_LOCK:
        original_get_save_ops = getattr(titan_ac, _GET_SAVE_OPS_ATTR, None)
        if not callable(original_get_save_ops):
            raise RuntimeError("Installed TorchTitan does not expose the expected SAC save-op helper")

        get_native_save_ops = cast("Callable[[], Collection[object]]", original_get_save_ops)

        def get_save_ops() -> set[object]:
            return set(get_native_save_ops()) | additional_ops

        setattr(titan_ac, _GET_SAVE_OPS_ATTR, get_save_ops)
        try:
            yield
        finally:
            setattr(titan_ac, _GET_SAVE_OPS_ATTR, original_get_save_ops)


@contextmanager
def _activate_scoped_sac(
    enabled_ops: frozenset[object],
    mode: AbstractContextManager[Any],
) -> Iterator[None]:
    token = _ACTIVE_SCOPED_SAC.set(
        _ScopedSACState(
            enabled_ops=enabled_ops,
            mode=mode,
        )
    )
    try:
        yield
    finally:
        _ACTIVE_SCOPED_SAC.reset(token)


def create_scoped_selective_checkpoint_contexts(
    enabled_ops: Collection[object],
) -> tuple[AbstractContextManager[None], AbstractContextManager[None]]:
    """Create checkpoint contexts that activate SAC only at marked op calls."""
    enabled_ops = frozenset(enabled_ops)
    forward_mode, recompute_mode = create_selective_checkpoint_contexts(list(enabled_ops))
    return (
        _activate_scoped_sac(enabled_ops, forward_mode),
        _activate_scoped_sac(enabled_ops, recompute_mode),
    )


def retain_op_output(
    save_ops: Collection[object],
    function: Callable[..., _T],
    /,
    *args: Any,
    **kwargs: Any,
) -> _T:
    """Run one marked call under SAC, if its output is enabled for retention."""
    state = _ACTIVE_SCOPED_SAC.get()
    if state is None or state.enabled_ops.isdisjoint(save_ops):
        return function(*args, **kwargs)

    with state.mode:
        return function(*args, **kwargs)


def apply_scoped_selective_ac(
    model: nn.Module,
    ac_config: ActivationCheckpointConfig,
    save_ops: Collection[object],
) -> None:
    """Checkpoint transformer blocks while retaining only explicitly marked ops."""
    context_fn = partial(
        create_scoped_selective_checkpoint_contexts,
        frozenset(save_ops),
    )
    layers = model.get_submodule("layers")
    for layer_id, transformer_block in layers.named_children():
        transformer_block = checkpoint_wrapper(
            transformer_block,
            context_fn=context_fn,
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
            early_stop=ac_config.early_stop,
            debug=ac_config.debug,
        )
        layers.register_module(layer_id, transformer_block)
