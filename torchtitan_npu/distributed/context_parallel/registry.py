# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plugin registry for custom Context Parallel strategies.

CP strategies are registered via ``register_cp_strategy(detector, applier)``.
When ``apply_cp_to_attention_module`` walks the model, each module is tested
against registered detectors.  The first matching strategy is applied.
Unmatched modules raise ``NotImplementedError``.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.tools.logging import logger

_cp_strategies: list[
    tuple[Callable[[nn.Module], bool], Callable[[nn.Module, DeviceMesh], None]]
] = []

# Handlers for attention-mask types that upstream ``cp_shard`` cannot
# sequence-shard (e.g. DeepSeek-V4 SMLA varlen metadata). See
# ``register_cp_mask_handler`` / ``adjust_cp_mask``.
_cp_mask_handlers: list[Callable[[Any, DeviceMesh], Any | None]] = []


def register_cp_strategy(
    detector: Callable[[nn.Module], bool],
    applier: Callable[[nn.Module, DeviceMesh], None],
) -> None:
    """Register a CP strategy.

    Args:
        detector: Receives a module, returns True if this strategy applies.
        applier: Receives (module, cp_mesh), applies the CP parallelization.
    """
    _cp_strategies.append((detector, applier))


def register_cp_mask_handler(
    handler: Callable[[Any, DeviceMesh], Any | None],
) -> None:
    """Register a Context-Parallel attention-mask handler.

    Some attention backends carry their CP metadata as a custom object that
    upstream ``cp_shard`` cannot sequence-shard. A handler receives
    ``(attention_masks, cp_mesh)`` and returns the per-rank-adjusted masks if it
    owns this mask type, or ``None`` to defer to the next handler. This keeps
    backend-specific logic out of the generic ``cp_shard`` patch.
    """
    _cp_mask_handlers.append(handler)


def adjust_cp_mask(attention_masks: Any, cp_mesh: DeviceMesh) -> tuple[bool, Any]:
    """Dispatch ``attention_masks`` to the registered CP mask handlers.

    Returns ``(handled, masks)``. ``handled is True`` means a handler owns this
    mask type: the masks must NOT be sequence-sharded and ``masks`` is the
    per-rank replacement for this CP rank. Otherwise the masks are returned
    unchanged for normal upstream handling.
    """
    for handler in _cp_mask_handlers:
        adjusted = handler(attention_masks, cp_mesh)
        if adjusted is not None:
            return True, adjusted
    return False, attention_masks


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
) -> None:
    """Apply custom CP strategies to each attention module.

    Each module is tested against registered detectors.  The first matching
    strategy is applied.  Unmatched modules raise ``NotImplementedError``.

    Args:
        attention_modules: Sequence of attention modules to apply CP to.
        cp_mesh: Device mesh for the context parallel dimension.
    """
    for module in attention_modules:
        module_name = type(module).__name__
        applied = False
        for detector, applier in _cp_strategies:
            if detector(module):
                logger.info(
                    f"CP: matched strategy for {module_name} "
                    f"cp_degree={cp_mesh.size()}"
                )
                applier(module, cp_mesh)
                applied = True
                break
        if not applied:
            raise NotImplementedError(
                f"No custom CP strategy found for module "
                f"{module_name}.  Registered detectors have been exhausted."
            )

    logger.info("Applied custom Context Parallel to the model")
