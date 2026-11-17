# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-independent checkpoint loading helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import torch

__all__ = ["filter_state_dict", "load_dcp_checkpoint"]


def filter_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    available_keys: set[str],
    *,
    allow_missing_key: Callable[[str], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Select checkpoint keys and reject required model keys that are missing."""

    missing = sorted(
        key
        for key in state_dict
        if key not in available_keys and not (allow_missing_key is not None and allow_missing_key(key))
    )
    if missing:
        raise RuntimeError(f"checkpoint is missing current model keys: {missing[:3]}")
    return {key: value for key, value in state_dict.items() if key in available_keys}


def load_dcp_checkpoint(
    model: Any,
    checkpoint: Path,
    *,
    initialize: Callable[[Any], None] | None = None,
    allow_missing_key: Callable[[str], bool] | None = None,
) -> Any:
    """Load a DCP checkpoint through the public ``state_dict`` protocol.

    Model-specific initialization and missing-key policy are injected by the
    caller, so this helper does not depend on a model class or field layout.
    """

    import torch.distributed.checkpoint as dcp

    if not checkpoint.is_dir():
        raise FileNotFoundError(f"DCP checkpoint not found: {checkpoint}")

    if initialize is not None:
        initialize(model)
    state_dict = model.state_dict()
    reader = dcp.FileSystemReader(str(checkpoint))
    available_keys = set(reader.read_metadata().state_dict_metadata)
    state_dict = filter_state_dict(
        state_dict,
        available_keys,
        allow_missing_key=allow_missing_key,
    )
    dcp.load(state_dict, checkpoint_id=str(checkpoint))
    return model
