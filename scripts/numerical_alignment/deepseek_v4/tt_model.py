# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan adapter for DSV4 numerical-alignment checks.

All knowledge of the DSV4 forward protocol and module layout stays here (or in
``deepseek_v4.trace``). Entry points should use this module as their stable facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..utils import checkpoint as checkpoint_utils
from . import config as dsv4_config

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


__all__ = [
    "build_model_config",
    "forward_logits",
    "load_model",
]


def build_model_config(
    n_layers: int,
    overrides: Sequence[str],
    global_batch_size: int = 1,
    *,
    seq_len: int = 4096,
    model_flavor: str = dsv4_config.DEFAULT_MODEL_FLAVOR,
    num_experts: int | None = None,
    top_k: int | None = None,
):
    """Build an overridden DSV4 model config truncated to ``n_layers``."""

    from torchtitan.config import apply_overrides

    cfg = dsv4_config.build_torchtitan_config(model_flavor)
    cfg.training.global_batch_size = global_batch_size
    cfg.training.seq_len = seq_len
    model_config = cfg.model_spec.model
    dsv4_config.truncate_model_config(model_config, n_layers)
    if (num_experts is None) != (top_k is None):
        raise ValueError("num_experts and top_k must be set together")
    if num_experts is not None and top_k is not None:
        dsv4_config.configure_reduced_moe(
            model_config,
            num_experts=num_experts,
            top_k=top_k,
        )
    # Mirror the trainer order: update_from_config installs the sharding
    # configs that the CANN overrides derive from, so it must run first.
    model_config.update_from_config(config=cfg)
    cfg.override.imports = list(overrides)
    apply_overrides(cfg.override, cfg)
    return model_config


def _allow_missing_checkpoint_key(key: str) -> bool:
    return key.endswith(".expert_bias_E")


def load_model(
    checkpoint: Path,
    n_layers: int,
    overrides: Sequence[str],
    device: torch.device,
    global_batch_size: int = 1,
    *,
    seq_len: int = 4096,
    model_flavor: str = dsv4_config.DEFAULT_MODEL_FLAVOR,
    num_experts: int | None = None,
    top_k: int | None = None,
):
    """Build one DSV4 override recipe and load ``checkpoint`` into it."""

    model = build_model_config(
        n_layers,
        overrides,
        global_batch_size,
        seq_len=seq_len,
        model_flavor=model_flavor,
        num_experts=num_experts,
        top_k=top_k,
    ).build()
    # Build the RoPE cache on the execution device. CPU precomputation changes
    # the BF16 rounding boundaries used by the inference reference.
    return checkpoint_utils.load_dcp_checkpoint(
        model,
        checkpoint,
        initialize=lambda module: module.init_states(buffer_device=device),
        allow_missing_key=_allow_missing_checkpoint_key,
    )


def forward_logits(
    model,
    tokens: torch.Tensor,
    *,
    last_token_only: bool = True,
) -> torch.Tensor:
    """Run the DSV4 TorchTitan forward protocol and return logits."""

    positions = (
        torch.arange(
            tokens.shape[1],
            dtype=torch.int64,
            device=tokens.device,
        )
        .unsqueeze(0)
        .expand(tokens.shape[0], -1)
    )
    tokens, _, extra_kwargs = model.build_attention_masks(tokens, None, {"positions": positions})
    logits = model(tokens, **extra_kwargs)
    return logits[:, -1, :] if last_token_only else logits
