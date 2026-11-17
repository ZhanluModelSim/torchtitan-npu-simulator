# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan adapter for DSV4 numerical-alignment checks.

All knowledge of the DSV4 forward protocol and module layout stays here (or in
``deepseek_v4.trace``). Entry points should use this module as their stable facade.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch

from torchtitan_npu.models.common.restore import DOC_CHUNK_LAYOUT

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

    cfg = dsv4_config.build_torchtitan_config(model_flavor, seq_len=seq_len)
    cfg.training.num_tokens_per_microbatch_per_dp_rank = global_batch_size * seq_len
    cfg.training.num_tokens_per_train_step = global_batch_size * seq_len
    cfg.training.max_context_length = seq_len
    model_config = cfg.model
    dsv4_config.truncate_model_config(model_config, n_layers)
    if (num_experts is None) != (top_k is None):
        raise ValueError("num_experts and top_k must be set together")
    if num_experts is not None and top_k is not None:
        dsv4_config.configure_reduced_moe(
            model_config,
            num_experts=num_experts,
            top_k=top_k,
        )
    # The recipe constructs sharding declarations before overrides.
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

    positions = torch.arange(tokens.shape[1], dtype=torch.int64, device=tokens.device).repeat(tokens.shape[0])
    # The model owns the metadata seam (upstream #4897/#4116): it builds one entry per
    # attention geometry and hands the map to the forward, so an offline forward owes it the
    # same preprocessing the trainer does.
    parallel_dims = SimpleNamespace(cp_enabled=False, get_mesh=lambda name: None)
    parallelism = SimpleNamespace(
        spmd_backend=None,
        context_parallel_load_balancer=DOC_CHUNK_LAYOUT,
        context_parallel_ptrr_mask_key=None,
    )
    flat_tokens = tokens.flatten()
    inputs, _, extra_kwargs = model.preprocess_inputs(
        {"input": flat_tokens, "labels": flat_tokens, "positions": positions},
        parallel_dims=parallel_dims,
        parallelism=parallelism,
        max_num_documents=tokens.shape[0],
        max_context_length=tokens.shape[1],
    )
    logits = model(inputs, **extra_kwargs).reshape(*tokens.shape, -1)
    return logits[:, -1, :] if last_token_only else logits
