# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/distributed/context_parallel.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
import torch.nn as nn

import torchtitan.distributed.context_parallel as titan_cp
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.tools.logging import logger


_orig_apply_cp_to_attention_module = titan_cp.apply_cp_to_attention_module


class CustomContextParallelContext:
    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        buffers: list[torch.Tensor] | None = None,
        buffer_seq_dims: list[int] | None = None,
        no_restore_buffers: set[torch.Tensor] | None = None,
        load_balance: bool = False,
    ):
        self._mesh = mesh
        self._buffers = buffers or []
        self._buffer_seq_dims = buffer_seq_dims or []
        self._no_restore_buffers = no_restore_buffers or set()
        self._load_balance = load_balance
        self._ctx: contextlib.AbstractContextManager | None = None

    @torch.no_grad()
    def __enter__(self):
        from torch.distributed.tensor.experimental import context_parallel

        self._ctx = context_parallel(
            self._mesh,
            buffers=self._buffers,
            buffer_seq_dims=self._buffer_seq_dims,
            no_restore_buffers=self._no_restore_buffers,
        )
        self._ctx.__enter__()
        return self

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


def validate_ulysses_configs(
    *,
    job_config: object | None,
    model_args: object | None,
    cp_mesh: DeviceMesh,
) -> None:
    cp_degree = cp_mesh.size()

    n_heads = getattr(model_args, "n_heads", None) if model_args is not None else None
    if n_heads is not None and n_heads % cp_degree != 0:
        raise ValueError(
            f"[ulysses] n_heads={n_heads} must be divisible by "
            f"context_parallel_degree={cp_degree}."
        )

    if job_config is not None:
        training = getattr(job_config, "training", None)
        seq_len = getattr(training, "seq_len", None) if training is not None else None
        if seq_len is not None and seq_len % cp_degree != 0:
            raise ValueError(
                f"[ulysses] seq_len={seq_len} must be divisible by "
                f"context_parallel_degree={cp_degree}."
            )

        parallelism = getattr(job_config, "parallelism", None)
        tp_degree = (
            getattr(parallelism, "tensor_parallel_degree", 1)
            if parallelism is not None
            else 1
        )
        if n_heads is not None and n_heads % (tp_degree * cp_degree) != 0:
            raise ValueError(
                f"[ulysses] n_heads={n_heads} must be divisible by "
                f"tp_degree * cp_degree = {tp_degree} * {cp_degree} = {tp_degree * cp_degree}."
            )


def validate_dsa_converters(
    *,
    job_config: object | None,
    converters: list[str] | tuple[str, ...] | None = None,
) -> None:
    resolved_converters = converters
    if resolved_converters is None and job_config is not None:
        model_cfg = getattr(job_config, "model", None)
        resolved_converters = (
            getattr(model_cfg, "converters", None) if model_cfg is not None else None
        )
    if not resolved_converters or "npu_dsa" not in resolved_converters:
        raise ValueError(
            '[dsa] attention_type="dsa" requires "npu_dsa" in converters. '
            f"Got converters={resolved_converters!r}."
        )


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
    *,
    attention_type: str | None = None,
    job_config: object | None = None,
    model_args: object | None = None,
    tp_mesh: DeviceMesh | None = None,
    converters: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Patched ``apply_cp_to_attention_module`` with NPU-only routing.

    Upstream signature is ``(attention_modules, cp_mesh)`` — ``attention_type``
    is inferred from the module class. To preserve compatibility with that
    contract while still letting NPU paths (``dsa`` / ``ulysses``) opt in to
    custom CP backends, ``attention_type`` is keyword-only with default
    ``None``; when unset we delegate straight to the upstream implementation.
    """
    first_module = attention_modules[0] if attention_modules else None
    module_name = type(first_module).__name__ if first_module is not None else "None"

    if attention_type == "dsa":
        validate_dsa_converters(job_config=job_config, converters=converters)
        logger.info(
            f"CP router selected route=dsa module={module_name} cp_degree={cp_mesh.size()}"
        )

        from torchtitan_npu.distributed.context_parallel.dsa_cp import (
            patch_dsa_for_context_parallel,
        )

        patch_dsa_for_context_parallel(
            cp_mesh=cp_mesh, model_args=model_args, tp_mesh=tp_mesh
        )
    elif attention_type == "ulysses":
        if first_module is not None and not isinstance(
            first_module, ScaledDotProductAttention
        ):
            raise ValueError(
                "[ulysses] expected ScaledDotProductAttention modules, "
                f"got {type(first_module).__name__}."
            )
        validate_ulysses_configs(
            job_config=job_config, model_args=model_args, cp_mesh=cp_mesh
        )
        logger.info(
            f"CP router selected route=ulysses module={module_name} cp_degree={cp_mesh.size()}"
        )

        from torchtitan_npu.distributed.context_parallel.ulysses_cp import (
            patch_ulysses_for_context_parallel,
        )

        patch_ulysses_for_context_parallel(cp_mesh=cp_mesh)
    else:
        logger.info(
            f"CP router selected route=upstream module={module_name} cp_degree={cp_mesh.size()}"
        )
        # ``attention_type is None`` (upstream-style call) or any other value
        # falls back to the upstream implementation, which infers the CP
        # plan from the first module's class.
        _orig_apply_cp_to_attention_module(
            attention_modules=attention_modules,
            cp_mesh=cp_mesh,
        )
    return None


titan_cp.apply_cp_to_attention_module = apply_cp_to_attention_module
