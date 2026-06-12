# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MTP Context Parallel Patch

This patch adds Multi-Token Prediction (MTP) support to context parallelism
in a non-invasive way by monkey-patching the prepare_context_parallel_input function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torchtitan.distributed.context_parallel as titan_cp
from torchtitan.tools.logging import init_logger, logger

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

init_logger()


_orig_prepare_cp_input = titan_cp.prepare_context_parallel_input

_cached_num_mtp_modules = None


def prepare_context_parallel_input(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    extra_kwargs: dict[str, Any],
    cp_mesh: DeviceMesh,
    device: torch.device,
    load_balancer_type: str | None = "headtail",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    global _cached_num_mtp_modules

    if _cached_num_mtp_modules is None:
        # Imported lazily: a module-level import would pull in
        # torchtitan.trainer before this patch installs its monkey-patch,
        # defeating the import-order fix in torchtitan_npu/__init__.py.
        from torchtitan_npu.patches.torchtitan._trainer_config_stash import (
            get_trainer_config,
        )

        trainer_config = get_trainer_config()
        _cached_num_mtp_modules = (
            getattr(trainer_config.training, "num_mtp_modules", 0) if trainer_config is not None else 0
        )

    if _cached_num_mtp_modules > 0:
        return _mtp_prepare_cp_input(
            inputs,
            labels,
            extra_kwargs,
            cp_mesh,
            device,
            _cached_num_mtp_modules,
            load_balancer_type,
        )

    return _orig_prepare_cp_input(inputs, labels, extra_kwargs, cp_mesh, device, load_balancer_type)


def _mtp_prepare_cp_input(inputs, labels, extra_kwargs, cp_mesh, device, num_mtp_modules, load_balancer_type):
    attention_masks = extra_kwargs.get("attention_masks", None)

    main_inputs = inputs[:, :-num_mtp_modules]
    main_labels = labels[:, :-num_mtp_modules]
    mtp_inputs = inputs[:, num_mtp_modules:]
    mtp_labels = labels[:, num_mtp_modules:]

    positions = torch.arange(0, main_inputs.shape[1], dtype=torch.int32, device=device).expand(main_inputs.shape)

    (
        (
            main_inputs,
            mtp_inputs,
            main_labels,
            mtp_labels,
            positions,
        ),
        attention_masks,
    ) = titan_cp.cp_shard(
        cp_mesh,
        (main_inputs, mtp_inputs, main_labels, mtp_labels, positions),
        attention_masks,
        load_balancer_type,
    )

    mtp_inputs = mtp_inputs[:, -num_mtp_modules:]
    mtp_labels = mtp_labels[:, -num_mtp_modules:]

    inputs = torch.cat([main_inputs, mtp_inputs], dim=-1)
    labels = torch.cat([main_labels, mtp_labels], dim=-1)

    extra_kwargs["positions"] = positions
    if attention_masks is not None:
        extra_kwargs["attention_masks"] = attention_masks

    return inputs, labels, extra_kwargs


titan_cp.prepare_context_parallel_input = prepare_context_parallel_input

logger.info("[Patch] MTP Context Parallel enabled")
