# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430

import functools
import logging

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torchtitan.distributed.context_parallel.api import cp_shard as original_cp_shard
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
)

logger = logging.getLogger(__name__)


def _varlen_from_masks(masks):
    """Return the shared VarlenMetadata behind a bare mask or an all-varlen dict."""
    if isinstance(masks, VarlenMetadata):
        return masks
    if isinstance(masks, dict):
        values = [value for value in masks.values() if value is not None]
        if values and all(isinstance(value, VarlenMetadata) for value in values):
            return values[0]
    return None


@functools.wraps(original_cp_shard)
def patched_cp_shard(
    cp_mesh: DeviceMesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks: AttentionMasksType | None,
    *args,
    **kwargs,
) -> tuple[tuple[torch.Tensor, ...], AttentionMasksType | CPVarlenMetadata | None]:
    """Build rank-local varlen metadata after sharding inputs for CP."""
    load_balancer_type = args[0] if args else kwargs.get("load_balancer_type", "headtail")
    input_seq_dim = args[1] if len(args) > 1 else kwargs.get("input_seq_dim", 1)
    varlen_metadata = _varlen_from_masks(attention_masks)
    is_varlen = varlen_metadata is not None
    batch_size = inputs[0].size(0)
    seq_len = inputs[0].size(input_seq_dim)

    inputs, output_masks = original_cp_shard(
        cp_mesh,
        inputs,
        None if is_varlen else attention_masks,
        *args,
        **kwargs,
    )

    if varlen_metadata is not None:
        assert load_balancer_type in (
            None,
            "headtail",
        ), f"varlen only support headtail as load balancer, got ({load_balancer_type})"

        cp_metadata = CPVarlenMetadata.from_global(
            varlen_metadata,
            cp_mesh,
            batch_size,
            seq_len,
            load_balancer=(
                _HeadTailLoadBalancer(seq_len, cp_mesh.size(0), cp_mesh.device_type)
                if load_balancer_type == "headtail"
                else None
            ),
        )
        if isinstance(attention_masks, dict):
            # Hybrid attention stacks carry one shared varlen metadata under
            # multiple mask keys; keep the dict shape so each block can still
            # select its mask by key after CP sharding.
            output_masks = {
                key: cp_metadata if isinstance(value, VarlenMetadata) else value
                for key, value in attention_masks.items()
            }
        else:
            output_masks = cp_metadata

    return inputs, output_masks  # pyrefly: ignore [bad-return]


def apply() -> None:
    try:
        import torchtitan.distributed.context_parallel.api as context_parallel_api
    except ImportError:
        import torchtitan.distributed.context_parallel as context_parallel_api

    # Qwen3.5 vision masks are created after NPU initialization.  Torch's
    # compiled helper starts a worker process, which cannot reinitialize NPU;
    # use the eager helper for this CP path.
    try:
        import torch.distributed.tensor.experimental._context_parallel._attention as cp_attention
        from torch.nn.attention.flex_attention import create_block_mask

        def _eager_create_block_mask(*args, **kwargs):
            kwargs.pop("separate_full_blocks", None)
            kwargs["_compile"] = False
            return create_block_mask(*args, **kwargs)

        cp_attention._compiled_create_block_mask = _eager_create_block_mask  # pyrefly: ignore [bad-assignment]
    except (ImportError, AttributeError):
        logger.debug("CP eager block-mask workaround is unavailable", exc_info=True)

    logger.info("[PATCH] torchtitan.distributed.context_parallel.api.cp_shard -> patched_cp_shard")
    context_parallel_api.cp_shard = patched_cp_shard  # pyrefly: ignore [bad-assignment]

    try:
        import torchtitan.distributed.context_parallel as context_parallel_root

        context_parallel_root.cp_shard = patched_cp_shard  # pyrefly: ignore [bad-assignment]
    except ImportError:
        pass


apply()
