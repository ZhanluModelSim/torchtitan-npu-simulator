# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430

import functools
import logging

import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torchtitan.distributed.context_parallel import cp_shard as original_cp_shard
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
    CPVarlenMetadata,
)

logger = logging.getLogger(__name__)


@functools.wraps(original_cp_shard)
def patched_cp_shard(
    cp_mesh,
    inputs: tuple[torch.Tensor, ...],
    attention_masks,
    load_balancer_type="headtail",
    input_seq_dim=1,
):
    """Build rank-local varlen metadata after sharding inputs for CP."""
    is_varlen = isinstance(attention_masks, VarlenMetadata)
    batch_size = inputs[0].size(0)
    seq_len = inputs[0].size(input_seq_dim)

    inputs, output_masks = original_cp_shard(
        cp_mesh,
        inputs,
        None if is_varlen else attention_masks,
        load_balancer_type,
        input_seq_dim,
    )

    if is_varlen:
        assert load_balancer_type in (
            None,
            "headtail",
        ), f"varlen only support headtail as load balancer, got ({load_balancer_type})"

        output_masks = CPVarlenMetadata.from_global(
            attention_masks,
            cp_mesh,
            batch_size,
            seq_len,
            load_balancer=(
                _HeadTailLoadBalancer(seq_len, cp_mesh.size(0), cp_mesh.device_type)
                if load_balancer_type == "headtail"
                else None
            ),
        )

    return inputs, output_masks


def apply() -> None:
    import torchtitan.distributed.context_parallel as cp_module

    logger.info(
        "[PATCH] torchtitan.distributed.context_parallel.cp_shard -> patched_cp_shard"
    )
    cp_module.cp_shard = patched_cp_shard  # pyrefly: ignore [bad-assignment]


apply()
