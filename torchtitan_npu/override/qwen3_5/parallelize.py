# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    apply_fsdp_to_multimodal_encoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)


def apply_qwen3_5_npu_fsdp(model, *, parallelism_context, training, parallelism):
    """Wrap vision independently of the decoder's pipeline stages."""
    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    if model.vision_encoder is not None:
        apply_fsdp_to_multimodal_encoder(
            model.vision_encoder,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
            cpu_offload=training.enable_cpu_offload,
            dp_mesh_dims=dp_mesh_dims,
        )
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=parallelism_context.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallelism_context.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
    )


def exchange_sequence_heads(tensors: Sequence[torch.Tensor], group: dist.ProcessGroup) -> tuple[torch.Tensor, ...]:
    """Combine CP exchanges while preserving each projection's head order."""
    degree = group.size()
    if any(tensor.shape[1] % degree for tensor in tensors):
        raise ValueError("Each projection's channel count must be divisible by CP.")
    shards = [tensor.chunk(degree, 1) for tensor in tensors]
    packed = torch.cat([parts[rank] for rank in range(degree) for parts in shards], 1)
    packed = spmd.redistribute(packed, group, src=spmd.S(0), dst=spmd.S(1))
    return packed.split([parts[0].shape[1] for parts in shards], 1)


def head_to_sequence_shard(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return spmd.redistribute(tensor, group, src=spmd.S(1), dst=spmd.S(0))


def shard_local_heads(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    return spmd.redistribute(tensor, group, src=spmd.R, dst=spmd.S(0))


@dataclass(frozen=True, slots=True)
class QwenCPMetadata:
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    actual_seq_qlen: torch.Tensor


def build_sequence_metadata(varlen):
    cu = varlen.cu_seq_q
    cu = cu.to(torch.int64)
    cu_cpu = cu.cpu()
    return QwenCPMetadata(cu, cu_cpu, cu_cpu[1:])


def _to_qwen_cp_metadata(mask):
    if mask is None or isinstance(mask, QwenCPMetadata) or not hasattr(mask, "cu_seq_q"):
        return mask
    return build_sequence_metadata(mask)


def prepare_sequence_metadata(module, args, kwargs):
    masks = kwargs.get("attention_masks")
    if masks is None:
        return args, kwargs
    if isinstance(masks, dict):
        # Qwen3.5 hybrid attention passes one mask dict per consumer key;
        # convert each varlen entry to QwenCPMetadata in place.
        kwargs["attention_masks"] = {key: _to_qwen_cp_metadata(value) for key, value in masks.items()}
    elif not isinstance(masks, QwenCPMetadata):
        kwargs["attention_masks"] = build_sequence_metadata(masks)
    return args, kwargs
