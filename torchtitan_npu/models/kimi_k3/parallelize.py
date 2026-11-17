# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi parallelization using native decoder FSDP and activation checkpointing."""

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh


def parallelize_kimi_k3(model, *, parallel_dims, training, parallelism, compile_config, ac_config, dump_folder):
    if parallelism.spmd_backend != "spmd_types":
        raise ValueError("Kimi currently requires the native spmd_types backend")
    model.parallelize(parallel_dims)
    if model.attn_res_block_size is not None:
        model.layers["0"].self_attention_res.requires_grad_(False)
    if compile_config.enable:
        raise ValueError("Kimi compile adaptation is not implemented")
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)
    mesh, mesh_dims = resolve_fsdp_mesh(parallel_dims)
    expert_mesh, expert_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    apply_fsdp_to_decoder(
        model,
        mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=False,
        dp_mesh_dims=mesh_dims,
        ep_degree=parallel_dims.ep,
        edp_mesh=expert_mesh,
        edp_mesh_dims=expert_mesh_dims,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )
    return model
