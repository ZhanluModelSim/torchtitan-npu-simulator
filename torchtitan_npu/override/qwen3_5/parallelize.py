# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, apply_fsdp_to_vision_encoder


@dataclass(frozen=True, slots=True)
class QwenCPMetadata:
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    actual_seq_qlen: torch.Tensor


def shard_local_heads(tensor, mesh):
    return tensor.chunk(mesh.size(), 0)[mesh.get_local_rank()].contiguous()


def sequence_to_head_shard(tensor, mesh, head_dim):
    tensor = DTensor.from_local(tensor, mesh, (Shard(1),), run_check=False)
    return tensor.redistribute(placements=(Shard(head_dim),)).to_local()


def head_to_sequence_shard(tensor, mesh, head_dim):
    tensor = DTensor.from_local(tensor, mesh, (Shard(head_dim),), run_check=False)
    return tensor.redistribute(placements=(Shard(1),)).to_local()


def exchange_sequence_heads(tensors, mesh, head_dim):
    degree = mesh.size()
    shards = [tensor.chunk(degree, head_dim) for tensor in tensors]
    packed = torch.cat([part[rank] for rank in range(degree) for part in shards], head_dim)
    packed = sequence_to_head_shard(packed, mesh, head_dim)
    return packed.split([part[0].size(head_dim) for part in shards], head_dim)


def build_sequence_metadata(varlen, mesh, batch, length):
    cu = varlen.cu_seq_q
    if hasattr(varlen, "k_global_gather_indices"):
        reset = torch.zeros(batch * length, dtype=torch.bool, device=cu.device)
        starts = cu[:-1].to(torch.long)
        reset[starts[cu.diff() == varlen.cu_seq_k.diff()]] = True
        reset = DTensor.from_local(reset.view(batch, length), mesh, (Shard(1),), run_check=False)
        reset = reset.redistribute(placements=(Replicate(),)).to_local()
        starts = reset.flatten().nonzero(as_tuple=False).flatten()
        starts = starts[starts != 0]
        cu = torch.cat((starts.new_zeros(1), starts, starts.new_tensor([reset.numel()])))
    cu = cu.to(torch.int64)
    cu_cpu = cu.cpu()
    return QwenCPMetadata(cu, cu_cpu, cu_cpu[1:])


def prepare_sequence_metadata(module, args, kwargs):
    varlen = kwargs.get("attention_masks")
    if varlen is not None and not isinstance(varlen, QwenCPMetadata):
        batch, length = args[0].shape[:2]
        kwargs["attention_masks"] = build_sequence_metadata(varlen, module.context_parallel_mesh, batch, length)
    return args, kwargs


def parallelize_qwen3_5_cp(model, *, parallel_dims, **kwargs):
    mesh = parallel_dims.get_mesh("cp")
    model.context_parallel_mesh = mesh
    model.register_forward_pre_hook(prepare_sequence_metadata, with_kwargs=True)
    for block in model.layers.values():
        module = block.attn.inner_attention if block.full_attn else block.attn
        module.context_parallel_mesh = mesh

    training, parallelism = kwargs["training"], kwargs["parallelism"]
    compile_config, ac_config = kwargs["compile_config"], kwargs["ac_config"]
    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    ac_policy = None if ac_config is None else ac_config.build(dump_folder=kwargs["dump_folder"])
    modules = (model,) if model.vision_encoder is None else (model, model.vision_encoder)
    for module in modules:
        if ac_policy is not None:
            ac_policy.apply(module)
        if model_compile_enabled:
            apply_compile(module, compile_config)
    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    if model.vision_encoder is not None:
        apply_fsdp_to_vision_encoder(
            model.vision_encoder,
            fsdp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
        )
    apply_fsdp_to_decoder(
        model,
        fsdp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=1,
        edp_mesh=None,
    )
    return model
