# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, apply_fsdp_to_vision_encoder
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)


@contextmanager
def _qwen_aot_eager_compile_options(backend: str):
    """Isolate Qwen3.5 recompilation caches for dynamic multimodal inputs.

    The upstream compiler applies fullgraph=True to every transformer block.
    Qwen3.5 multimodal batches legitimately vary in vision-token and
    packed-document metadata shapes, so one shared checkpoint-wrapper code
    object can exceed Dynamo's recompilation cache. Keep this policy local to
    the Qwen3.5 adapter and only for aot_eager; other models and backends
    retain the upstream compile contract.
    """
    if backend != "aot_eager":
        yield
        return

    original_compile = torch.nn.Module.compile
    try:
        compile_parameters = inspect.signature(torch.compile).parameters
    except (TypeError, ValueError):
        compile_parameters = {}

    def _compile_qwen_block(self, *args, **kwargs):
        # Isolate per-call cache entries when supported; heterogeneous Qwen
        # blocks otherwise share checkpoint-wrapper recompilation counters.
        # Keep upstream fullgraph=True so each block remains an aot_eager
        # graph; isolation prevents unrelated block variants exhausting the
        # shared Dynamo cache.
        if "isolate_recompiles" in compile_parameters:
            kwargs.setdefault("isolate_recompiles", True)
        return original_compile(self, *args, **kwargs)

    torch.nn.Module.compile = _compile_qwen_block
    try:
        yield
    finally:
        torch.nn.Module.compile = original_compile


def _apply_qwen_compile(module, *, parallel_dims, compile_config):
    """Apply upstream compilation with a Qwen3.5-specific shape policy."""
    with _qwen_aot_eager_compile_options(compile_config.backend):
        return apply_compile(
            module,
            parallel_dims=parallel_dims,
            compile_config=compile_config,
        )


@dataclass(frozen=True, slots=True)
class QwenCPMetadata:
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor
    actual_seq_qlen: torch.Tensor


def parallelize_qwen3_5_npu(model, *, parallel_dims, **kwargs):
    """Apply the standard Qwen3.5 sharding path with NPU vision handling.

    The pinned upstream parallelizer enables pipeline partitioning for the
    vision encoder.  Its multimodal inputs are assembled before pipeline
    stages, so the encoder must remain replicated while the decoder keeps the
    configured pipeline setting.  Context-parallel recipes use the dedicated
    function below.
    """
    training, parallelism = kwargs["training"], kwargs["parallelism"]
    compile_config, ac_config = kwargs["compile_config"], kwargs["ac_config"]
    if parallelism.spmd_backend == "full_dtensor":
        raise NotImplementedError("full_dtensor is not supported yet.")

    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    if parallel_dims.cp_enabled:
        if parallel_dims.ep_enabled:
            raise NotImplementedError("Qwen3.5 NPU Context Parallel does not support Expert Parallel yet.")
        if parallel_dims.tp_enabled:
            raise NotImplementedError("Qwen3.5 NPU Context Parallel does not support Tensor Parallel yet.")
        if getattr(model, "vision_encoder", None) is not None:
            raise NotImplementedError(
                "Qwen3.5-VL NPU Context Parallel is not supported for multimodal attention masks yet."
            )
        return parallelize_qwen3_5_cp(model, parallel_dims=parallel_dims, **kwargs)

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        ac_policy = ac_config.build(dump_folder=kwargs["dump_folder"])
        ac_policy.apply(model)
        if model.vision_encoder is not None:
            ac_policy.apply(model.vision_encoder)

    if model_compile_enabled:
        _apply_qwen_compile(model, parallel_dims=parallel_dims, compile_config=compile_config)
        if model.vision_encoder is not None:
            _apply_qwen_compile(
                model.vision_encoder,
                parallel_dims=parallel_dims,
                compile_config=compile_config,
            )

    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = ["dp_replicate", "efsdp"] if parallel_dims.dp_replicate_enabled else ["efsdp"]
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    if model.vision_encoder is not None:
        apply_fsdp_to_vision_encoder(
            model.vision_encoder,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
            dp_mesh_dims=dp_mesh_dims,
        )
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
    )
    return model


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


def _to_qwen_cp_metadata(mask, mesh, batch, length):
    if mask is None or isinstance(mask, QwenCPMetadata) or not hasattr(mask, "cu_seq_q"):
        return mask
    return build_sequence_metadata(mask, mesh, batch, length)


def prepare_sequence_metadata(module, args, kwargs):
    masks = kwargs.get("attention_masks")
    if masks is None:
        return args, kwargs
    mesh = module.context_parallel_mesh
    batch, length = args[0].shape[:2]
    if isinstance(masks, dict):
        # Qwen3.5 hybrid attention passes one mask dict per consumer key;
        # convert each varlen entry to rank-local QwenCPMetadata in place.
        kwargs["attention_masks"] = {
            key: _to_qwen_cp_metadata(value, mesh, batch, length) for key, value in masks.items()
        }
    elif not isinstance(masks, QwenCPMetadata):
        kwargs["attention_masks"] = build_sequence_metadata(masks, mesh, batch, length)
    return args, kwargs


def parallelize_qwen3_5_cp(model, *, parallel_dims, **kwargs):
    # Keep this guard here as well as in the public dispatcher: recipes may
    # call the CP helper directly, and VL's mixed multimodal mask path is not
    # compatible with the sequence-metadata adapter yet.
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3.5 NPU Context Parallel does not support Expert Parallel yet.")
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3.5 NPU Context Parallel does not support Tensor Parallel yet.")
    if getattr(model, "vision_encoder", None) is not None:
        raise NotImplementedError(
            "Qwen3.5-VL NPU Context Parallel is not supported for multimodal attention masks yet."
        )
    mesh = parallel_dims.get_mesh("cp")
    model.context_parallel_mesh = mesh
    model.register_forward_pre_hook(prepare_sequence_metadata, with_kwargs=True)
    for block in model.layers.values():
        module = block.attn.inner_attention if block.full_attn else block.attn
        module.context_parallel_mesh = mesh

    training, parallelism = kwargs["training"], kwargs["parallelism"]
    if parallelism.spmd_backend == "spmd_types":
        # FSDP2 with dp_mesh_dims (cp axis included when CP is enabled)
        # requires params to be DTensors on the dense storage mesh;
        # distribute them before wrapping, mirroring the dispatcher.
        model.parallelize(parallel_dims)
    compile_config, ac_config = kwargs["compile_config"], kwargs["ac_config"]
    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    ac_policy = None if ac_config is None else ac_config.build(dump_folder=kwargs["dump_folder"])
    modules = (model,) if model.vision_encoder is None else (model, model.vision_encoder)
    for module in modules:
        if ac_policy is not None:
            ac_policy.apply(module)
        if model_compile_enabled:
            _apply_qwen_compile(module, parallel_dims=parallel_dims, compile_config=compile_config)
    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    if model.vision_encoder is not None:
        apply_fsdp_to_vision_encoder(
            model.vision_encoder,
            dp_mesh,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=False,
            dp_mesh_dims=dp_mesh_dims,
        )
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        pp_enabled=False,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=1,
        edp_mesh=None,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=None,
    )
    return model


def _patch_spmd_type_annotation() -> None:
    """Let rank-local CP masks pass the spmd_types input annotation.

    ``annotate_qwen35_input_spmd_types`` asserts the "deltanet" mask is the
    global VarlenMetadata; under CP the model pre-hook already replaced the
    dict entries with rank-local QwenCPMetadata, whose cu_seqlens are ragged
    across cp ranks and carry no global SPMD type. Strip them (only for the
    annotation call) instead of failing the assert.
    """
    import functools

    try:
        from torchtitan.models.qwen3_5 import model as qwen3_5_model
        from torchtitan.models.qwen3_5 import sharding as qwen3_5_sharding
    except ImportError:
        return

    original = getattr(
        qwen3_5_sharding,
        "_annotate_qwen35_input_spmd_types_original",
        qwen3_5_sharding.annotate_qwen35_input_spmd_types,
    )
    if getattr(qwen3_5_sharding.annotate_qwen35_input_spmd_types, "_cp_patched", False):
        return
    qwen3_5_sharding._annotate_qwen35_input_spmd_types_original = original  # pyrefly: ignore [missing-attribute]

    @functools.wraps(original)
    def _annotated(*args, **kwargs):
        masks = kwargs.get("attention_masks")
        if isinstance(masks, dict) and any(isinstance(v, QwenCPMetadata) for v in masks.values()):
            local = {k: None if isinstance(v, QwenCPMetadata) else v for k, v in masks.items()}
            kwargs = dict(kwargs)
            kwargs["attention_masks"] = None if all(v is None for v in local.values()) else local
        return original(*args, **kwargs)

    _annotated._cp_patched = True  # pyrefly: ignore [missing-attribute]
    qwen3_5_sharding.annotate_qwen35_input_spmd_types = _annotated
    if getattr(qwen3_5_model, "annotate_qwen35_input_spmd_types", None) is original:
        qwen3_5_model.annotate_qwen35_input_spmd_types = _annotated


_patch_spmd_type_annotation()


__all__ = ["QwenCPMetadata", "parallelize_qwen3_5_cp", "parallelize_qwen3_5_npu"]
