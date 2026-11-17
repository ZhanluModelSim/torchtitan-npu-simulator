# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 parallelization: EP/FSDP/FullAC assembly (CP1-only).

Derived from the DSV4 policy with the CP/MTP branches removed: V4.1
rejects CP>1 and PP>1 in its config, so this assembly wires only what the
supported matrix needs.  The decoder FSDP wrapper is the generic (non-MTP)
helper.  torch.compile is supported (see _apply_compile_v4_1).
"""

import torch
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.full_dtensor import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh, validate_config

from torchtitan_npu.extensions.torchtitan.distributed.fsdp import apply_fsdp_to_decoder

from .engram.host import HostEngramTable
from .model import DeepSeekV41MultimodalModel


def _shard_engram_tables(
    model,
    *,
    edp_mesh,
    edp_mesh_dims,
    training: TrainingConfig,
) -> set[torch.nn.Parameter]:
    """Exclude CPU tables from FSDP and initialize their replica/backend state."""
    assert edp_mesh is not None
    ignored_params: set[torch.nn.Parameter] = set()
    for module in model.modules():
        if isinstance(module, HostEngramTable):
            ignored_params.add(module.weight)
            # A table FSDP does not manage still has replicas along the sparse
            # data-parallel axes, and their gradients have to be summed.
            wire_replicas = getattr(module, "wire_sparse_grad_replicas", None)
            if wire_replicas is not None:
                wire_replicas(edp_mesh=edp_mesh, edp_mesh_dims=edp_mesh_dims)
            init_buffer = getattr(module, "init_elastic_buffer", None)
            if init_buffer is not None:
                init_buffer(param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param])
    return ignored_params


def _apply_compile_v4_1(model, *, compile_config: CompileConfig) -> None:
    """V4.1-specific compile assembly around the shared torchtitan semantics.

    The dynamo flags below mirror torchtitan's apply_compile, which V4.1
    cannot call directly: it hardcodes fullgraph=True and leaves dynamic to
    auto-detection, while V4.1 needs static shapes (dynamic=False) and, under
    data-dependent MoE routing, a graph break around the dispatcher
    communication.  Once apply_compile grows fullgraph/dynamic parameters
    this collapses back to a plain call.  The regional-inductor helper is
    intentionally not used: V4.1's sparse core is the gather_attn
    reference implementation (no FlexAttention), for which it is the
    identity.  The shared per-block code object plus role-level graphs keep
    the recompile count within the stock recompile_limit.

    V4.1-specific pieces beyond the shared semantics:
    - dynamic=False: shapes are static for a fixed seq_len; dynamic
      re-tracing is neither expected nor supported.
    - MoE routing branch: with force-load-balanced routing the all_to_all
      split sizes are constant, so inductor keeps the whole graph
      (fullgraph=True, communication included); with data-dependent routing
      the sizes vary per step while inductor bakes the traced values as
      constants (the Eq(sum, T*K) guard and the split-size runtime
      failure), so the dispatcher's dispatch/combine run eagerly outside
      the graph (fullgraph=False) instead.
    - The dispatcher reorder opt-in: aten argsort does not survive dynamo
      fake-eval under the spmd_types patch stack, so the traceable exact
      stable formulation is swapped in for the compiled run only (eager
      keeps the upstream helper; the permutation is identical).
    """
    import torch
    import torch._dynamo.config
    from torchtitan.tools.logging import logger

    # Mirrors torchtitan.distributed.compile.apply_compile; keep in sync.
    # capture_scalar_outputs: data-dependent dynamic shapes in token-choice
    # MoE dispatch (unbacked SymInts for the expert segments).
    torch._dynamo.config.capture_scalar_outputs = True
    # Skip replaying forward side effects (e.g. RoPE cache updates) during
    # the AC recompute in backward; eager AC replays them, compiled does not.
    setattr(torch._dynamo.config, "skip_fwd_side_effects_in_bwd_under_checkpoint", True)  # noqa: B010

    # Traceable reorder opt-in for the compiled run (see the override module).
    from torchtitan_npu.override.common.token_dispatcher import (
        apply_compile_friendly_local_reorder,
    )

    apply_compile_friendly_local_reorder()

    # Force-load-balanced MoE routing (round-robin, --debug.moe-force-load-balance)
    # keeps the all_to_all split sizes constant (T*K/E per rank on every step),
    # so inductor can trace the whole dispatcher inline with fullgraph=True.
    # With data-dependent routing those sizes vary per step while inductor
    # bakes the traced values as constants (the Eq(sum, T*K) guard and the
    # split-size runtime failure), so the dispatcher's dispatch/combine must
    # instead run eagerly outside the graph (fullgraph=False).  aot_eager
    # replays the aten ops verbatim and always keeps fullgraph=True.
    routers = [m for m in model.modules() if hasattr(m, "_debug_force_load_balance")]
    force_balanced_moe = bool(routers) and all(m._debug_force_load_balance for m in routers)
    fullgraph = compile_config.backend != "inductor" or force_balanced_moe
    if compile_config.backend == "inductor":
        import torch._inductor.config

        # Skip the unbacked-SymInt runtime assertions (Eq(u32+..., T*K))
        # that inductor derives from the tracing run's expert distribution.
        setattr(torch._inductor.config, "do_not_emit_runtime_assertions", True)  # noqa: B010
        if not force_balanced_moe:
            # The MoE communication (all_to_all_single with data-dependent
            # splits) runs eagerly: wrap the NPU dispatcher's dispatch/combine
            # so dynamo treats them as opaque and graph-breaks around them.
            # aot_eager keeps fullgraph=True and needs no wrapping.
            from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher

            AscAllToAllTokenDispatcher.dispatch = torch._dynamo.disable(recursive=True)(
                AscAllToAllTokenDispatcher.dispatch
            )
            AscAllToAllTokenDispatcher.combine = torch._dynamo.disable(recursive=True)(
                AscAllToAllTokenDispatcher.combine
            )

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block.compile(backend=compile_config.backend, fullgraph=fullgraph, dynamic=False)

    logger.info(
        f"Compiling each V4.1 TransformerBlock with torch.compile "
        f"(fullgraph={fullgraph}, force_balanced_moe={force_balanced_moe})"
    )


def apply_activation_checkpointing(model, ac_config, dump_folder):
    """Apply the selected policy plus any model-specific extension blocks.

    Only the multimodal stack has blocks outside the decoder layer list; the text stack
    is fully covered by the policy's own walk, so it takes no second call.
    """
    policy = ac_config.build(dump_folder=dump_folder)
    policy.apply(model)
    if isinstance(model, DeepSeekV41MultimodalModel):
        model.apply_activation_checkpointing_extensions(policy)


def parallelize_deepseek_v4_1(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Parallelize V4.1: sharding config, AC, then the decoder FSDP wrap."""
    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    elif parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    if ac_config is not None:
        apply_activation_checkpointing(model, ac_config, dump_folder)

    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    if model_compile_enabled:
        _apply_compile_v4_1(model, compile_config=compile_config)

    if parallelism.spmd_backend in ("full_dtensor", "spmd_types"):
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

    # The vision tower is wrapped as one unit before the decoder wrapper sees it; a text
    # stack has no such submodule and goes straight to the decoder wrapper.
    if isinstance(model, DeepSeekV41MultimodalModel):
        model.apply_fsdp_extensions(
            dp_mesh=dp_mesh,
            training=training,
            parallelism=parallelism,
            parallel_dims=parallel_dims,
        )

    ignored_params = _shard_engram_tables(
        model,
        edp_mesh=edp_mesh if parallel_dims.ep_enabled else dp_mesh,
        edp_mesh_dims=edp_mesh_dims if parallel_dims.ep_enabled else dp_mesh_dims,
        training=training,
    )
    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        ignored_params=ignored_params,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    if getattr(model, "lora_config", None) is not None:
        # Keep ordinary V4.1 imports independent of the optional V4 LoRA implementation.
        from torchtitan_npu.models.deepseek_v4.lora import is_lora_parameter

        for name, parameter in model.named_parameters():
            parameter.requires_grad_(is_lora_parameter(name))
    return model
