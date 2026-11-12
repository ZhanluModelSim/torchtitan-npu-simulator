# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4.1 FSDP assembly, Host Engram exclusion and model compile policy."""

from fnmatch import fnmatchcase
from functools import partial

import torch
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh, resolve_sparse_fsdp_mesh

from torchtitan_npu.config import ModelCompileConfig
from torchtitan_npu.models.common.compile_policy import configure_model_compile
from torchtitan_npu.models.common.fsdp import make_param_dtype_override_fn

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
            module.wire_sparse_grad_replicas(edp_mesh=edp_mesh, edp_mesh_dims=edp_mesh_dims)
            init_buffer = getattr(module, "init_elastic_buffer", None)
            if init_buffer is not None:
                init_buffer(param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param])
    return ignored_params


def _apply_compile_v4_1(model, *, compile_config: ModelCompileConfig) -> None:
    """V4.1-specific compile assembly around the shared torchtitan semantics.

    The dynamo flags below mirror torchtitan's apply_compile, which V4.1
    cannot call directly: it hardcodes fullgraph=True and leaves dynamic to
    auto-detection, while V4.1 needs static shapes (dynamic=False) and, under
    data-dependent MoE routing, a graph break around the dispatcher
    communication.  Once apply_compile grows fullgraph/dynamic parameters
    this collapses back to a plain call.  The regional-inductor helper is
    intentionally not used: V4.1's sparse core is the gather_attn
    reference implementation (no FlexInnerAttention), for which it is the
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
    import logging

    import torch

    logger = logging.getLogger(__name__)

    configure_model_compile()

    # Traceable reorder opt-in for the compiled run (see the override module).
    from torchtitan_npu.override.common.token_dispatcher import (
        configure_dispatchers_for_compile,
    )

    # Force-load-balanced MoE routing (round-robin) keeps the all_to_all split
    # sizes constant (T*K/E per rank on every step), so inductor can trace the
    # whole dispatcher inline with fullgraph=True. With data-dependent routing
    # those sizes vary per step while inductor bakes the traced values as
    # constants (the Eq(sum, T*K) guard and the split-size runtime failure), so
    # the dispatcher's dispatch/combine must instead run eagerly outside the
    # graph (fullgraph=False). aot_eager replays the aten ops verbatim and always
    # keeps fullgraph=True.
    # TODO(compile): Restore force-balanced detection when model compile is
    # re-enabled. Upstream v0.4.0 removed --debug.moe-force-load-balance and the
    # router hook; the replacement is RoundRobinTokenChoiceTopKRouter, which the
    # V4/V4.1 recipes must select explicitly. Until then treat routing as
    # data-dependent for inductor.
    force_balanced_moe = False
    fullgraph = compile_config.backend != "inductor" or force_balanced_moe
    configure_dispatchers_for_compile(
        model, eager_all_to_all=compile_config.backend == "inductor" and not force_balanced_moe
    )
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


def apply_deepseek_v4_1_fsdp(model, *, parallelism_context, training, parallelism):
    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
    # The vision tower is wrapped as one unit before the decoder wrapper sees it; a text
    # stack has no such submodule and goes straight to the decoder wrapper.
    if isinstance(model, DeepSeekV41MultimodalModel):
        model.apply_fsdp_extensions(
            dp_mesh=dp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            training=training,
            parallelism=parallelism,
            parallelism_context=parallelism_context,
        )

    ignored_params = _shard_engram_tables(
        model,
        edp_mesh=edp_mesh if parallelism_context.ep_enabled else dp_mesh,
        edp_mesh_dims=edp_mesh_dims if parallelism_context.ep_enabled else dp_mesh_dims,
        training=training,
    )
    apply_fsdp = apply_fsdp_to_decoder
    if ignored_params:
        from torchtitan_npu.extensions.torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder as apply_fsdp_with_ignored_params,
        )

        apply_fsdp = partial(apply_fsdp_with_ignored_params, ignored_params=ignored_params)
    parameter_dtypes = {}
    for name, _ in model.named_parameters(remove_duplicate=False):
        clean = ".".join(part for part in name.split(".") if part != "_checkpoint_wrapped_module")
        if any(fnmatchcase(clean, pattern) for pattern in model.config.fp32_param_patterns):
            parameter_dtypes[name] = torch.float32
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallelism_context.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallelism_context.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        symm_mem_scope=parallelism.fsdp_symm_mem_scope,
        param_dtype_override_fn=make_param_dtype_override_fn(model, parameter_dtypes) if parameter_dtypes else None,
    )
