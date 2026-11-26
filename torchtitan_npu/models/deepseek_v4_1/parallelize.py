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
from functools import partial, wraps

import torch
from torch.distributed.tensor import DTensor
from torch.fx.traceback import annotate_fn
from torchtitan.config import (
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder, resolve_fsdp_mesh, resolve_sparse_fsdp_mesh
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_graph_trainer_model,
    apply_simple_fsdp,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile as apply_graph_trainer_compile
from torchtitan.experiments.graph_trainer.configs import MOE_BLOCK_FQN, validate_ep_overlap_config
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    MixedPrecisionPolicy,
    ReplicateComputation,
    _register_parametrization,
)

from torchtitan_npu.config import ModelCompileConfig
from torchtitan_npu.models.common.compile_policy import configure_model_compile
from torchtitan_npu.models.common.fsdp import make_param_dtype_override_fn

from .engram.host import HostEngramTable
from .model import DeepSeekV41MultimodalModel


def validate_deepseek_v4_1_parallelism(config, *, training: TrainingConfig | None, parallelism):
    from torchtitan_npu.models.common.moe import validate_dispatcher_capacity

    if (
        parallelism.tensor_parallel_degree != 1
        or parallelism.context_parallel_degree != 1
        or parallelism.pipeline_parallel_degree != 1
    ):
        raise NotImplementedError("DeepSeek V4.1 currently supports TP=CP=PP=1 only")
    for layer in config.layers:
        if layer.engram is not None and layer.engram.table.num_embeddings % parallelism.expert_parallel_degree:
            raise ValueError("Engram physical table size must be divisible by EP degree")
    if training is None:
        raise ValueError("DeepSeek V4.1 parallelization requires a training configuration")
    validate_dispatcher_capacity(config, training=training, parallelism=parallelism)


def _make_fp32_parametrization(param_fqn, dp_mesh, placements, mode, policy, _param_name: str) -> torch.nn.Module:
    return ReplicateComputation(
        param_fqn=param_fqn,
        device_mesh=dp_mesh,
        param_sharding=placements,
        mode=mode,
        mp_policy=policy,
        non_dp_mesh_types={},
    )


def _preserve_simple_fsdp_fp32_parameters(model, *, parallelism_context, training):
    """Replace only the selected unshard policies, retaining the existing shards.

    SimpleFSDP currently exposes one policy for the whole model. Its parameter
    descriptors can be reinstalled without redistributing or reading a weight.
    The V4.1 FP32 parameters are dense (neither TP nor EP sharded).
    """
    policy = MixedPrecisionPolicy(
        param_dtype=torch.float32, reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    )
    for fqn, module in list(model.named_modules()):
        for name, parameter in module._parameters.items():
            param_fqn = f"{fqn}.{name}" if fqn else name
            if not any(fnmatchcase(param_fqn, pattern) for pattern in model.config.fp32_param_patterns):
                continue
            assert isinstance(parameter, DTensor)
            mesh = parameter.device_mesh
            mesh_names = mesh.mesh_dim_names
            assert mesh_names is not None
            dp_names = tuple(axis for axis in mesh_names if axis in {"fsdp", "dp_replicate"})
            assert dp_names and all(mesh[axis].size() == 1 for axis in mesh_names if axis not in dp_names)
            dp_mesh = mesh[dp_names]
            placements = tuple(
                placement for axis, placement in zip(mesh_names, parameter.placements, strict=True) if axis in dp_names
            )
            mode = (
                "hybrid_shard"
                if dp_mesh.ndim == 2
                else ("replicate" if parallelism_context.dp_replicate_enabled else "fully_shard")
            )
            _register_parametrization(
                module,
                [name],
                partial(_make_fp32_parametrization, param_fqn, dp_mesh, placements, mode, policy),
            )


def _chunk_v41_moe_forward(forward, fqn: str):
    region = {"chunked_region_fqn": fqn}
    split = annotate_fn({**region, "chunked_region_role": "split_boundary"})(torch.split)
    bodies = tuple(
        annotate_fn({**region, "chunked_region_role": "body", "chunk_id": chunk_id})(forward) for chunk_id in (0, 1)
    )
    cat = annotate_fn({**region, "chunked_region_role": "materialization"})(torch.cat)

    @wraps(forward)
    def chunked_forward(x, *, image_mask=None, padding_mask_T=None, aux_loss_denominator=None):
        num_tokens = x.shape[0]
        torch._check(num_tokens % 2 == 0, lambda: f"EP overlap requires an even token count in {fqn}")
        sizes = [num_tokens // 2] * 2
        hidden_chunks = split(x, sizes, dim=0)
        image_chunks = (None, None) if image_mask is None else split(image_mask, sizes, dim=0)
        padding_chunks = (None, None) if padding_mask_T is None else split(padding_mask_T, sizes, dim=0)
        return cat(
            [
                body(
                    hidden.contiguous(),
                    image_mask=None if image is None else image.contiguous(),
                    padding_mask_T=None if padding is None else padding.contiguous(),
                    aux_loss_denominator=aux_loss_denominator,
                )
                for body, hidden, image, padding in zip(
                    bodies, hidden_chunks, image_chunks, padding_chunks, strict=True
                )
            ],
            dim=0,
        )

    return chunked_forward


def parallelize_graph_trainer_deepseek_v4_1(
    model, *, parallelism_context, training, parallelism, compile_config, activation_remat, dump_folder
):
    validate_deepseek_v4_1_parallelism(model.config, training=training, parallelism=parallelism)
    for module in model.modules():
        if isinstance(module, HostEngramTable):
            module.validate_graph_training()
    if training.num_tokens_per_microbatch_per_dp_rank % parallelism_context.seq_len_divisor:
        raise ValueError("The token budget must be divisible by the sequence sharding degree")
    enabled = compile_config.ep_overlap.enabled
    module_fqn = None
    if enabled:
        _, module_fqn = validate_ep_overlap_config(compile_config.ep_overlap)
        if module_fqn != MOE_BLOCK_FQN:
            raise ValueError(f"DeepSeek V4.1 EP overlap requires module_fqn={MOE_BLOCK_FQN!r}")

    annotate_graph_trainer_model(model)
    model._parallelize(parallelism_context)
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
    if not parallelism_context.ep_enabled:
        edp_mesh, edp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
    _shard_engram_tables(model, edp_mesh=edp_mesh, edp_mesh_dims=edp_mesh_dims, training=training)
    # SimpleFSDP has no ignored-parameter API. Hide only the host storage
    # during parametrization; restore the original CPU parameters for the
    # optimizer, checkpointing and runtime lookup.
    host_weights = [(mod, mod._parameters.pop("weight")) for mod in model.modules() if isinstance(mod, HostEngramTable)]
    try:
        model = apply_simple_fsdp(model, parallelism_context=parallelism_context, training=training)
        _preserve_simple_fsdp_fp32_parameters(model, parallelism_context=parallelism_context, training=training)
    finally:
        for module, weight in host_weights:
            module.register_parameter("weight", weight)
    # V4.1 adds a per-token image mask to the upstream MoE forward contract.
    for fqn, module in model.named_modules():
        if module_fqn is not None and matches_module_fqn_pattern(module_fqn, fqn):
            module.forward = _chunk_v41_moe_forward(module.forward, fqn)
    return apply_graph_trainer_compile(model, compile_config=compile_config, parallelism_context=parallelism_context)


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
