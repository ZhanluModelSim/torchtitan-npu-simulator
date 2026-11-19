# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Any

import torch
from torch.fx.traceback import annotate_fn
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.compile import (
    apply_compile as apply_graph_trainer_compile,
)
from torchtitan.experiments.graph_trainer.configs import MOE_BLOCK_FQN, validate_ep_overlap_config

if TYPE_CHECKING:
    from torchtitan.config import TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    from torchtitan_npu.models.deepseek_v4.model import (
        DeepSeekV4Model,
        GraphTrainerDeepSeekV4Model,
    )


def _dsv4_param_dtypes(
    model: DeepSeekV4Model,
    training: TrainingConfig,
) -> dict[str, torch.dtype]:
    """Return the fqn-fp32 policy map for DSV4 SmoE hyper-connection and MoE router parameters."""

    def matches_any(fqn: str, patterns: tuple[str, ...]) -> bool:
        """Check if a module/parameter FQN matches any of the fnmatch patterns."""
        clean = ".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module")
        return any(matches_module_fqn_pattern(pattern, clean) for pattern in patterns)

    fp32_module_fqns = (
        "layers.*.hc_attn_pre",
        "layers.*.hc_ffn_pre",
        "layers.*.feed_forward.router",
        "layers.*.hc_head",
        "mtp_layers.*.hc_attn_pre",
        "mtp_layers.*.hc_ffn_pre",
        "mtp_layers.*.feed_forward.router",
        "mtp_layers.*.hc_head",
        "hc_head",
    )
    fp32_parameter_fqns = (
        "layers.*.attention.compressor.ape",
        "layers.*.attention.compressor_128.ape",
        "layers.*.attention.compressor_128.ape",
        "layers.*.attention.indexer.compressor.ape",
        "mtp_layers.*.attention.compressor.ape",
        "mtp_layers.*.attention.compressor_128.ape",
        "mtp_layers.*.attention.compressor_128.ape",
        "mtp_layers.*.attention.indexer.compressor.ape",
        # Learned per-head softmax logits: created FP32 in attention.py and
        # required FP32 by the attention kernels.
        "layers.*.attention.attn_sink",
        "layers.*.attention.attn_sink.weight",
        "mtp_layers.*.attention.attn_sink",
        "mtp_layers.*.attention.attn_sink.weight",
    )

    fp32_modules = [module for name, module in model.named_modules() if matches_any(name, fp32_module_fqns)]
    fp32_param_ids = {id(param) for module in fp32_modules for param in module.parameters()}

    overrides: dict[str, torch.dtype] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in fp32_param_ids or matches_any(name, fp32_parameter_fqns):
            overrides[name] = torch.float32

    return overrides


def parallelize_deepseek_v4(model, **kwargs):
    """Apply upstream eager parallelization with DSV4 FP32 parameter policies."""
    from torchtitan.models.base.model import BaseModel

    return BaseModel.parallelize(model, **kwargs)


def annotate_deepseek_v4(model: GraphTrainerDeepSeekV4Model) -> None:
    """Attach annotations to FX graph nodes for DeepSeek V4.

    - Expert Parallel (EP) annotations: Tags "dispatch", "combine", and "compute"
      regions in MoE for debugging purposes.
    - Module FQN annotation: Tags each submodule's forward with its
      fully-qualified name for downstream passes (bucketing, SAC region
      boundaries, etc.).
    """
    annotate_moe_ep_regions()
    annotate_module_fqns(model)


def _chunk_dsv4_moe_forward(forward, fqn: str):
    region = {"chunked_region_fqn": fqn}
    split = annotate_fn({**region, "chunked_region_role": "split_boundary"})(torch.split)
    bodies = tuple(
        annotate_fn({**region, "chunked_region_role": "body", "chunk_id": chunk_id})(forward) for chunk_id in (0, 1)
    )
    cat = annotate_fn({**region, "chunked_region_role": "materialization"})(torch.cat)

    @wraps(forward)
    def chunked_forward(x, *, input_ids_T=None, padding_mask_T=None, aux_loss_denominator=None):
        # DSV4's hash router needs token IDs split at the same boundary as x.
        num_tokens = x.shape[0]
        torch._check(num_tokens % 2 == 0, lambda: f"EP overlap requires an even token count in {fqn}")
        sizes = [num_tokens // 2] * 2
        hidden_chunks = split(x, sizes, dim=0)
        id_chunks = (None, None) if input_ids_T is None else split(input_ids_T, sizes, dim=0)
        mask_chunks = (None, None) if padding_mask_T is None else split(padding_mask_T, sizes, dim=0)
        return cat(
            [
                body(
                    hidden.contiguous(),
                    **({} if ids is None else {"input_ids_T": ids.contiguous()}),
                    padding_mask_T=None if mask is None else mask.contiguous(),
                    aux_loss_denominator=aux_loss_denominator,
                )
                for body, hidden, ids, mask in zip(bodies, hidden_chunks, id_chunks, mask_chunks, strict=True)
            ],
            dim=0,
        )

    return chunked_forward


def parallelize_graph_trainer_deepseek_v4(
    model: GraphTrainerDeepSeekV4Model,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    activation_remat: Any,
    dump_folder: str,
):
    token_divisor = parallelism_context.cp * (
        parallelism_context.tp if parallelism_context.enable_sequence_parallel else 1
    )
    token_budget = training.num_tokens_per_microbatch_per_dp_rank
    assert token_budget % token_divisor == 0, (
        f"Token budget {token_budget} must be divisible by the activation sharding degree {token_divisor}."
    )

    # DeepSeek V4 sparse attention does not yet support context parallelism.
    if parallelism_context.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for DeepSeek V4 sparse attention in the GraphTrainer path."
        )

    annotate_deepseek_v4(model)

    # SPMD declarations must be resolved even when every model-parallel axis is
    # singleton; the non-graph path applies the same unconditional lifecycle.
    with parallelism_context.activate_spmd():
        model._parallelize(parallelism_context)

    # Apply simple_fsdp unconditionally. The ``fsdp`` mesh always exists with a
    # real backend (see ParallelismContext._mesh_exist), even at degree 1, so that
    # MixedPrecisionPolicy's param_dtype cast still applies in single-GPU runs.
    # pyrefly: ignore [bad-assignment]
    model = apply_simple_fsdp(model, parallelism_context=parallelism_context, training=training)

    if compile_config.ep_overlap.enabled:
        _, module_fqn = validate_ep_overlap_config(compile_config.ep_overlap)
        if module_fqn != MOE_BLOCK_FQN:
            raise ValueError(f"DeepSeek-V4 EP overlap requires module_fqn={MOE_BLOCK_FQN!r}")
        for fqn, module in model.named_modules():
            if matches_module_fqn_pattern(module_fqn, fqn):
                module.forward = _chunk_dsv4_moe_forward(module.forward, fqn)  # pyrefly: ignore [bad-override]

    # Apply compilation based on mode
    # pyrefly: ignore [bad-assignment]
    model = apply_graph_trainer_compile(
        model,
        compile_config=compile_config,
        parallelism_context=parallelism_context,
    )

    return model
