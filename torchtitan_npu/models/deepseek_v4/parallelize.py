# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.compile import (
    apply_compile as apply_graph_trainer_compile,
)

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


def parallelize_graph_trainer_deepseek_v4(
    model: GraphTrainerDeepSeekV4Model,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: Any,
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

    # Apply compilation based on mode
    # pyrefly: ignore [bad-assignment]
    model = apply_graph_trainer_compile(
        model,
        compile_config=compile_config,
        parallelism_context=parallelism_context,
    )

    return model
