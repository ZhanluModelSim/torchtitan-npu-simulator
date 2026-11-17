# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    annotate_moe_ep_regions,
    apply_simple_fsdp,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.compile import (
    apply_compile as apply_graph_trainer_compile,
)
from torchtitan.models.deepseek_v3.parallelize import parallelize_deepseekv3

from torchtitan_npu.patches.torch.distributed.fsdp import policy_overrides

from .lora import is_lora_parameter

if TYPE_CHECKING:
    from torchtitan.config import CompileConfig, ParallelismConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
    from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

    from torchtitan_npu.models.deepseek_v4.model import (
        DeepSeekV4Model,
        GraphTrainerDeepSeekV4Model,
    )


def _dsv4_fp32_overrides(
    model: DeepSeekV4Model,
    training: TrainingConfig,
) -> dict[str, MixedPrecisionPolicy]:
    """Return the fqn-fp32 policy map for DSV4 SmoE hyper-connection and MoE router parameters."""
    default = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        output_dtype=None,
        cast_forward_inputs=False,
    )
    fp32 = replace(default, param_dtype=torch.float32)

    def matches_any(fqn: str, patterns: tuple[str, ...]) -> bool:
        """Check if a module/parameter FQN matches any of the fnmatch patterns."""
        clean = ".".join(part for part in fqn.split(".") if part != "_checkpoint_wrapped_module")
        return any(matches_module_fqn_pattern(pattern, clean) for pattern in patterns)

    fp32_module_fqns = (
        "layers.*.hc_attn_pre",
        "layers.*.hc_ffn_pre",
        "layers.*.moe.router",
        "layers.*.hc_head",
        "mtp_layers.*.hc_attn_pre",
        "mtp_layers.*.hc_ffn_pre",
        "mtp_layers.*.moe.router",
        "mtp_layers.*.hc_head",
        "hc_head",
    )
    fp32_parameter_fqns = (
        "layers.*.attention.compressor.ape",
        "layers.*.attention.indexer.compressor.ape",
        "mtp_layers.*.attention.compressor.ape",
        "mtp_layers.*.attention.indexer.compressor.ape",
        # Learned per-head softmax logits: created FP32 in attention.py and
        # required FP32 by the attention kernels.
        "layers.*.attention.attn_sink",
        "mtp_layers.*.attention.attn_sink",
    )

    fp32_modules = [module for name, module in model.named_modules() if matches_any(name, fp32_module_fqns)]
    fp32_param_ids = {id(param) for module in fp32_modules for param in module.parameters()}

    overrides: dict[str, MixedPrecisionPolicy] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in fp32_param_ids or matches_any(name, fp32_parameter_fqns):
            overrides[name] = fp32

    return overrides


def parallelize_deepseek_v4(
    model: DeepSeekV4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply DSV4 parallelism with per-parameter FP32 SmoE policies."""
    with policy_overrides(_dsv4_fp32_overrides(model, training)):
        parallelized_model = parallelize_deepseekv3(
            model,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
    if getattr(parallelized_model, "lora_config", None) is not None:
        for name, parameter in parallelized_model.named_parameters():
            if not is_lora_parameter(name):
                parameter.requires_grad_(False)
    return parallelized_model


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
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    # The graph_trainer simple_fsdp wrapper is built on raw DTensor ops
    # (``distribute_tensor`` / ``tensor._spec``) and errors under the
    # spmd_types backend; only the eager path supports that backend for now.
    assert get_spmd_backend() != "spmd_types", "The GraphTrainer path does not yet support the spmd_types backend."

    # TP currently cannot handle uneven seq_len because we set
    # ``use_local_output=True`` to use plain Tensors for legacy reasons.
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    # DeepSeek V4 sparse attention does not yet support context parallelism.
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not yet supported for DeepSeek V4 sparse attention in the GraphTrainer path."
        )

    annotate_deepseek_v4(model)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)

    # Apply simple_fsdp unconditionally. The ``fsdp`` mesh always exists with a
    # real backend (see ParallelDims._mesh_exist), even at degree 1, so that
    # MixedPrecisionPolicy's param_dtype cast still applies in single-GPU runs.
    # pyrefly: ignore [bad-assignment]
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

    # Apply compilation based on mode
    # pyrefly: ignore [bad-assignment]
    model = apply_graph_trainer_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
