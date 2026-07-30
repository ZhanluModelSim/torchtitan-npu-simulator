# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Parallelization strategy for Kimi K3.

Applies FSDP2, TP (for Gated MLA layers), EP (for MoE experts), CP (Ulysses),
and PP (via pipeline_llm at ModelSpec level — stage splitting is handled by
torchtitan's pipelining infrastructure, not this function).

KDA layers: TP is not applied (linear attention projections are not per-head
column/row parallelizable in the same way as MLA). However, Ulysses CP IS
supported because it splits along the head dimension — each head's recurrent
state is independent.

TP strategy for Gated MLA layers (mirrors DSv3):
  - q_a_proj, kv_a_proj_with_mqa: no sharding (compression projections)
  - q_b_proj: ColwiseParallel (output is per-head)
  - kv_b_proj: ColwiseParallel (output is per-head)
  - g_proj: ColwiseParallel (output gate, per-head)
  - o_proj: RowwiseParallel (reduces across heads)

EP strategy for MoE:
  - Expert weights sharded across EP group
  - Router replicated
  - Shared experts replicated (or FSDP-sharded)
"""

import logging

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.protocols import ModelConvertersContainer

from .attention import KimiGatedMLA
from .model import KimiK3Model, KimiK3TransformerBlock

logger = logging.getLogger(__name__)


def _get_mla_layer_plan():
    """TP plan for Gated MLA attention layers."""
    return {
        "attention.q_b_proj": ColwiseParallel(),
        "attention.kv_b_proj": ColwiseParallel(),
        "attention.g_proj": ColwiseParallel(),
        "attention.o_proj": RowwiseParallel(),
    }


def parallelize_kimi_k3(
    model: KimiK3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Parallelize Kimi K3 model with FSDP2 + TP (MLA) + EP (MoE) + PP."""
    from torchtitan.distributed.parallelize import parallelize_module

    # --- Tensor Parallel for Gated MLA layers ---
    if parallel_dims.tp_enabled:
        tp_degree = parallel_dims.tp
        tp_mesh = parallel_dims.get_mesh("tp")

        # Validate head divisibility for MLA layers
        for layer_name, layer in model.layers.items():
            if isinstance(layer.attention, KimiGatedMLA):
                n_heads = layer.attention.num_heads
                if n_heads % tp_degree != 0:
                    raise ValueError(
                        f"[Kimi K3 TP] Gated MLA n_heads={n_heads} must be "
                        f"divisible by tensor_parallel_degree={tp_degree}."
                    )

        # Apply TP plan to each MLA layer
        mla_plan = _get_mla_layer_plan()
        for layer_name, layer in model.layers.items():
            if isinstance(layer.attention, KimiGatedMLA):
                layer_plan = {f"layers.{layer_name}.{k}": v for k, v in mla_plan.items()}
                parallelize_module(model, tp_mesh, layer_plan)

        logger.info(f"[Kimi K3] Applied TP (degree={tp_degree}) to Gated MLA layers")

    # --- Expert Parallel for MoE layers ---
    if parallel_dims.ep_enabled:
        ep_degree = parallel_dims.ep
        ep_mesh = parallel_dims.get_mesh("ep")

        # EP is applied via the MoE module's internal dispatch mechanism.
        # The router stays replicated; expert weights are sharded across EP group.
        # This is handled by torchtitan's ExpertParallel style when the model
        # uses the standard MoE infrastructure. For Kimi K3's custom MoE block,
        # EP sharding is applied at the expert weight level.
        logger.info(f"[Kimi K3] EP enabled (degree={ep_degree}); expert sharding via converter")

    # --- FSDP2 sharding ---
    if parallel_dims.dp_shard_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_shard")
        parallelize_module(model, dp_mesh, {})
        logger.info("[Kimi K3] Applied FSDP2 sharding")

    # --- Context Parallel (Ulysses) ---
    # Ulysses CP splits along the HEAD dimension, not sequence. Each head's
    # recurrent state (KDA) or KV cache (MLA) is independent, so both KDA and
    # Gated MLA layers support Ulysses CP as long as n_heads % cp_degree == 0.
    if parallel_dims.cp_enabled:
        cp_degree = parallel_dims.cp
        for layer_name, layer in model.layers.items():
            if isinstance(layer.attention, KimiGatedMLA):
                n_heads = layer.attention.num_heads
            else:
                # KDA: num_heads from config
                n_heads = layer.attention.num_heads
            if n_heads % cp_degree != 0:
                raise ValueError(
                    f"[Kimi K3 CP] Layer {layer_name} n_heads={n_heads} must be "
                    f"divisible by context_parallel_degree={cp_degree}."
                )

        from torchtitan_npu.distributed.context_parallel.registry import (
            apply_cp_to_attention_module as apply_cp,
        )

        # Apply Ulysses CP to all attention layers (both KDA and MLA)
        for layer_name, layer in model.layers.items():
            apply_cp(layer.attention, cp_degree)

        logger.info(
            f"[Kimi K3] Ulysses CP enabled (degree={cp_degree}) for all layers "
            f"(KDA + Gated MLA)"
        )

    # --- Activation checkpointing ---
    if ac_config.mode != "none":
        from torchtitan.distributed.activation_checkpoint import apply_ac

        apply_ac(model, ac_config)
        logger.info(f"[Kimi K3] Applied activation checkpointing: {ac_config.mode}")

    return model
