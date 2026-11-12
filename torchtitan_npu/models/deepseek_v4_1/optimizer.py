# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass

import torch
from torchtitan.components.optim import AdamW, BaseOptimizer
from torchtitan.distributed.flex_shard import (
    BlockShard,
    BucketConfig,
    ComputeLayout,
)
from torchtitan.distributed.parallelism_context import MeshAxisName

from torchtitan_npu.config import (
    MuonOptimizerProfile,
    OptimizerConfig,
)
from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
    HostSparseOptimizersContainer,
    SparseAdam,
)
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts
from torchtitan_npu.models.deepseek_v4_1.model import DeepSeekV41Model


@dataclass(kw_only=True)
class EngramTableOptimizerConfig(SparseAdam.Config):
    """The recipe-owned table group, distinct from user-defined groups."""


def _v41_muon_profile(model: DeepSeekV41Model.Config) -> MuonOptimizerProfile:
    """Build the V4.1-owned parameter and FlexShard policy for Muon.

    Same sharding policy as the DSV4 profile where the parameter spaces
    overlap (per-head ``wq_b``/``wo_a`` BlockShard, expert-sharded routed
    GEMMs, Owned for the rest).  V4.1 differences: no MTP depths, no
    ``hc_head``, no compressor APE.  The indexer parameters do receive LI
    gradients on the new baseline, but they stay on the AdamW fallback group
    together with the vision tower and every 1-D parameter — this profile
    does not change the parameter policy the verified stacks ran with.
    """
    model_config = model
    # The target upstream names the dense parameter storage axis dp_shard.
    dense_dp_axes = (MeshAxisName.DP_SHARD.value,)
    owned = make_owned_layout(dense_dp_axes)
    owned_attention_projections = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_projections = (*owned_attention_projections, "wq_b", "wo_a")
    expert_projections = ("w13", "w2")
    routed_expert_projections = ("w13.weight", "w2.weight")
    compressor_projections = ("wkv", "wgate")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    expert_sharding = make_expert_layout(dense_dp_axes)

    compute_sharding_by_fqn: dict[str, ComputeLayout] = {}
    bucket_configs: list[BucketConfig] = []
    for layer_id, layer_config in enumerate(model_config.layers):
        attention = layer_config.attention
        attention_shardings = {
            f"layers.{layer_id}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in owned_attention_projections.items()
        }
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head and
        # ``wo_a`` one [o_lora_rank, per_group_in] matrix per group, both
        # flattened on dim 0; BlockShard computes Muon per matrix.
        attention_shardings[f"layers.{layer_id}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.head_dim,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        attention_shardings[f"layers.{layer_id}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.wo_a.out_features,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        # Every layer carries a compressor; only the sources own weights.
        for projection in compressor_projections:
            if getattr(attention.compressor, projection, None) is not None:
                attention_shardings[f"layers.{layer_id}.attention.compressor.{projection}.weight"] = owned
        attention_shardings[f"layers.{layer_id}.hc_attn_pre.hc_fn"] = owned
        dense_shardings = {
            f"layers.{layer_id}.feed_forward.shared_experts.{projection}.weight": owned
            for projection in expert_projections
        }
        dense_shardings[f"layers.{layer_id}.feed_forward.router.gate.weight"] = owned
        dense_shardings[f"layers.{layer_id}.hc_ffn_pre.hc_fn"] = owned
        if layer_config.engram is not None:
            # q_weight/k_weight are normalization scales, not projections.
            dense_shardings[f"layers.{layer_id}.engram.gate.wkv"] = owned
        routed_shardings = {
            f"layers.{layer_id}.feed_forward.routed_experts.{projection}": expert_sharding
            for projection in routed_expert_projections
        }
        # Split each layer into three buckets: the routed-expert GEMMs are
        # by far the largest and get a bucket of their own so the
        # Newton-Schulz all-gather peak stays bounded on the 40-layer
        # FSDP8 shape (a single per-layer bucket OOMs at ~89% HBM).
        compute_sharding_by_fqn.update(attention_shardings)
        compute_sharding_by_fqn.update(dense_shardings)
        compute_sharding_by_fqn.update(routed_shardings)
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.attn", patterns=tuple(attention_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.dense", patterns=tuple(dense_shardings)))
        bucket_configs.append(BucketConfig(name=f"layers.{layer_id}.routed", patterns=tuple(routed_shardings)))

    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.compressor\.(?:{'|'.join(compressor_projections)})\.weight|"
        rf"feed_forward\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"feed_forward\.routed_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"feed_forward\.router\.gate\.weight|"
        r"engram\.gate\.wkv|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn" + r")$"
    )
    from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon

    profile = MuonOptimizerProfile(
        optimizer_config_type=DistMuon.Config,
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": tuple(bucket_configs),
            }
        },
    )
    return with_lora_compute_layouts(model, profile)


def _v41_optimizer_config(model: DeepSeekV41Model.Config) -> OptimizerConfig:
    has_engram = any(layer.engram is not None for layer in model.layers)
    optimizer_type = HostSparseOptimizersContainer.Config if has_engram else OptimizerConfig
    profile = _v41_muon_profile(model)
    optimizers: list[BaseOptimizer.Config] = [AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-6)]
    if has_engram:
        optimizers.insert(
            0,
            EngramTableOptimizerConfig(
                pattern=r".*\.engram\.table\.weight$",
                lr=5e-5,
                betas=(0.9, 0.95),
                eps=1e-6,
            ),
        )
    return optimizer_type(
        name="AdamW" if has_engram else "Muon",
        _optimizer_defaults={
            "AdamW": optimizers,
            "Muon": [
                profile.optimizer_config_type(
                    pattern=profile.muon_pattern,
                    lr=1e-5,
                    adjust_lr_fn="match_rms_adamw",
                    **profile.optimizer_factory_kwargs["DistMuon"],
                ),
                AdamW.Config(pattern=r".*", lr=1e-5, eps=1e-8, fused=True, foreach=False),
            ],
        },
        _muon_profile=profile,
    )
