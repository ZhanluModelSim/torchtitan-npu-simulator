# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torchtitan.components.optim import AdamW
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
from torchtitan_npu.models.common.muon import make_expert_layout, make_owned_layout, with_lora_compute_layouts

if TYPE_CHECKING:
    from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model


def _dsv4_muon_profile(model: DeepSeekV4Model.Config) -> MuonOptimizerProfile:
    """Build the DSV4-owned parameter and FlexShard policy for Muon."""
    model_config = model
    dense_dp_axes = (MeshAxisName.DP_SHARD.value,)
    owned = make_owned_layout(dense_dp_axes)
    attention_shardings = {"wq_a": owned, "wkv": owned, "wo_b": owned}
    attention_per_head_projections = ("wq_b", "wo_a")
    attention_projections = tuple(attention_shardings) + attention_per_head_projections
    expert_projections = ("w13", "w2")
    routed_expert_projections = ("w13.weight", "w2.weight")
    compressor_projections = ("wkv", "wgate")
    indexer_projections = ("wq_b", "weights_proj")
    hc_pre_modules = ("hc_attn_pre", "hc_ffn_pre")
    mtp_projections = ("e_proj", "h_proj")

    def compute_shardings_for_transformer_layer(
        prefix: str,
        layer_config: Any,
        *,
        include_mtp_projections: bool,
    ) -> dict[str, ComputeLayout]:
        attention = layer_config.attention
        shardings = {
            f"{prefix}.attention.{projection}.weight": compute_sharding
            for projection, compute_sharding in attention_shardings.items()
        }
        shardings.update(
            {f"{prefix}.feed_forward.shared_experts.{projection}.weight": owned for projection in expert_projections}
        )
        # ``wq_b`` stores one [head_dim, q_lora_rank] matrix per head flattened
        # into [n_heads * head_dim, q_lora_rank]; ``wo_a`` stores one
        # [o_lora_rank, per_group_in] matrix per group flattened likewise.
        # BlockShard computes Muon on each per-head/per-group matrix.
        shardings[f"{prefix}.attention.wq_b.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.head_dim,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        shardings[f"{prefix}.attention.wo_a.weight"] = ComputeLayout(
            shardings_by_mesh_axis={
                axis: BlockShard(dim=0, block_sizes=(attention.o_lora_rank,)) for axis in dense_dp_axes
            },
            compute_dtype=torch.bfloat16,
        )
        for compressor_name in ("compressor", "compressor_128"):
            if getattr(attention, compressor_name, None) is not None:
                for projection in compressor_projections:
                    shardings[f"{prefix}.attention.{compressor_name}.{projection}.weight"] = owned
                shardings[f"{prefix}.attention.{compressor_name}.ape"] = owned
        expert_sharding = make_expert_layout(dense_dp_axes)
        for projection in routed_expert_projections:
            shardings[f"{prefix}.feed_forward.routed_experts.{projection}"] = expert_sharding
        shardings[f"{prefix}.feed_forward.router.gate.weight"] = owned
        for module in hc_pre_modules:
            shardings[f"{prefix}.{module}.hc_fn"] = owned
        indexer = layer_config.attention.indexer
        if indexer is not None:
            shardings[f"{prefix}.attention.indexer.wq_b.weight"] = ComputeLayout(
                shardings_by_mesh_axis={
                    axis: BlockShard(
                        dim=0,
                        block_sizes=(indexer.index_head_dim,),
                    )
                    for axis in dense_dp_axes
                },
                compute_dtype=torch.bfloat16,
            )
            shardings[f"{prefix}.attention.indexer.weights_proj.weight"] = owned
            for projection in compressor_projections:
                shardings[f"{prefix}.attention.indexer.compressor.{projection}.weight"] = owned
            shardings[f"{prefix}.attention.indexer.compressor.ape"] = owned
        if include_mtp_projections:
            for projection in mtp_projections:
                shardings[f"{prefix}.{projection}.weight"] = owned
            shardings[f"{prefix}.hc_head.hc_fn"] = owned
        return shardings

    main_layer_shardings = tuple(
        compute_shardings_for_transformer_layer(
            f"layers.{layer_id}",
            model_config.layers[layer_id],
            include_mtp_projections=False,
        )
        for layer_id in range(model_config.n_layers)
    )
    mtp_layer_shardings = tuple(
        compute_shardings_for_transformer_layer(
            f"mtp_layers.{layer_id}",
            layer_config,
            include_mtp_projections=True,
        )
        for layer_id, layer_config in enumerate(model_config.mtp_layers or ())
    )
    per_layer = main_layer_shardings + mtp_layer_shardings
    compute_sharding_by_fqn = {}
    for layer_shardings in per_layer:
        compute_sharding_by_fqn.update(layer_shardings)
    compute_sharding_by_fqn["hc_head.hc_fn"] = owned
    routed_expert_marker = ".feed_forward.routed_experts."

    def bucket_configs_for_layers(
        prefix: str,
        layer_shardings: tuple[dict[str, ComputeLayout], ...],
    ) -> tuple[BucketConfig, ...]:
        layer_bucket_configs = []
        for layer_id, shardings in enumerate(layer_shardings):
            layer_name = f"{prefix}.{layer_id}"
            routed_expert_patterns = tuple(fqn for fqn in shardings if routed_expert_marker in fqn)
            dense_patterns = tuple(fqn for fqn in shardings if routed_expert_marker not in fqn)
            layer_bucket_configs.append(BucketConfig(name=layer_name, patterns=dense_patterns))
            layer_bucket_configs.append(
                BucketConfig(
                    name=f"{layer_name}.routed-experts",
                    patterns=routed_expert_patterns,
                )
            )
        return tuple(layer_bucket_configs)

    bucket_configs = bucket_configs_for_layers("layers", main_layer_shardings)
    bucket_configs += bucket_configs_for_layers("mtp_layers", mtp_layer_shardings)
    # hc_head is global rather than layer-scoped, so it needs its own bucket.
    bucket_configs += (BucketConfig(name="hc_head", patterns=("hc_head.hc_fn",)),)
    muon_pattern = (
        r"(?:"
        rf"attention\.(?:{'|'.join(attention_projections)})\.weight|"
        rf"attention\.indexer\.(?:{'|'.join(indexer_projections)})\.weight|"
        rf"attention\.(?:compressor(?:_128)?|indexer\.compressor)\.(?:{'|'.join(compressor_projections)})\.weight|"
        r"attention\.(?:compressor(?:_128)?|indexer\.compressor)\.ape|"
        rf"feed_forward\.shared_experts\.(?:{'|'.join(expert_projections)})\.weight|"
        rf"feed_forward\.routed_experts\.(?:{'|'.join(routed_expert_projections)})|"
        r"feed_forward\.router\.gate\.weight|"
        rf"(?:{'|'.join(hc_pre_modules)})\.hc_fn|"
        rf"(?:{'|'.join(mtp_projections)})\.weight|"
        r"hc_head\.hc_fn" + r")$"
    )
    from torchtitan_npu.extensions.torchtitan.components.optimizer.muon import DistMuon

    profile = MuonOptimizerProfile(
        optimizer_config_type=DistMuon.Config,
        muon_pattern=muon_pattern,
        optimizer_factory_kwargs={
            "DistMuon": {
                "compute_sharding_by_fqn": compute_sharding_by_fqn,
                "bucket_configs": bucket_configs,
            }
        },
    )
    return with_lora_compute_layouts(model, profile)


def _dsv4_optimizer_config(
    model: DeepSeekV4Model.Config,
    *,
    lr: float,
) -> OptimizerConfig:
    """Build recipe-owned AdamW and Muon + AdamW defaults for DSV4.

    ``--optim.optimizer.name=Muon`` selects the mixed optimizer defaults.
    Non-None scalar CLI values override the selected mode's defaults;
    the profile supplies Muon parameter matching and FlexShard layouts.
    """
    profile = _dsv4_muon_profile(model)
    return OptimizerConfig(
        _optimizer_defaults={
            "AdamW": [AdamW.Config(pattern=r".*", lr=lr, eps=1e-6)],
            "Muon": [
                profile.optimizer_config_type(
                    pattern=profile.muon_pattern,
                    lr=lr,
                    ns_steps=10,
                    adjust_lr_fn="match_rms_adamw",
                    **profile.optimizer_factory_kwargs["DistMuon"],
                ),
                AdamW.Config(pattern=r".*", lr=lr, eps=1e-8, fused=True, foreach=False),
            ],
        },
        _muon_profile=profile,
    )
