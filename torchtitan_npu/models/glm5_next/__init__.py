# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""glm5_next (GLM-5.3-Flash) model registration and config builders.

Architecture reference: torchtitan_npu/simulator/raw_model/unified_mm/;
framework contract: torchtitan_npu/models/glm5_next/MODEL_CONTRACT.md.
Flavors: ``debug`` (real-execution smoke), ``reduced`` (meta core-combo),
``full`` (official 96-layer spec, meta/capacity validation only).
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.protocols.model_spec import ModelSpec

from .model import Glm5NextModel, GlmVisionTower
from .parallelize import parallelize_glm5_next
from .state_dict_adapter import Glm5NextStateDictAdapter


def _debug_model() -> Glm5NextModel.Config:
    """Tiny spec exercising every path: KDA, DSA, MoE, dense, mHC, loop, vision."""
    return Glm5NextModel.Config(
        vocab_size=2048,
        hidden_size=256,
        num_hidden_layers=9,
        pre_layers=4,
        looped_layers=1,
        post_layers=4,
        loop_train_steps=2,
        num_attention_heads=8,
        kda_num_heads=8,
        kda_head_dim=32,
        kda_conv_kernel_size=4,
        kda_gate_lower_bound=-5.0,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        v_head_dim=32,
        indexer_n_heads=4,
        indexer_head_dim=16,
        index_topk=32,
        index_kpool=4,
        first_k_dense_replace=2,
        intermediate_size=768,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=64,
        routed_scaling_factor=2.5,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        max_seq_len=128,
        image_token_id=100,
        video_start_token_id=101,
        vision_config=GlmVisionTower.Config(
            depth=2,
            hidden_size=64,
            num_heads=4,
            intermediate_size=128,
            out_hidden_size=256,
            patch_size=14,
            spatial_merge_size=2,
            image_size=56,
            projection_intermediate_size=256,
        ),
    )


def _reduced_model() -> Glm5NextModel.Config:
    """Reduced spec with production-like ratios for meta single-step tests."""
    return Glm5NextModel.Config(
        vocab_size=16384,
        hidden_size=1024,
        num_hidden_layers=12,
        pre_layers=4,
        looped_layers=2,
        post_layers=6,
        loop_train_steps=4,
        num_attention_heads=16,
        kda_num_heads=16,
        kda_head_dim=64,
        q_lora_rank=256,
        kv_lora_rank=128,
        qk_nope_head_dim=128,
        v_head_dim=128,
        indexer_n_heads=8,
        indexer_head_dim=32,
        index_topk=256,
        index_kpool=4,
        first_k_dense_replace=2,
        intermediate_size=3072,
        n_routed_experts=32,
        num_experts_per_tok=8,
        n_shared_experts=1,
        moe_intermediate_size=128,
        routed_scaling_factor=2.5,
        hc_mult=4,
        hc_sinkhorn_iters=8,
        max_seq_len=512,
        image_token_id=100,
        video_start_token_id=101,
        vision_config=GlmVisionTower.Config(
            depth=4,
            hidden_size=256,
            num_heads=8,
            intermediate_size=512,
            out_hidden_size=1024,
            patch_size=14,
            spatial_merge_size=2,
            image_size=112,
            projection_intermediate_size=1024,
        ),
    )


def _full_model() -> Glm5NextModel.Config:
    """Official GLM-5.3-Flash spec (config.json); meta/capacity only."""
    return Glm5NextModel.Config(
        vocab_size=154880,
        hidden_size=24576,
        num_hidden_layers=96,
        pre_layers=16,
        looped_layers=56,
        post_layers=24,
        loop_train_steps=4,
        num_attention_heads=192,
        kda_num_heads=192,
        kda_head_dim=128,
        q_lora_rank=6144,
        kv_lora_rank=2048,
        qk_nope_head_dim=256,
        v_head_dim=256,
        indexer_n_heads=64,
        indexer_head_dim=128,
        index_topk=8192,
        index_kpool=8,
        first_k_dense_replace=4,
        intermediate_size=73728,
        n_routed_experts=2048,
        num_experts_per_tok=16,
        n_shared_experts=1,
        moe_intermediate_size=3072,
        routed_scaling_factor=2.5,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        max_seq_len=65536,
        vision_config=GlmVisionTower.Config(
        depth=32,
        hidden_size=2048,
        num_heads=32,
        intermediate_size=8192,
        out_hidden_size=24576,
        patch_size=14,
        spatial_merge_size=2,
        image_size=672,
        projection_intermediate_size=49152,
        ),
    )


def pipelining_glm5_next(*args, **kwargs):
    """PP is not supported: fail fast before upstream stage splitting.

    The loop region (one shared block executed T times), the vision tower,
    and the hc_head cross-stage contract are undefined for pipeline
    parallelism; see torchtitan_npu/models/glm5_next/MODEL_CONTRACT.md
    section 10.
    """
    raise NotImplementedError(
        "glm5_next pipeline parallelism is deferred until the loop-region and "
        "hc_head cross-stage contract is defined; see "
        "torchtitan_npu/models/glm5_next/MODEL_CONTRACT.md"
    )


glm5_next_configs = {
    "debug": _debug_model,
    "reduced": _reduced_model,
    "full": _full_model,
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="glm5_next",
        flavor=flavor,
        model=glm5_next_configs[flavor](),
        build_loss_fn=build_cross_entropy_loss,
        parallelize_fn=parallelize_glm5_next,
        pipelining_fn=pipelining_glm5_next,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Glm5NextStateDictAdapter,
    )
