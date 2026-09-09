# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ar_llm (DeepSeekV4-Sparse) model registration and config builders.

Provides model_registry() for torchtitan's ModelSpec discovery. Architecture
reference: torchtitan_npu/simulator/raw_model/ar_llm/; framework contract:
torchtitan_npu/models/ar_llm/MODEL_CONTRACT.md.
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec

from .model import ArLlmModel
from .parallelize import parallelize_ar_llm
from .state_dict_adapter import ArLlmStateDictAdapter


def _debug_model() -> ArLlmModel.Config:
    """One 6-layer unit exercising CSA, HCA, KDA, both routers, mHC, Engram."""
    return ArLlmModel.Config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        n_heads=8,
        head_dim=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        o_groups=2,
        o_lora_rank=32,
        max_seq_len=128,
        rope_theta=10000.0,
        yarn_factor=4.0,
        yarn_original_max=256,
        kda_d_state=32,
        kda_d_k=16,
        kda_d_v=16,
        csa_compress_ratio=2,
        csa_window_size=16,
        hca_compress_ratio=2,
        indexer_n_heads=4,
        indexer_head_dim=16,
        indexer_topk=8,
        num_routed_experts=8,
        num_shared_experts=1,
        moe_intermediate_size=64,
        moe_latent_dim=48,
        num_experts_per_token=2,
        hc_mult=2,
        sinkhorn_iters=4,
        engram_layers=[2],
        engram_ngram_orders=[2],
        engram_num_hash_heads=2,
        engram_table_capacity=64,
        engram_memory_dim=16,
    )


def _reduced_model() -> ArLlmModel.Config:
    """Two 6-layer units with production-like dim ratios, small experts."""
    return ArLlmModel.Config(
        vocab_size=16384,
        dim=1024,
        n_layers=12,
        n_heads=16,
        head_dim=64,
        qk_nope_head_dim=48,
        qk_rope_head_dim=16,
        q_lora_rank=256,
        kv_lora_rank=128,
        o_groups=4,
        o_lora_rank=256,
        max_seq_len=512,
        rope_theta=10000.0,
        yarn_factor=8.0,
        yarn_original_max=512,
        kda_d_state=64,
        kda_d_k=32,
        kda_d_v=32,
        csa_compress_ratio=4,
        csa_window_size=128,
        hca_compress_ratio=16,
        indexer_n_heads=8,
        indexer_head_dim=32,
        indexer_topk=64,
        num_routed_experts=32,
        num_shared_experts=2,
        moe_intermediate_size=256,
        moe_latent_dim=384,
        num_experts_per_token=8,
        hc_mult=4,
        sinkhorn_iters=8,
        engram_layers=[2, 8],
        engram_ngram_orders=[2, 3],
        engram_num_hash_heads=4,
        engram_table_capacity=1024,
        engram_memory_dim=256,
    )


def _50t_model() -> ArLlmModel.Config:
    """Official 50T spec (raw preset_50T); meta/capacity validation only."""
    return ArLlmModel.Config(
        vocab_size=524288,
        dim=16384,
        n_layers=60,
        n_heads=128,
        head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        q_lora_rank=4096,
        kv_lora_rank=1024,
        o_groups=32,
        o_lora_rank=4096,
        max_seq_len=65536,
        rope_theta=16384.0,
        yarn_factor=32.0,
        yarn_original_max=65536,
        kda_d_state=256,
        kda_d_k=128,
        kda_d_v=128,
        csa_compress_ratio=16,
        csa_window_size=1024,
        hca_compress_ratio=256,
        indexer_n_heads=64,
        indexer_head_dim=128,
        indexer_topk=4096,
        num_routed_experts=2048,
        num_shared_experts=2,
        moe_intermediate_size=4096,
        moe_latent_dim=7168,
        num_experts_per_token=16,
        hc_mult=4,
        sinkhorn_iters=20,
        engram_layers=[3, 9, 15, 21, 27, 33, 39, 45, 51, 57],
        engram_ngram_orders=[2, 3],
        engram_num_hash_heads=8,
        engram_table_capacity=8_388_608,
        engram_memory_dim=4096,
    )


def _100t_model() -> ArLlmModel.Config:
    """Official 100T spec (raw preset_100T); meta/capacity validation only."""
    return ArLlmModel.Config(
        vocab_size=524288,
        dim=16384,
        n_layers=60,
        n_heads=128,
        head_dim=256,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        q_lora_rank=4096,
        kv_lora_rank=1024,
        o_groups=32,
        o_lora_rank=4096,
        max_seq_len=65536,
        rope_theta=16384.0,
        yarn_factor=64.0,
        yarn_original_max=65536,
        kda_d_state=512,
        kda_d_k=256,
        kda_d_v=256,
        csa_compress_ratio=24,
        csa_window_size=2048,
        hca_compress_ratio=512,
        indexer_n_heads=64,
        indexer_head_dim=128,
        indexer_topk=4096,
        num_routed_experts=4096,
        num_shared_experts=2,
        moe_intermediate_size=4096,
        moe_latent_dim=7168,
        num_experts_per_token=16,
        hc_mult=4,
        sinkhorn_iters=20,
        engram_layers=[
            2, 5, 8, 11, 14, 17, 20, 23, 26, 29,
            32, 35, 38, 41, 44, 47, 50, 53, 56, 59,
        ],
        engram_ngram_orders=[2, 3],
        engram_num_hash_heads=8,
        engram_table_capacity=8_388_608,
        engram_memory_dim=4096,
    )


ar_llm_configs = {
    "debug": _debug_model,
    "reduced": _reduced_model,
    "50t": _50t_model,
    "100t": _100t_model,
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="ar_llm",
        flavor=flavor,
        model=ar_llm_configs[flavor](),
        build_loss_fn=build_cross_entropy_loss,
        parallelize_fn=parallelize_ar_llm,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=ArLlmStateDictAdapter,
    )
