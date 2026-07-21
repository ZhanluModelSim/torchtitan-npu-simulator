# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec

from .model import DeepSeekV4Model
from .moe import MoEArgs
from .parallelize import parallelize_deepseek_v4
from .state_dict_adapter import DeepSeekV4StateDictAdapter


def _make_single_node_moe_args() -> MoEArgs:
    return MoEArgs(
        num_experts=16,
        num_shared_experts=1,
        top_k=6,
        score_func="sqrtsoftplus",
        route_norm=True,
        score_before_experts=False,
        use_grouped_mm=True,
        n_hash_layers=3,
        swiglu_limit=10,
    )


def _make_smoke_moe_args() -> MoEArgs:
    return MoEArgs(
        num_experts=8,
        num_shared_experts=1,
        top_k=2,
        score_func="sqrtsoftplus",
        route_norm=True,
        score_before_experts=False,
        use_grouped_mm=False,
        n_hash_layers=0,
        swiglu_limit=10,
    )


def _make_mini_1b_moe_args() -> MoEArgs:
    return MoEArgs(
        num_experts=16,
        num_shared_experts=1,
        top_k=2,
        score_func="sqrtsoftplus",
        route_norm=True,
        score_before_experts=False,
        use_grouped_mm=True,
        n_hash_layers=2,
        swiglu_limit=10,
    )


def _v4_flash_default_model_config() -> dict:
    moe_args = MoEArgs(
        num_experts=256,
        num_shared_experts=1,
        top_k=6,
        score_func="sqrtsoftplus",
        route_norm=True,
        score_before_experts=False,
        use_grouped_mm=True,
        n_hash_layers=3,
        swiglu_limit=10,
    )
    return {
        "vocab_size": 129280,
        "n_layers": 43,
        "n_heads": 64,
        "max_batch_size": 4,
        "max_seq_len": 4096,
        "dim": 4096,
        "moe_inter_dim": 2048,
        "head_dim": 512,
        "rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "window_size": 128,
        "compress_ratios": (1, 1, 4) + (128, 4) * 20,
        "moe_args": moe_args,
        "hc_sinkhorn_iters": 20,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "compress_rope_theta": 160000,
        "original_seq_len": 65536,
        "rope_theta": 10000,
        "rope_factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "enable_indexer_loss": True,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
    }


def _v4_pro_default_model_config() -> dict:
    moe_args = MoEArgs(
        num_experts=384,
        num_shared_experts=1,
        top_k=6,
        score_func="sqrtsoftplus",
        route_norm=True,
        score_before_experts=False,
        use_grouped_mm=True,
        n_hash_layers=3,
        swiglu_limit=10,
    )
    return {
        "vocab_size": 129280,
        "n_layers": 61,
        "n_heads": 128,
        "max_batch_size": 4,
        "max_seq_len": 4096,
        "dim": 7168,
        "moe_inter_dim": 3072,
        "head_dim": 512,
        "rope_head_dim": 64,
        "q_lora_rank": 1536,
        "o_lora_rank": 1024,
        "o_groups": 16,
        "window_size": 128,
        "compress_ratios": (128,) + (128, 4) * 30,
        "moe_args": moe_args,
        "hc_sinkhorn_iters": 20,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "compress_rope_theta": 160000,
        "original_seq_len": 65536,
        "rope_theta": 10000,
        "rope_factor": 16,
        "beta_fast": 32,
        "beta_slow": 1,
        "enable_indexer_loss": True,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 1024,
    }


def _mini_1b_model() -> DeepSeekV4Model.Config:
    config = _v4_flash_default_model_config()
    config.update(
        {
            "n_layers": 24,
            "n_heads": 16,
            "max_batch_size": 4,
            "max_seq_len": 4096,
            "dim": 1024,
            "moe_inter_dim": 512,
            "head_dim": 64,
            "rope_head_dim": 32,
            "q_lora_rank": 384,
            "o_lora_rank": 384,
            "o_groups": 4,
            "window_size": 64,
            "compress_ratios": (1, 1) + (4, 96) * 10 + (4, 1),
            "moe_args": _make_mini_1b_moe_args(),
            "hc_sinkhorn_iters": 20,
            "hc_mult": 4,
            "hc_eps": 1e-6,
            "compress_rope_theta": 160000,
            "original_seq_len": 0,
            "rope_theta": 10000,
            "rope_factor": 1,
            "beta_fast": 32,
            "beta_slow": 1,
            "enable_indexer_loss": True,
            "index_n_heads": 8,
            "index_head_dim": 64,
            "index_topk": 128,
        }
    )
    return DeepSeekV4Model.Config(**config)


def _v4_flash_43layers_16experts() -> DeepSeekV4Model.Config:
    config = _v4_flash_default_model_config()
    config.update(
        {
            "moe_args": _make_single_node_moe_args(),
        }
    )
    return DeepSeekV4Model.Config(**config)


def _v4_flash_43layers_256experts() -> DeepSeekV4Model.Config:
    config = _v4_flash_default_model_config()
    return DeepSeekV4Model.Config(**config)


def _v4_pro_16layers_16experts() -> DeepSeekV4Model.Config:
    config = _v4_pro_default_model_config()
    config.update(
        {
            "n_layers": 16,
            "moe_args": _make_single_node_moe_args(),
        }
    )
    return DeepSeekV4Model.Config(**config)


def _v4_pro_61layers_384experts() -> DeepSeekV4Model.Config:
    config = _v4_pro_default_model_config()
    return DeepSeekV4Model.Config(**config)


deepseekv4_configs = {
    "v4_flash_43layers_16experts": _v4_flash_43layers_16experts,
    "v4_flash_43layers_256experts": _v4_flash_43layers_256experts,
    "v4_pro_16layers_16experts": _v4_pro_16layers_16experts,
    "v4_pro_61layers_384experts": _v4_pro_61layers_384experts,
    "mini_1b": _mini_1b_model,
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="deepseek_v4",
        flavor=flavor,
        model=deepseekv4_configs[flavor](),
        build_loss_fn=build_cross_entropy_loss,
        parallelize_fn=parallelize_deepseek_v4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV4StateDictAdapter,
    )
