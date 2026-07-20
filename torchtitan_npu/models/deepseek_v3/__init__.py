# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE
from torchtitan.models.deepseek_v3 import (
    _EMBEDDING_INIT,
    _NORM_INIT,
    DeepSeekV3StateDictAdapter,
    _build_dsv3_layers,
    _output_linear_init,
)
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.models.deepseek_v3.model import DeepSeekV3ModelNpu
from torchtitan_npu.models.deepseek_v3.parallelize import parallelize_deepseekv3
from torchtitan_npu.models.deepseek_v3.state_dict_adapter import (
    DeepSeek16BStateDictAdapterNpu,
)


def _make_dsv3_model_config(
    vocab_size: int = 129280,
    dim: int = 7168,
    inter_dim: int = 18432,
    moe_inter_dim: int = 2048,
    n_layers: int = 61,
    n_dense_layers: int = 3,
    n_heads: int = 128,
    num_experts: int = 256,
    num_shared_experts: int = 1,
    q_lora_rank: int = 1536,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    router_top_k: int = 8,
    router_score_func: str = "sigmoid",
    router_route_scale: float = 2.5,
    router_route_norm: bool = True,
    mscale: float = 1.0,
) -> DeepSeekV3ModelNpu.Config:
    rope_dim = qk_rope_head_dim
    layers = _build_dsv3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        dense_hidden_dim=inter_dim,
        moe_hidden_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=router_top_k,
        router_score_func=router_score_func,  # pyrefly: ignore [bad-argument-type]
        router_route_scale=router_route_scale,
        router_route_norm=router_route_norm,
        score_before_experts=False,
    )
    return DeepSeekV3ModelNpu.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        rope=RoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            backend="complex",
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        layers=layers,
    )


def _16b_model() -> DeepSeekV3ModelNpu.Config:
    model_config = _make_dsv3_model_config(
        vocab_size=102400,
        dim=2048,
        inter_dim=10944,
        moe_inter_dim=1408,
        n_layers=28,
        n_dense_layers=1,
        n_heads=16,
        num_experts=64,
        num_shared_experts=2,
        q_lora_rank=0,
        kv_lora_rank=3072,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        v_head_dim=128,
        router_top_k=6,
        router_score_func="softmax",
        router_route_scale=1.0,
        router_route_norm=False,
        mscale=1.0,
    )
    model_config.rope = RoPE.Config(
        dim=64,
        max_seq_len=4096,
        theta=10000.0,
        backend="complex",
        scaling="none",
    )
    return model_config


def _671b_debug_model() -> DeepSeekV3ModelNpu.Config:
    return _make_dsv3_model_config(
        dim=128,
        inter_dim=512,
        moe_inter_dim=128,
        n_layers=4,
        n_dense_layers=3,
        num_experts=8,
    )


def _671b_debug_4layers_256experts() -> DeepSeekV3ModelNpu.Config:
    return _make_dsv3_model_config(
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=4,
        n_dense_layers=3,
        num_experts=256,
    )


def _671b_debug_16die_model() -> DeepSeekV3ModelNpu.Config:
    return _make_dsv3_model_config(
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=4,
        n_dense_layers=3,
        num_experts=8,
    )


def _671b_debug_128die_model() -> DeepSeekV3ModelNpu.Config:
    return _make_dsv3_model_config(
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        num_experts=256,
    )


deepseekv3_configs = {
    "16B": _16b_model,
    "671B_debug": _671b_debug_model,
    "671B_debug_4layers_256experts": _671b_debug_4layers_256experts,
    "671B_debug_16die": _671b_debug_16die_model,
    "671B_debug_128die": _671b_debug_128die_model,
}


def model_registry(flavor: str) -> ModelSpec:
    model_config = deepseekv3_configs[flavor]()
    adapter_cls = DeepSeek16BStateDictAdapterNpu if flavor == "16B" else DeepSeekV3StateDictAdapter
    return ModelSpec(
        name="deepseek_v3",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_deepseekv3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=adapter_cls,
    )
