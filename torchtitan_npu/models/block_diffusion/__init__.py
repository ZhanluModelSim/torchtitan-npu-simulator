# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Block Diffusion model registry for TorchTitan NPU."""

from collections.abc import Callable
from functools import partial

from torch import nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_gqa_config,
    make_router_config,
)
from torchtitan.protocols.model_spec import ModelSpec

from .attention import PrefixCanvasSDPA
from .feed_forward import BlockDiffusionGroupedExperts, BlockDiffusionMoE
from .model import BlockDiffusionModel, BlockDiffusionTransformerBlock
from .parallelize import parallelize_block_diffusion
from .state_dict_adapter import BlockDiffusionStateDictAdapter

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    num_experts: int,
    moe_intermediate_size: int,
    num_experts_per_tok: int,
    block_size: int,
    use_grouped_mm: bool,
) -> list[BlockDiffusionTransformerBlock.Config]:
    layers = []
    for _layer_id in range(n_layers):
        attention = make_gqa_config(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            wqkv_param_init=_LINEAR_INIT,
            wo_param_init=_LINEAR_INIT,
            inner_attention=PrefixCanvasSDPA.Config(block_size=block_size),
            mask_type="causal",
            rope_backend="complex",
        )
        dense_ffn = None
        moe = None
        if num_experts == 0:
            dense_ffn = make_ffn_config(
                dim=dim,
                hidden_dim=intermediate_size,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_LINEAR_INIT,
            )
        else:
            shared_expert = make_ffn_config(
                dim=dim,
                hidden_dim=intermediate_size,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_LINEAR_INIT,
            )
            moe = BlockDiffusionMoE.Config(
                num_experts=num_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_LINEAR_INIT,
                    top_k=num_experts_per_tok,
                    score_func="softmax",
                    route_norm=True,
                ),
                experts=BlockDiffusionGroupedExperts.Config(
                    dim=dim,
                    hidden_dim=moe_intermediate_size,
                    num_experts=num_experts,
                    param_init={
                        "w1": partial(nn.init.trunc_normal_, std=0.02),
                        "w2": partial(nn.init.trunc_normal_, std=0.02),
                        "w3": partial(nn.init.trunc_normal_, std=0.02),
                    },
                    use_grouped_mm=use_grouped_mm,
                ),
                shared_experts=shared_expert,
                score_before_experts=False,
                load_balance_coeff=1e-3,
            )

        layers.append(
            BlockDiffusionTransformerBlock.Config(
                attention=attention,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=1e-5,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=1e-5,
                    param_init=_NORM_INIT,
                ),
                feed_forward=dense_ffn,
                moe=moe,
            )
        )
    return layers


def make_block_diffusion_config(
    *,
    vocab_size: int,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
    num_experts: int,
    moe_intermediate_size: int,
    num_experts_per_tok: int,
    block_size: int,
    mask_token_id: int,
    max_seq_len: int,
    max_denoise_steps: int = 48,
    enable_weight_tying: bool = False,
    use_grouped_mm: bool = True,
) -> BlockDiffusionModel.Config:
    return BlockDiffusionModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
        num_experts_per_tok=num_experts_per_tok,
        block_size=block_size,
        mask_token_id=mask_token_id,
        max_denoise_steps=max_denoise_steps,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-5,
            param_init=_NORM_INIT,
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            bias=False,
            param_init=_output_init(dim),
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=max_seq_len,
            theta=1_000_000.0,
            backend="complex",
            scaling="llama",
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            block_size=block_size,
            use_grouped_mm=use_grouped_mm,
        ),
    )


def _debug_model() -> BlockDiffusionModel.Config:
    return make_block_diffusion_config(
        vocab_size=2048,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        intermediate_size=256,
        num_experts=8,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        block_size=16,
        mask_token_id=100,
        max_seq_len=512,
        use_grouped_mm=False,
    )


def _dense_debug_model() -> BlockDiffusionModel.Config:
    return make_block_diffusion_config(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        intermediate_size=128,
        num_experts=0,
        moe_intermediate_size=0,
        num_experts_per_tok=1,
        block_size=8,
        mask_token_id=100,
        max_seq_len=256,
        use_grouped_mm=False,
    )


def _reduced_model() -> BlockDiffusionModel.Config:
    return make_block_diffusion_config(
        vocab_size=8192,
        dim=1024,
        n_layers=8,
        n_heads=16,
        n_kv_heads=8,
        head_dim=64,
        intermediate_size=2048,
        num_experts=32,
        moe_intermediate_size=512,
        num_experts_per_tok=4,
        block_size=64,
        mask_token_id=100,
        max_seq_len=4096,
    )


def _full_model() -> BlockDiffusionModel.Config:
    return make_block_diffusion_config(
        vocab_size=262144,
        dim=8192,
        n_layers=97,
        n_heads=64,
        n_kv_heads=8,
        head_dim=128,
        intermediate_size=14336,
        num_experts=1024,
        moe_intermediate_size=4096,
        num_experts_per_tok=8,
        block_size=256,
        mask_token_id=100,
        max_seq_len=262144,
    )


block_diffusion_configs = {
    "debug": _debug_model,
    "dense_debug": _dense_debug_model,
    "reduced": _reduced_model,
    "full": _full_model,
}


def model_registry(flavor: str) -> ModelSpec:
    if flavor not in block_diffusion_configs:
        raise ValueError(
            f"Unknown Block Diffusion flavor {flavor!r}; "
            f"available: {sorted(block_diffusion_configs)}"
        )
    return ModelSpec(
        name="block_diffusion",
        flavor=flavor,
        model=block_diffusion_configs[flavor](),
        parallelize_fn=parallelize_block_diffusion,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=BlockDiffusionStateDictAdapter,
    )


__all__ = [
    "BlockDiffusionModel",
    "BlockDiffusionTransformerBlock",
    "block_diffusion_configs",
    "make_block_diffusion_config",
    "model_registry",
    "parallelize_block_diffusion",
]
