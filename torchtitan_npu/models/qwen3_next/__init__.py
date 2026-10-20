# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3-Next model package: hybrid Gated DeltaNet / gated attention MoE."""

from torchtitan_npu.models.qwen3_next.loss import build_qwen3_next_chunk_loss
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RoPE
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.protocols.model_spec import ModelSpec

from .model import Qwen3NextModel, Qwen3NextTransformerBlock
from .parallelize import parallelize_qwen3_next
from .state_dict_adapter import Qwen3NextStateDictAdapter

__all__ = [
    "parallelize_qwen3_next",
    "Qwen3NextModel",
    "qwen3_next_configs",
]

_EPS = 1e-6


def _trunc_normal_002(param):
    import torch.nn as nn

    nn.init.trunc_normal_(param, std=0.02)


def _zeros(param):
    import torch.nn as nn

    nn.init.zeros_(param)


def _norm(dim: int) -> "RMSNorm.Config":
    from torchtitan.models.common.rmsnorm import RMSNorm

    return RMSNorm.Config(normalized_shape=dim, eps=_EPS)


def _ffn(dim: int, hidden_dim: int) -> FeedForward.Config:
    from torchtitan.models.common.linear import Linear

    init = {"weight": _trunc_normal_002, "bias": _zeros}
    return FeedForward.Config(
        w1=Linear.Config(in_features=dim, out_features=hidden_dim, param_init=init),
        w2=Linear.Config(in_features=hidden_dim, out_features=dim, param_init=init),
        w3=Linear.Config(in_features=dim, out_features=hidden_dim, param_init=init),
    )


def _gdn_config(dim: int, *, num_v_heads, num_k_heads, head_k_dim, head_v_dim, conv_kernel_size=4, gdn_chunk_size=64):
    from .gated_deltanet import GatedDeltaNet

    return GatedDeltaNet.Config(
        dim=dim,
        num_v_heads=num_v_heads,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        conv_kernel_size=conv_kernel_size,
        norm_eps=_EPS,
        gdn_chunk_size=gdn_chunk_size,
    )


def _gated_attn_config(dim: int, *, n_heads, n_kv_heads, head_dim):
    from .attention import GatedAttention

    return GatedAttention.Config(
        dim=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        norm_eps=_EPS,
    )


def _moe_config(dim: int, *, num_experts, top_k, moe_intermediate_size, shared_expert_intermediate_size, norm_topk_prob=True):
    from .feed_forward import Qwen3NextMoE

    return Qwen3NextMoE.Config(
        hidden_size=dim,
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        norm_topk_prob=norm_topk_prob,
    )


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    layer_types: list[str],
    mlp_only_layers: list[int],
    dense_intermediate_size: int,
    gdn_kwargs: dict,
    attn_kwargs: dict,
    moe_kwargs: dict,
) -> list[Qwen3NextTransformerBlock.Config]:
    """Build per-layer configs following the Qwen3-Next hybrid layout."""
    layers = []
    for layer_id in range(n_layers):
        layer_type = layer_types[layer_id]
        if layer_type == "linear_attention":
            token_mixer = _gdn_config(dim, **gdn_kwargs)
        elif layer_type == "full_attention":
            token_mixer = _gated_attn_config(dim, **attn_kwargs)
        else:
            raise ValueError(f"Unknown layer type: {layer_type}")

        if layer_id in mlp_only_layers:
            ffn = _ffn(dim, dense_intermediate_size)
            moe = None
        else:
            ffn = None
            moe = _moe_config(dim, **moe_kwargs)

        layers.append(
            Qwen3NextTransformerBlock.Config(
                dim=dim,
                layer_id=layer_id,
                token_mixer=token_mixer,
                feed_forward=ffn,
                moe=moe,
                norm_eps=_EPS,
            )
        )
    return layers


def _hybrid_layer_types(n_layers: int) -> list[str]:
    """3 linear-attention layers + 1 full-attention layer per group of 4."""
    return [
        "linear_attention" if (i % 4) != 3 else "full_attention"
        for i in range(n_layers)
    ]


def _debugmodel() -> Qwen3NextModel.Config:
    """Minimal debug config: 4 layers (3 GDN + 1 gated attention), 8 experts."""
    dim = 256
    vocab_size = 2048
    n_layers = 4
    head_dim = 64
    return Qwen3NextModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init={"weight": skip_param_init},
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init={"weight": _trunc_normal_002, "bias": _zeros},
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=512,
            theta=10000.0,
            backend="cos_sin",
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            layer_types=_hybrid_layer_types(n_layers),
            mlp_only_layers=[],
            dense_intermediate_size=512,
            gdn_kwargs=dict(
                num_v_heads=8,
                num_k_heads=4,
                head_k_dim=32,
                head_v_dim=32,
            ),
            attn_kwargs=dict(n_heads=8, n_kv_heads=4, head_dim=head_dim),
            moe_kwargs=dict(
                num_experts=8,
                top_k=2,
                moe_intermediate_size=128,
                shared_expert_intermediate_size=128,
            ),
        ),
    )


def _80b_a3b() -> Qwen3NextModel.Config:
    """Full Qwen3-Next-80B-A3B production config."""
    dim = 2048
    vocab_size = 151936
    n_layers = 48
    head_dim = 256
    return Qwen3NextModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init={"weight": skip_param_init},
        ),
        output=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init={"weight": _trunc_normal_002, "bias": _zeros},
        ),
        rope=RoPE.Config(
            dim=head_dim,
            max_seq_len=262144,
            theta=10000000.0,
            backend="cos_sin",
        ),
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            layer_types=_hybrid_layer_types(n_layers),
            mlp_only_layers=[],
            dense_intermediate_size=512,
            gdn_kwargs=dict(
                num_v_heads=32,
                num_k_heads=16,
                head_k_dim=128,
                head_v_dim=128,
            ),
            attn_kwargs=dict(n_heads=16, n_kv_heads=2, head_dim=head_dim),
            moe_kwargs=dict(
                num_experts=512,
                top_k=10,
                moe_intermediate_size=512,
                shared_expert_intermediate_size=512,
            ),
        ),
    )


def _80b_a3b_2layer() -> Qwen3NextModel.Config:
    """80B dims trimmed to 2 layers (one full-attention + one
    linear-attention), both MoE — the measured comparison config."""
    cfg = _80b_a3b()
    cfg.layers = _build_layers(
        n_layers=2,
        dim=2048,
        layer_types=["full_attention", "linear_attention"],
        mlp_only_layers=[],
        dense_intermediate_size=512,
        gdn_kwargs=dict(
            num_v_heads=32,
            num_k_heads=16,
            head_k_dim=128,
            head_v_dim=128,
        ),
        attn_kwargs=dict(n_heads=16, n_kv_heads=2, head_dim=256),
        moe_kwargs=dict(
            num_experts=512,
            top_k=10,
            moe_intermediate_size=512,
            shared_expert_intermediate_size=512,
        ),
    )
    return cfg


qwen3_next_configs = {
    "debugmodel": _debugmodel,
    "80B-A3B": _80b_a3b,
    "80B-A3B-2L": _80b_a3b_2layer,
}


def model_registry(flavor: str) -> ModelSpec:
    config = qwen3_next_configs[flavor]()
    return ModelSpec(
        name="qwen3_next",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_next,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_qwen3_next_chunk_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Qwen3NextStateDictAdapter,
    )
