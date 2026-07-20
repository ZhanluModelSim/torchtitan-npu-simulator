# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Literal

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import Embedding, Linear, RMSNorm, RoPE
from torchtitan.models.common.attention import FlexAttention, ScaledDotProductAttention
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.deepseek_v3 import (
    _EMBEDDING_INIT,
    _LINEAR_INIT,
    _NORM_INIT,
    _depth_experts_init,
    _depth_init,
    _output_linear_init,
)
from torchtitan.protocols.model_spec import ModelSpec

from .model import Attention, DeepSeekV32ModelNpu, TransformerBlockV32
from .parallelize import parallelize_deepseekv32
from .state_dict_adapter import DeepSeekV32StateDictAdapter


def _make_dsv32_attn_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float = 1.0,
    norm_eps: float = 1e-6,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    index_topk: int = 2048,
    enable_mla_absorb: bool = True,
    inner_attention=None,
    mask_type: str = "causal",
) -> Attention.Config:
    """Build a fully-specified DeepSeek V3.2 ``Attention.Config``.

    Mirrors ``_make_dsv3_attn_config`` for the v3 MLA fields and adds the v32
    lightning-indexer + MLA-absorb knobs. v32 always uses LoRA query
    projection (``q_lora_rank > 0``); ``q_lora_rank == 0`` is rejected.
    """
    assert q_lora_rank > 0, "DeepSeek V3.2 requires q_lora_rank > 0"
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=None,
        wq_a=Linear.Config(in_features=dim, out_features=q_lora_rank, param_init=_LINEAR_INIT),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(normalized_shape=q_lora_rank, eps=norm_eps, param_init=_NORM_INIT),
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank, eps=norm_eps, param_init=_NORM_INIT),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        inner_attention=(inner_attention if inner_attention is not None else ScaledDotProductAttention.Config()),
        mask_type=mask_type,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        enable_mla_absorb=enable_mla_absorb,
    )


def _build_dsv32_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    num_mtp_modules: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    norm_eps: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    index_n_heads: int = 64,
    index_head_dim: int = 128,
    index_topk: int = 2048,
    enable_mla_absorb: bool = True,
    inner_attention=None,
    mask_type: str = "causal",
) -> list:
    """Build per-layer ``TransformerBlockV32.Config`` list (main + MTP).

    Layers ``[0, n_dense_layers)`` are dense; layers
    ``[n_dense_layers, n_layers)`` are MoE; the trailing
    ``num_mtp_modules`` layers are MTP modules — they are still
    ``TransformerBlockV32.Config`` (MTPModule subclasses TransformerBlockV32);
    layer-position selection happens in ``DeepSeekV32ModelNpu.__init__``.

    All MTP layers reuse the MoE block topology (same dense/MoE split rule
    extended past ``n_layers``: every MTP layer is MoE because
    ``mtp_layer_id >= n_dense_layers``).
    """
    layers = []
    total_layers = n_layers + num_mtp_modules
    for layer_id in range(total_layers):
        attn_cfg = _make_dsv32_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            norm_eps=norm_eps,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            enable_mla_absorb=enable_mla_absorb,
            inner_attention=inner_attention,
            mask_type=mask_type,
        )

        is_dense = layer_id < n_dense_layers
        if is_dense:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_moe_config(
                num_experts=num_experts,
                score_before_experts=score_before_experts,
                router=make_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    param_init=_depth_experts_init(layer_id),
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )

        layers.append(
            TransformerBlockV32.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
            )
        )
    return layers


def _extend_dsv32_layers_with_mtp(
    layers: list,
    n_dense_layers: int,
    num_mtp_modules: int,
) -> list:
    """
    Layer ids continue from ``len(layers)`` so ``_depth_init`` /
    ``_depth_experts_init`` produce the same init scale as a fresh registry
    build with the larger ``num_mtp_modules``. Build params are recovered
    from the last MoE layer (MTP layers are always MoE because
    ``mtp_layer_id >= n_dense_layers`` by construction).
    """
    moe_layer = next((l for l in reversed(layers) if l.moe is not None), None)
    assert moe_layer is not None, "_extend_dsv32_layers_with_mtp requires at least one MoE layer template"
    attn = moe_layer.attention
    moe = moe_layer.moe

    moe_hidden_dim = moe.experts.hidden_dim
    num_shared_experts = moe.shared_experts.w1.out_features // moe_hidden_dim if moe.shared_experts is not None else 0
    # ``dense_hidden_dim`` only affects dense layers, which we slice away
    # below — but pass a real value so the registry helper doesn't trip.
    dense_layer = next((l for l in layers if l.feed_forward is not None), None)
    dense_hidden_dim = dense_layer.feed_forward.w1.out_features if dense_layer is not None else moe_hidden_dim

    n_main = len(layers)
    all_layers = _build_dsv32_layers(
        n_layers=n_main,
        n_dense_layers=n_dense_layers,
        num_mtp_modules=num_mtp_modules,
        dim=attn.dim,
        n_heads=attn.n_heads,
        q_lora_rank=attn.q_lora_rank,
        kv_lora_rank=attn.kv_lora_rank,
        qk_nope_head_dim=attn.qk_nope_head_dim,
        qk_rope_head_dim=attn.qk_rope_head_dim,
        v_head_dim=attn.v_head_dim,
        mscale=attn.mscale,
        norm_eps=attn.q_norm.eps,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=moe.num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=moe.router.top_k,
        router_score_func=moe.router.score_func,
        router_num_expert_groups=moe.router.num_expert_groups,
        router_num_limited_groups=moe.router.num_limited_groups,
        router_route_scale=moe.router.route_scale,
        router_route_norm=moe.router.route_norm,
        score_before_experts=moe.score_before_experts,
        index_n_heads=attn.index_n_heads,
        index_head_dim=attn.index_head_dim,
        index_topk=attn.index_topk,
        enable_mla_absorb=attn.enable_mla_absorb,
        inner_attention=attn.inner_attention,
        mask_type=attn.mask_type,
    )
    return all_layers[n_main:]


def _make_dsv32_model_config(
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
    norm_eps: float = 1e-6,
    num_mtp_modules: int = 0,
    mask_type: str = "causal",
    inner_attention=None,
    route_scale: float = 1.0,
) -> DeepSeekV32ModelNpu.Config:
    rope_dim = qk_rope_head_dim
    layers = _build_dsv32_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        num_mtp_modules=num_mtp_modules,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=1.0,
        norm_eps=norm_eps,
        dense_hidden_dim=inter_dim,
        moe_hidden_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=8,
        router_score_func="sigmoid",
        router_route_scale=route_scale,
        router_route_norm=True,
        score_before_experts=False,
        mask_type=mask_type,
        inner_attention=inner_attention,
    )
    return DeepSeekV32ModelNpu.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT),
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
        num_mtp_modules=num_mtp_modules,
    )


def _smoketest_model() -> DeepSeekV32ModelNpu.Config:
    # Minimal config for integration smoke tests; mirrors the dsv4 smoketest
    # philosophy of "just enough to exercise every code path quickly".
    # ``num_experts=8`` is kept to satisfy ``router_top_k=8`` hardcoded in
    # ``_make_dsv32_model_config`` (top_k must be <= num_experts).
    return _make_dsv32_model_config(
        dim=128,
        inter_dim=256,
        moe_inter_dim=64,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        num_experts=8,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
    )


def _v32_671b_61layers_16experts_model() -> DeepSeekV32ModelNpu.Config:
    # FlexAttention.Config is the placeholder required by
    # upstream validation (runtime inner_attention is replaced
    # by ``DSASparseAttention`` in ``Attention.__init__``).
    return _make_dsv32_model_config(
        n_layers=61,
        mask_type="block_causal",
        inner_attention=FlexAttention.Config(),
        num_experts=16,
        route_scale=2.5,
    )


def _v32_671b_61layers_256experts_model() -> DeepSeekV32ModelNpu.Config:
    return _make_dsv32_model_config(
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        num_experts=256,
        route_scale=2.5,
    )


deepseekv32_configs = {
    "smoketest": _smoketest_model,
    "_v32_671b_61layers_16experts": _v32_671b_61layers_16experts_model,
    "_v32_671b_61layers_256experts": _v32_671b_61layers_256experts_model,
}


def model_registry(flavor: str) -> ModelSpec:
    model_config = deepseekv32_configs[flavor]()
    return ModelSpec(
        name="deepseek_v32",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_deepseekv32,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=DeepSeekV32StateDictAdapter,
    )
