# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi K3 model registration and config builders.

Provides model_registry() for torchtitan's ModelSpec discovery, and layer
builder helpers that construct hybrid KDA/Gated-MLA decoder layers.
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec

from .attention import KimiDeltaAttention, KimiGatedMLA
from .feed_forward import KimiMLP, KimiSparseMoeBlock
from .model import KimiK3Model, KimiK3TransformerBlock
from .parallelize import parallelize_kimi_k3
from .state_dict_adapter import KimiK3StateDictAdapter

# KDA layer indices from the official config, converted from 1-based to 0-based.
_FULL_KDA_LAYERS = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
    20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38,
    40, 41, 42, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 58,
    60, 61, 62, 64, 65, 66, 68, 69, 70, 72, 73, 74, 76, 77, 78,
    80, 81, 82, 84, 85, 86, 88, 89, 90,
]


def _build_kimi_k3_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kda_head_dim: int,
    dense_hidden_dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: str = "sigmoid",
    num_expert_groups: int = 1,
    topk_group: int = 1,
    routed_expert_hidden_size: int | None = None,
    latent_moe_use_norm: bool = True,
    routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    kda_layers: list[int] | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    norm_eps: float = 1e-5,
    conv_kernel_size: int = 4,
    gate_lower_bound: float | None = -5.0,
    use_full_rank_gate: bool = True,
    attn_res_block_size: int | None = 12,
) -> list[KimiK3TransformerBlock.Config]:
    """Build per-layer TransformerBlock configs with hybrid KDA/MLA attention."""
    if kda_layers is None:
        # Official pattern uses 1-based layer numbers: three KDA, then one MLA.
        kda_layers = [i for i in range(n_layers) if (i + 1) % 4 != 0]

    layers = []
    for layer_id in range(n_layers):
        # Attention config: KDA or Gated MLA
        if layer_id in kda_layers:
            attn_cfg = KimiDeltaAttention.Config(
                dim=dim,
                num_heads=n_heads,
                head_dim=kda_head_dim,
                conv_kernel_size=conv_kernel_size,
                gate_lower_bound=gate_lower_bound,
                use_full_rank_gate=use_full_rank_gate,
                norm_eps=norm_eps,
            )
        else:
            attn_cfg = KimiGatedMLA.Config(
                dim=dim,
                n_heads=n_heads,
                q_lora_rank=q_lora_rank,
                kv_lora_rank=kv_lora_rank,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                v_head_dim=v_head_dim,
                norm_eps=norm_eps,
            )

        # FFN: dense for first n_dense_layers, MoE for the rest
        is_dense = layer_id < n_dense_layers
        if is_dense:
            ffn = KimiMLP.Config(
                hidden_size=dim,
                intermediate_size=dense_hidden_dim,
                beta=situ_beta,
                linear_beta=situ_linear_beta,
            )
            moe_cfg = None
        else:
            ffn = None
            moe_cfg = KimiSparseMoeBlock.Config(
                hidden_size=dim,
                num_experts=num_experts,
                top_k=router_top_k,
                moe_intermediate_size=moe_inter_dim,
                num_shared_experts=num_shared_experts,
                routed_expert_hidden_size=routed_expert_hidden_size,
                latent_moe_use_norm=latent_moe_use_norm,
                score_func=router_score_func,
                num_expert_groups=num_expert_groups,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                renormalize=renormalize,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                norm_eps=norm_eps,
            )

        layers.append(
            KimiK3TransformerBlock.Config(
                attention=attn_cfg,
                feed_forward=ffn,
                moe=moe_cfg,
                norm_eps=norm_eps,
                dim=dim,
                layer_id=layer_id,
                attn_res_block_size=attn_res_block_size,
            )
        )
    return layers


def _make_kimi_k3_model_config(
    *,
    param_init: dict | None = None,
    vocab_size: int = 163840,
    dim: int = 7168,
    n_layers: int = 93,
    n_dense_layers: int = 1,
    n_heads: int = 96,
    q_lora_rank: int = 1536,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    kda_head_dim: int = 128,
    dense_hidden_dim: int = 33792,
    moe_inter_dim: int = 3072,
    num_experts: int = 896,
    num_shared_experts: int = 2,
    router_top_k: int = 16,
    router_score_func: str = "sigmoid",
    num_expert_groups: int = 1,
    topk_group: int = 1,
    routed_expert_hidden_size: int | None = 3584,
    latent_moe_use_norm: bool = True,
    routed_scaling_factor: float = 1.0,
    renormalize: bool = True,
    kda_layers: list[int] | None = None,
    situ_beta: float = 4.0,
    situ_linear_beta: float | None = 25.0,
    norm_eps: float = 1e-5,
    conv_kernel_size: int = 4,
    gate_lower_bound: float | None = -5.0,
    use_full_rank_gate: bool = True,
    attn_res_block_size: int | None = 12,
) -> KimiK3Model.Config:
    layers = _build_kimi_k3_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kda_head_dim=kda_head_dim,
        dense_hidden_dim=dense_hidden_dim,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=router_top_k,
        router_score_func=router_score_func,
        num_expert_groups=num_expert_groups,
        topk_group=topk_group,
        routed_expert_hidden_size=routed_expert_hidden_size,
        latent_moe_use_norm=latent_moe_use_norm,
        routed_scaling_factor=routed_scaling_factor,
        renormalize=renormalize,
        kda_layers=kda_layers,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        norm_eps=norm_eps,
        conv_kernel_size=conv_kernel_size,
        gate_lower_bound=gate_lower_bound,
        use_full_rank_gate=use_full_rank_gate,
        attn_res_block_size=attn_res_block_size,
    )
    return KimiK3Model.Config(
        param_init=param_init,
        vocab_size=vocab_size,
        dim=dim,
        layers=layers,
        norm_eps=norm_eps,
        attn_res_block_size=attn_res_block_size,
    )


def _debug_model() -> KimiK3Model.Config:
    """Minimal debug config: 4 layers (3 KDA + 1 MLA), 8 experts, small dims."""
    return _make_kimi_k3_model_config(
        vocab_size=2048,
        dim=256,
        n_layers=4,
        n_dense_layers=1,
        n_heads=8,
        q_lora_rank=64,
        kv_lora_rank=64,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        kda_head_dim=32,
        dense_hidden_dim=512,
        moe_inter_dim=128,
        num_experts=8,
        num_shared_experts=2,
        router_top_k=3,
        routed_expert_hidden_size=128,
        kda_layers=[0, 1, 2],
    )


def _16layer_reduced_model() -> KimiK3Model.Config:
    """Reduced 16-layer model matching MindSpeed-MM's A3 single-node config."""
    return _make_kimi_k3_model_config(
        vocab_size=163840,
        dim=7168,
        n_layers=16,
        n_dense_layers=1,
        n_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dense_hidden_dim=33792,
        moe_inter_dim=3072,
        num_experts=32,
        num_shared_experts=2,
        router_top_k=16,
        routed_expert_hidden_size=3584,
    )


def _full_model() -> KimiK3Model.Config:
    """Full Kimi K3 2.8T model: 93 layers, 896 experts, top-16."""
    return _make_kimi_k3_model_config(
        vocab_size=163840,
        dim=7168,
        n_layers=93,
        n_dense_layers=1,
        n_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        dense_hidden_dim=33792,
        moe_inter_dim=3072,
        num_experts=896,
        num_shared_experts=2,
        router_top_k=16,
        routed_expert_hidden_size=3584,
        kda_layers=_FULL_KDA_LAYERS,
    )


kimi_k3_configs = {
    "debug": _debug_model,
    "16layer_reduced": _16layer_reduced_model,
    "full": _full_model,
}


def model_registry(flavor: str) -> ModelSpec:
    model_config = kimi_k3_configs[flavor]()
    return ModelSpec(
        name="kimi_k3",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=KimiK3StateDictAdapter,
    )
