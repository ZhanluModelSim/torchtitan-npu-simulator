# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""mm_gc model registration and config builders.

Provides model_registry() for torchtitan's ModelSpec discovery and layer
builders that assemble the hybrid SLA2 / Multi-Head-MoE decoder layers.
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.protocols.model_spec import ModelSpec

from .attention import SLA2Attention
from .feed_forward import MMGcMLP, MultiHeadMoE
from .model import MMGcModel, MMGcTransformerBlock
from .parallelize import parallelize_mm_gc
from .state_dict_adapter import MMGcStateDictAdapter


def _make_layer_configs(
    *,
    layer_ids: list[int],
    dense_layer_ids: set[int],
    dim: int,
    n_heads: int,
    head_dim: int,
    seq_len: int,
    norm_eps: float,
    rope_theta: float,
    dense_inter_dim: int,
    moe_num_heads: int,
    moe_head_hidden_size: int,
    experts_per_head: int,
    top_k: int,
    moe_expert_inter_mult: int,
    moe_shared_inter_dim: int,
    sla2_topk_1m: float,
    sla2_topk_5m: float,
    sla2_stage: int,
    sla2_router_data_path: str | None,
) -> list[MMGcTransformerBlock.Config]:
    layers = []
    for layer_id in layer_ids:
        attention_cfg = SLA2Attention.Config(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            seq_len=seq_len,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            sla2_topk_1m=sla2_topk_1m,
            sla2_topk_5m=sla2_topk_5m,
            sla2_stage=sla2_stage,
            sla2_router_data_path=sla2_router_data_path,
            layer_idx=layer_id,
        )
        if layer_id in dense_layer_ids:
            ffn_cfg = MMGcMLP.Config(
                hidden_size=dim,
                intermediate_size=dense_inter_dim,
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = MultiHeadMoE.Config(
                hidden_size=dim,
                moe_num_heads=moe_num_heads,
                moe_head_hidden_size=moe_head_hidden_size,
                experts_per_head=experts_per_head,
                top_k=top_k,
                moe_expert_inter_mult=moe_expert_inter_mult,
                moe_shared_inter_dim=moe_shared_inter_dim,
            )
        layers.append(
            MMGcTransformerBlock.Config(
                attention=attention_cfg,
                feed_forward=ffn_cfg,
                moe=moe_cfg,
                norm_eps=norm_eps,
                dim=dim,
                layer_id=layer_id,
            )
        )
    return layers


def _make_model_config(
    *,
    vocab_size: int,
    dim: int,
    n_layers: int,
    seq_len: int,
    n_heads: int,
    head_dim: int,
    norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    dense_inter_dim: int,
    moe_num_heads: int,
    moe_head_hidden_size: int,
    experts_per_head: int,
    top_k: int,
    moe_expert_inter_mult: int = 4,
    moe_shared_inter_dim: int,
    sla2_topk_1m: float = 0.005,
    sla2_topk_5m: float = 0.001,
    sla2_stage: int = 1,
    sla2_router_data_path: str | None = None,
) -> MMGcModel.Config:
    dense_layer_ids = {i for i in range(n_layers) if i < 2 or i >= n_layers - 2}
    layers = _make_layer_configs(
        layer_ids=list(range(n_layers)),
        dense_layer_ids=dense_layer_ids,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        norm_eps=norm_eps,
        rope_theta=rope_theta,
        dense_inter_dim=dense_inter_dim,
        moe_num_heads=moe_num_heads,
        moe_head_hidden_size=moe_head_hidden_size,
        experts_per_head=experts_per_head,
        top_k=top_k,
        moe_expert_inter_mult=moe_expert_inter_mult,
        moe_shared_inter_dim=moe_shared_inter_dim,
        sla2_topk_1m=sla2_topk_1m,
        sla2_topk_5m=sla2_topk_5m,
        sla2_stage=sla2_stage,
        sla2_router_data_path=sla2_router_data_path,
    )
    return MMGcModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        seq_len=seq_len,
        rope_theta=rope_theta,
        layers=layers,
        norm_eps=norm_eps,
    )


def _debug_model() -> MMGcModel.Config:
    return _make_model_config(
        vocab_size=2048,
        dim=256,
        n_layers=6,
        seq_len=128,
        n_heads=8,
        head_dim=32,
        dense_inter_dim=512,
        moe_num_heads=4,
        moe_head_hidden_size=64,
        experts_per_head=16,
        top_k=4,
        moe_shared_inter_dim=256,
        sla2_topk_1m=0.5,
        sla2_topk_5m=0.5,
    )


def _reduced_model() -> MMGcModel.Config:
    return _make_model_config(
        vocab_size=102400,
        dim=12288,
        n_layers=16,
        seq_len=4096,
        n_heads=96,
        head_dim=128,
        dense_inter_dim=16384,
        moe_num_heads=32,
        moe_head_hidden_size=384,
        experts_per_head=32,
        top_k=6,
        moe_shared_inter_dim=6144,
    )


def _full_model() -> MMGcModel.Config:
    return _make_model_config(
        vocab_size=102400,
        dim=12288,
        n_layers=96,
        seq_len=4096,
        n_heads=96,
        head_dim=128,
        dense_inter_dim=16384,
        moe_num_heads=32,
        moe_head_hidden_size=384,
        experts_per_head=1024,
        top_k=6,
        moe_shared_inter_dim=6144,
    )


mm_gc_configs = {
    "debug": _debug_model,
    "reduced": _reduced_model,
    "full": _full_model,
}


def model_registry(flavor: str) -> ModelSpec:
    model_config = mm_gc_configs[flavor]()
    return ModelSpec(
        name="mm_gc",
        flavor=flavor,
        model=model_config,
        parallelize_fn=parallelize_mm_gc,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=MMGcStateDictAdapter,
    )
