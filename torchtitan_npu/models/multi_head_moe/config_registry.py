# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native training configurations for the SLA2 + Multi-Head-MoE decoder.

The reference geometry (``external_models/multi_head_moe``) is a 2.5T preview:
dim=8192, 80 layers, 32 MoE heads x 896 experts. The debug flavor shrinks the
geometry but keeps every structural path: front-2/back-2 dense layers with
Multi-Head-MoE layers in between, per-head gates over a flattened grouped
expert pool, and the shared expert.
"""

from dataclasses import dataclass
from functools import partial

import torch.nn as nn
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, LocalTokenDispatcher, RoutedExperts
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .model import (
    BlockRouter,
    InnerBlockSLA2,
    MultiHeadGate,
    MultiHeadMoE,
    MultiHeadMoEModel,
    RMSNorm,
    RopeGeometry,
    SLA2Attention,
    TransformerBlock,
)

_LIN = Linear.Config


def _sla2_attention_config(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    block_size: int,
    norm_eps: float,
    rope_theta: float,
    max_seq_len: int,
    sla2_topk_ratio_1m: float,
    sla2_topk_ratio_5m: float,
    router_proj_dim: int,
    temp: float,
) -> SLA2Attention.Config:
    def _head_norm() -> RMSNorm.Config:
        return RMSNorm.Config(dim=head_dim, eps=norm_eps)

    def _head_linear() -> _LIN:
        return _LIN(in_features=dim, out_features=n_heads * head_dim, bias=False)

    return SLA2Attention.Config(
        n_heads=n_heads,
        head_dim=head_dim,
        block_size=block_size,
        sla2_topk_ratio_1m=sla2_topk_ratio_1m,
        sla2_topk_ratio_5m=sla2_topk_ratio_5m,
        rope_theta=rope_theta,
        rope=RopeGeometry(max_seq_len=max_seq_len),
        to_q=_head_linear(),
        to_k=_head_linear(),
        to_v=_head_linear(),
        to_out=_LIN(in_features=n_heads * head_dim, out_features=dim, bias=False),
        norm_q=_head_norm(),
        norm_k=_head_norm(),
        inner=InnerBlockSLA2.Config(
            d_head=head_dim,
            n_heads=n_heads,
            block_size=block_size,
            temp=temp,
            router=BlockRouter.Config(
                proj_q=_LIN(in_features=head_dim, out_features=router_proj_dim, bias=False),
                proj_k=_LIN(in_features=head_dim, out_features=router_proj_dim, bias=False),
            ),
        ),
    )


def _moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_heads: int,
    head_hidden: int,
    experts_per_head: int,
    k_per_head: int,
    expert_inter_mult: int,
    shared_inter_dim: int,
) -> MultiHeadMoE.Config:
    head_total = moe_heads * head_hidden
    # Depth-scaled trunc-normal init on the stacked grouped-expert weights
    # (upstream convention; the patched GroupedExperts has no default init).
    std = depth_scaled_std(0.02, layer_id)
    experts_init = {
        "w1_EFD": partial(nn.init.trunc_normal_, std=std),
        "w2_EDF": partial(nn.init.trunc_normal_, std=std),
        "w3_EFD": partial(nn.init.trunc_normal_, std=std),
    }
    return MultiHeadMoE.Config(
        moe_heads=moe_heads,
        head_hidden=head_hidden,
        proj_in=_LIN(in_features=dim, out_features=head_total, bias=False),
        proj_out=_LIN(in_features=head_total, out_features=dim, bias=False),
        router=MultiHeadGate.Config(
            moe_heads=moe_heads,
            head_hidden=head_hidden,
            experts_per_head=experts_per_head,
            k_per_head=k_per_head,
        ),
        routed_experts=RoutedExperts.Config(
            inner_experts=GroupedExperts.Config(
                dim=head_hidden,
                hidden_dim=head_hidden * expert_inter_mult,
                num_experts=moe_heads * experts_per_head,
                param_init=experts_init,
            ),
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=moe_heads * experts_per_head,
                top_k=k_per_head,
            ),
        ),
        shared_experts=FeedForward.Config(
            w1=_LIN(in_features=dim, out_features=shared_inter_dim, bias=False),
            w2=_LIN(in_features=shared_inter_dim, out_features=dim, bias=False),
            w3=_LIN(in_features=dim, out_features=shared_inter_dim, bias=False),
        ),
    )


def _layer_config(
    *,
    layer_idx: int,
    n_layers: int,
    dim: int,
    inter_dim: int,
    norm_eps: float,
    moe_kwargs: dict,
    attn_kwargs: dict,
) -> TransformerBlock.Config:
    """Front-2 / back-2 dense layers, Multi-Head MoE in between (reference rule)."""
    is_dense = layer_idx < 2 or layer_idx >= n_layers - 2
    return TransformerBlock.Config(
        attention=_sla2_attention_config(**attn_kwargs),
        attention_norm=RMSNorm.Config(dim=dim, eps=norm_eps),
        ffn_norm=RMSNorm.Config(dim=dim, eps=norm_eps),
        feed_forward=(
            FeedForward.Config(
                w1=_LIN(in_features=dim, out_features=inter_dim, bias=False),
                w2=_LIN(in_features=inter_dim, out_features=dim, bias=False),
                w3=_LIN(in_features=dim, out_features=inter_dim, bias=False),
            )
            if is_dense
            else None
        ),
        moe=None if is_dense else _moe_config(layer_id=layer_idx, **moe_kwargs),
    )


@dataclass(frozen=True)
class _Geometry:
    """Debug geometry: every structural path, shrunk numbers.

    The published 2.5T preview is dim=8192 / 80 layers / 32 heads x 896
    experts; the debug flavor keeps the layer split (front-2/back-2 dense),
    the per-head gate and the shared expert with capture-sized widths.
    """

    dim: int = 256
    vocab_size: int = 102400
    n_layers: int = 8
    n_heads: int = 4
    head_dim: int = 64
    inter_dim: int = 512
    block_size: int = 64
    router_proj_dim: int = 64
    temp: float = 0.1
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    sla2_topk_ratio_1m: float = 0.005
    sla2_topk_ratio_5m: float = 0.001
    moe_heads: int = 4
    head_hidden: int = 64
    experts_per_head: int = 16
    k_per_head: int = 4
    expert_inter_mult: int = 2
    shared_inter_dim: int = 128
    max_seq_len: int = 4096


def _make_model_config(
    *,
    n_layers: int,
    dim: int,
    vocab_size: int,
    inter_dim: int,
    norm_eps: float,
    moe_kwargs: dict,
    attn_kwargs: dict,
) -> MultiHeadMoEModel.Config:
    return MultiHeadMoEModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
        ),
        norm=RMSNorm.Config(dim=dim, eps=norm_eps),
        lm_head=_LIN(in_features=dim, out_features=vocab_size, bias=False),
        layers=[
            _layer_config(
                layer_idx=layer_idx,
                n_layers=n_layers,
                dim=dim,
                inter_dim=inter_dim,
                norm_eps=norm_eps,
                moe_kwargs=moe_kwargs,
                attn_kwargs=attn_kwargs,
            )
            for layer_idx in range(n_layers)
        ],
    )


def _debugmodel_model_config() -> MultiHeadMoEModel.Config:
    g = _Geometry()
    attn_kwargs = dict(
        dim=g.dim,
        n_heads=g.n_heads,
        head_dim=g.head_dim,
        block_size=g.block_size,
        norm_eps=g.norm_eps,
        rope_theta=g.rope_theta,
        max_seq_len=g.max_seq_len,
        sla2_topk_ratio_1m=g.sla2_topk_ratio_1m,
        sla2_topk_ratio_5m=g.sla2_topk_ratio_5m,
        router_proj_dim=g.router_proj_dim,
        temp=g.temp,
    )
    moe_kwargs = dict(
        dim=g.dim,
        moe_heads=g.moe_heads,
        head_hidden=g.head_hidden,
        experts_per_head=g.experts_per_head,
        k_per_head=g.k_per_head,
        expert_inter_mult=g.expert_inter_mult,
        shared_inter_dim=g.shared_inter_dim,
    )
    return _make_model_config(
        n_layers=g.n_layers,
        dim=g.dim,
        vocab_size=g.vocab_size,
        inter_dim=g.inter_dim,
        norm_eps=g.norm_eps,
        moe_kwargs=moe_kwargs,
        attn_kwargs=attn_kwargs,
    )


multi_head_moe_configs = {
    "debugmodel": _debugmodel_model_config,
}


def model_registry(flavor: str) -> ModelSpec:
    from .parallelize import parallelize_multi_head_moe

    if flavor not in multi_head_moe_configs:
        raise ValueError(f"Unknown multi_head_moe flavor: {flavor}. Available: {list(multi_head_moe_configs)}")
    return ModelSpec(
        name="multi_head_moe",
        flavor=flavor,
        model=multi_head_moe_configs[flavor](),
        parallelize_fn=parallelize_multi_head_moe,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def multi_head_moe_debugmodel() -> Trainer.Config:
    model_spec = model_registry("debugmodel")
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        optimizer=default_adamw(lr=8e-4),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            seq_len=128,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=FullAC.Config(),
    )
