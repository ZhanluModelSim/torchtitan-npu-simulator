# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast

from torchtitan.config import derive
from torchtitan.config.transform import ModelConfigConverter, validate_converter_compatibility
from torchtitan.models.common import (
    ComplexRoPE,
    CosSinRoPE,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    Softmax,
)
from torchtitan.models.common.aux_loss import (
    register_aux_loss_zero_hook,
)
from torchtitan.models.common.moe import register_moe_load_balancing_hook
from torchtitan.models.deepseek_v3 import (
    _EMBEDDING_INIT,
    _LINEAR_INIT,
    _NORM_INIT,
    DeepSeekV3TransformerBlock,
    _build_dsv3_layers,
    _build_mtp_layers,
    _output_linear_init,
)

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .model import Attention, DeepSeekV32Model, Indexer, SparseInnerAttention
from .state_dict_adapter import DeepSeekV32StateDictAdapter

__all__ = [
    "deepseekv3_2_configs",
]


def _build_dsv3_2_layers(
    *,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    **kwargs,
):
    layers = _build_dsv3_layers(
        enable_sp=False,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        rope=rope,
        # Not a knob: every layer's inner attention is replaced below, so whatever this builds
        # is thrown away.  It is spelled ``varlen`` -- the type the replacement actually is --
        # only so the intermediate config never carries flex semantics a reader could mistake
        # for the deployed backend; do not delete it as dead code.
        attn_backend="varlen",
        **kwargs,
    )

    for layer_cfg in layers:
        layer_cfg.attention = derive(
            layer_cfg.attention,
            Attention.Config,
            inner_attention=SparseInnerAttention.Config(index_topk=index_topk),
            indexer=Indexer.Config(
                dim=dim,
                q_lora_rank=q_lora_rank,
                index_n_heads=index_n_heads,
                index_head_dim=index_head_dim,
                rope_head_dim=qk_rope_head_dim,
                index_topk=index_topk,
                wq_b=Linear.Config(
                    in_features=q_lora_rank,
                    out_features=index_n_heads * index_head_dim,
                    param_init=_LINEAR_INIT,
                ),
                wk=Linear.Config(
                    in_features=dim,
                    out_features=index_head_dim,
                    param_init=_LINEAR_INIT,
                ),
                k_norm=LayerNorm.Config(normalized_shape=index_head_dim),
                weights_proj=Linear.Config(
                    in_features=dim,
                    out_features=index_n_heads,
                    param_init=_LINEAR_INIT,
                ),
                rope=CosSinRoPE.Config(
                    dim=qk_rope_head_dim,
                    max_context_length=rope.max_context_length,
                    theta=rope.theta,
                    scaling=rope.scaling,
                    rope_factor=rope.rope_factor,
                    beta_fast=rope.beta_fast,
                    beta_slow=rope.beta_slow,
                    original_seq_len=rope.original_seq_len,
                ),
            ),
            w_uk=BatchedLinear.Config(
                n_heads=n_heads,
                in_features=qk_nope_head_dim,
                out_features=kv_lora_rank,
                param_init=_LINEAR_INIT,
            ),
            w_uv=BatchedLinear.Config(
                n_heads=n_heads,
                in_features=kv_lora_rank,
                out_features=v_head_dim,
                param_init=_LINEAR_INIT,
            ),
        )
    return layers


def _debugmodel_v3_2(
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    num_mtp_layers: int = 0,
    max_context_length: int = 16384,
) -> DeepSeekV32Model.Config:
    dim = 7168
    n_layers = 1
    vocab_size = 129280
    n_heads = 64
    q_lora_rank = 1536
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    v_head_dim = 128
    index_n_heads = 32
    index_head_dim = 128
    index_topk = 1024  # 2048 being too large for AICPU
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    num_experts = 8
    n_dense_layers = 0

    layers = _build_dsv3_2_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func=Softmax.Config(),
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=ComplexRoPE.Config(
            dim=qk_rope_head_dim,
            max_context_length=max_context_length,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
    )
    return DeepSeekV32Model.Config(
        max_context_length=max_context_length,
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        mtp_layers=_build_mtp_layers(
            cast("DeepSeekV3TransformerBlock.Config", layers[-1]),
            dim=dim,
            num_mtp_layers=num_mtp_layers,
        ),
    )


deepseekv3_2_configs = {
    "debugmodel": _debugmodel_v3_2,
}


def _post_optimizer_build_fn(optimizers, model_parts, parallelism_context):
    register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
    register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    num_mtp_layers: int = 0,
    converters: list[ModelConfigConverter.Config] | None = None,
    seq_len: int | None = None,
) -> DeepSeekV32Model.Config:
    config = deepseekv3_2_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        num_mtp_layers=num_mtp_layers,
        max_context_length=16384 if seq_len is None else seq_len,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for c in converters:
            config = c.build().convert(config)
    return config


DeepSeekV32Model.state_dict_adapter_cls = DeepSeekV32StateDictAdapter
