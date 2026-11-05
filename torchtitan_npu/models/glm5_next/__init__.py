# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5.3-Flash model package: mHC + hybrid KDA/DSA fine-grained MoE VLM."""

from functools import partial

import torch.nn as nn
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm

from .model import Glm5NextModel, Glm5NextMultimodalModel
from .parallelize import parallelize_glm5_next
from .state_dict_adapter import Glm5NextStateDictAdapter

__all__ = [
    "Glm5NextModel",
    "Glm5NextMultimodalModel",
    "Glm5NextStateDictAdapter",
    "glm5_next_configs",
    "model_registry",
    "parallelize_glm5_next",
]

_EPS = 1e-6

_DEFAULT_SWIGLU_LIMIT = 10.0
_DEFAULT_LOWER_BOUND = -5.0

_WEIGHT_INIT = partial(nn.init.normal_, mean=0.0, std=0.02)
_LINEAR_INIT = {"weight": _WEIGHT_INIT}

IMAGE_TOKEN_ID = 1998  # <|image|> id in tests/assets/tokenizer/vlm_tokenizer


def _kda_config(dim: int, *, num_heads, head_dim, lower_bound=_DEFAULT_LOWER_BOUND):
    from .kda import KDALinearAttention

    return KDALinearAttention.Config(
        dim=dim,
        num_heads=num_heads,
        head_dim=head_dim,
        norm_eps=_EPS,
        lower_bound=lower_bound,
    )


def _dsa_config(
    dim: int,
    *,
    num_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    index_n_heads,
    index_head_dim,
    index_topk,
    index_kpool,
):
    from .attention import DSAAttention, DSAIndexer

    return DSAAttention.Config(
        dim=dim,
        num_attention_heads=num_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        norm_eps=_EPS,
        indexer=DSAIndexer.Config(
            dim=dim,
            q_lora_rank=q_lora_rank,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            index_kpool=index_kpool,
        ),
    )


def _dense_config(dim: int, intermediate: int):
    from .feed_forward import GlmMLP

    return GlmMLP.Config(dim=dim, hidden_dim=intermediate, swiglu_limit=_DEFAULT_SWIGLU_LIMIT)


def _moe_config(dim: int, *, num_experts, top_k, moe_intermediate, n_shared=1):
    from .feed_forward import GlmMoE

    return GlmMoE.Config(
        dim=dim,
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_intermediate,
        n_shared_experts=n_shared,
        swiglu_limit=_DEFAULT_SWIGLU_LIMIT,
    )


def _build_layers(
    *,
    n_layers: int,
    dim: int,
    first_k_dense: int,
    dense_intermediate: int,
    kda_kwargs: dict,
    dsa_kwargs: dict,
    moe_kwargs: dict,
    hc_mult: int,
) -> list["Glm5NextBlock.Config"]:
    from .model import Glm5NextBlock

    layers = []
    for layer_id in range(n_layers):
        mixer = _kda_config(dim, **kda_kwargs) if (layer_id % 4) != 3 else _dsa_config(dim, **dsa_kwargs)
        if layer_id < first_k_dense:
            ffn, moe = _dense_config(dim, dense_intermediate), None
        else:
            ffn, moe = None, _moe_config(dim, **moe_kwargs)
        layers.append(
            Glm5NextBlock.Config(
                attention=mixer,
                layer_id=layer_id,
                attention_norm=RMSNorm.Config(normalized_shape=dim, eps=_EPS),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, eps=_EPS),
                feed_forward=ffn,
                moe=moe,
                hc_mult=hc_mult,
            )
        )
    return layers


def _vision_config(image_size: int = 448, depth: int = 24, hidden_size: int = 1024):
    from .vision import GlmVisionTower

    return GlmVisionTower.Config(
        hidden_size=hidden_size,
        depth=depth,
        num_heads=16,
        intermediate_size=4096,
        out_hidden_size=4096,
        projection_intermediate_size=10240,
        image_size=image_size,
        norm_eps=1e-5,
    )


def _model_config(
    *,
    dim: int,
    vocab_size: int,
    layers: list,
    vision=None,
) -> "Glm5NextModel.Config":
    model_cls = Glm5NextMultimodalModel if vision is not None else Glm5NextModel
    return model_cls.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_LINEAR_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=_EPS),
        layers=layers,
        image_token_id=IMAGE_TOKEN_ID,
        vision_encoder=vision,
    )


def _debugmodel() -> Glm5NextMultimodalModel.Config:
    """Minimal debug config exercising every path: 4 layers (3 KDA + 1
    DSA), dense first-2, MoE last-2 (8 experts top-2), mHC x4, vision."""
    dim = 256
    vocab_size = 2048
    n_layers = 4
    return _model_config(
        dim=dim,
        vocab_size=vocab_size,
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            first_k_dense=2,
            dense_intermediate=512,
            kda_kwargs=dict(num_heads=8, head_dim=32),
            dsa_kwargs=dict(
                num_heads=8,
                q_lora_rank=64,
                kv_lora_rank=64,
                qk_nope_head_dim=32,
                qk_rope_head_dim=0,
                v_head_dim=32,
                index_n_heads=4,
                index_head_dim=32,
                index_topk=32,
                index_kpool=8,
            ),
            moe_kwargs=dict(
                num_experts=8,
                top_k=2,
                moe_intermediate=128,
            ),
            hc_mult=4,
        ),
        vision=_vision_config(image_size=56, depth=2, hidden_size=64),
    )


def _debugmodel_text() -> Glm5NextModel.Config:
    """The debugmodel geometry with the vision tower detached: the text-only
    capture flavor (matches the native text-only SFT recipe)."""
    config = _debugmodel()
    config.vision_encoder = None
    return Glm5NextModel.Config(
        vocab_size=config.vocab_size,
        dim=config.dim,
        tok_embeddings=config.tok_embeddings,
        lm_head=config.lm_head,
        norm=config.norm,
        layers=config.layers,
        image_token_id=config.image_token_id,
        vision_encoder=None,
    )


def _reduced_2l() -> Glm5NextModel.Config:
    """Full production dims trimmed to 4 layers (3 KDA + 1 DSA), first-2
    dense + MoE — the parallelism-matrix comparison config (text-only)."""
    return _full(trim_layers=4)


def _full(trim_layers: int | None = None, *, num_experts: int = 288, index_kpool: int = 16) -> Glm5NextModel.Config:
    """GLM-5.3-Flash production dims (GLM-5.3-Flash-BF16 HF config).

    text: 45 layers (34 KDA + 11 DSA), hidden 4096, vocab 154880, NoPE,
    mHC x4 (sinkhorn 20); every 4th layer DSA (MLA + lightning indexer);
    first 3 dense MLP (inter 12288), 42 MoE layers (288 experts top-8 +
    1 shared, moe_inter 2048, sigmoid routing, scaling 2.5, clamp 10).
    """
    from .model import Glm5NextModel  # noqa: F401

    n_layers = 45 if trim_layers is None else trim_layers
    dim = 4096
    return _model_config(
        dim=dim,
        vocab_size=154880,
        layers=_build_layers(
            n_layers=n_layers,
            dim=dim,
            first_k_dense=3,
            dense_intermediate=12288,
            kda_kwargs=dict(num_heads=64, head_dim=128),
            dsa_kwargs=dict(
                num_heads=64,
                q_lora_rank=1536,
                kv_lora_rank=512,
                qk_nope_head_dim=256,
                qk_rope_head_dim=0,
                v_head_dim=256,
                index_n_heads=32,
                index_head_dim=128,
                index_topk=2048,
                index_kpool=index_kpool,
            ),
            moe_kwargs=dict(
                num_experts=num_experts,
                top_k=8,
                moe_intermediate=2048,
            ),
            hc_mult=4,
        ),
    )


def _910b_test() -> Glm5NextModel.Config:
    """The 910B3 functional-test geometry (glm53_flash_910b_test.yaml):
    production dims trimmed to 4 layers (3 KDA + 1 DSA), 48 routed experts
    + 1 shared, indexer kpool 4. Text-only (the real run's vision tower is
    outside the capture graph)."""
    return _full(trim_layers=4, num_experts=48, index_kpool=4)


glm5_next_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_text": _debugmodel_text,
    "reduced-2L": _reduced_2l,
    "GLM-5.3-Flash": _full,
    "910b-test": _910b_test,
}


def model_registry(flavor: str) -> "Glm5NextModel.Config | Glm5NextMultimodalModel.Config":
    if flavor not in glm5_next_configs:
        raise ValueError(f"Unknown glm5_next flavor: {flavor}. Available: {list(glm5_next_configs.keys())}")
    return glm5_next_configs[flavor]()
