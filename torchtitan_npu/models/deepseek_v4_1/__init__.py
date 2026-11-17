# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model_spec import ModelSpec

from torchtitan_npu.patches.torchtitan.models.common.linear import BatchedLinear

from .attention import Attention, CompressedSparseInnerAttention2
from .compressor import Compressor
from .indexer import (
    FULL,
    REINDEX,
    REUSE,
    HierarchicalIndexer,
    Selector,
)
from .mhc import HcPost, HcPre
from .model import DeepSeekV41Model, DeepSeekV41MultimodalModel, DeepSeekV41TransformerBlock
from .state_dict_adapter import DeepSeekV41StateDictAdapter

# std=1.0 mirrors the reference debug initialization; the marker values
# are the reference marker values, so changing the std changes the marker
# parameterization.
_MARKER_INIT = {name: partial(nn.init.normal_, std=1.0) for name in ("image_start", "image_newline", "image_end")}

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE
    from torchtitan.protocols.model import ModelConfigConverter


@dataclass(frozen=True, kw_only=True)
class EngramArgs:
    """Model-flavor arguments for attaching Engram to selected layers.

    ``vocab_size_per_ngram`` gives the target rows per hash head for each
    N-gram order; the actual sizes are distinct primes at least that large.
    ``n_embed_per_ngram`` is the concatenated width of all hash heads for one
    order. For example, 640 with 8 heads produces 80 values per fetched row
    and a total memory width of 1280 for orders 2 and 3.
    """

    layer_ids: tuple[int, ...]
    vocab_size_per_ngram: tuple[int, ...]
    n_embed_per_ngram: int
    ngram_orders: tuple[int, ...] = (2, 3)
    num_heads_per_ngram: int = 8
    pad_id: int = 0
    hash_seed: int = 0
    norm_eps: float = 1e-5
    # A fixed model-shape alignment. The runtime EP degree must divide it;
    # this avoids changing checkpoint tensor shapes when EP changes. Published
    # V4.1 table sizes are logical row counts; this training layout adds padding
    # without changing the hash bucket ranges.
    table_padding_multiple: int = 2048
    token_id_map_path: str | None = None
    require_token_id_map: bool = True
    # When set, the compressed vocabulary the tokenizer map must produce. The
    # compressed size bounds the hash multipliers, so a map from a different
    # tokenizer silently changes every row ID; checking it turns that into a
    # startup error.
    compressed_vocab_size: int | None = None


# Published DeepSeek-V4.1-Flash text_config, revision 2bc89ac599031fa673cab993f1df02fc4a98c673:
# https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/2bc89ac599031fa673cab993f1df02fc4a98c673
_ENGRAM_V41_FLASH_GEOMETRY = EngramArgs(
    layer_ids=(1, 14),
    ngram_orders=(2, 3, 4),
    vocab_size_per_ngram=(16_000_000, 16_000_000, 16_000_000),
    n_embed_per_ngram=2048,
    num_heads_per_ngram=8,
    pad_id=2,
    norm_eps=1e-20,
    require_token_id_map=True,
    compressed_vocab_size=99_092,
)


_ENGRAM_DEBUG_GEOMETRY = EngramArgs(
    layer_ids=(1, 14),
    ngram_orders=(2, 3, 4),
    vocab_size_per_ngram=(1024, 1024, 1024),
    n_embed_per_ngram=256,
    num_heads_per_ngram=2,
    pad_id=2,
    norm_eps=1e-20,
    require_token_id_map=False,
)


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_SINK_INIT = {"attn_sink": partial(nn.init.trunc_normal_, std=0.02)}
_HC_PARAM_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": partial(nn.init.trunc_normal_, std=0.02),
    "hc_scale": partial(nn.init.trunc_normal_, std=0.02),
}
_SWIGLU_LIMIT = 10.0
# Positions per candidate block of the hierarchical indexer.  A compile-time constant of
# both fused kernels (``CANDIDATE_BLOCK_SIZE`` in the QLI and QSLI sources), not a tunable.
CANDIDATE_BLOCK_SIZE = 8


# Flash and debug widths share the same 40-layer KV/indexer reuse topology.
V41_CANDIDATE_SOURCE_LAYER = 20
V41_FULL_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
V41_KV_SOURCE_LAYERS = (2, 8, 14, 20)
V41_FULL_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)

# The miniature topology the single-card fused-kernel runs use: the released recipe cut to
# the three layers that reach every branch of the fused selector -- one Full Mode layer at
# ratio 2 with the pool off, one Full Mode pool source at ratio 1, and one Reindex Mode
# layer that searches what the source built.  Every additional layer would only recompile
# the same kernels, and the cannbotdsl JIT dominates the cost of a cold run.
V41_STRIPPED_COMPRESS_RATIOS = (2, 1, 1)
V41_STRIPPED_KV_SOURCE_LAYERS = (0, 1)
V41_STRIPPED_INDEX_SOURCE_LAYERS = (0, 1, 2)
V41_STRIPPED_CANDIDATE_SOURCE_LAYER = 1


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _make_compressor_config(
    *,
    dim: int,
    head_dim: int,
    compress_ratio: int,
    norm_eps: float,
    is_source: bool,
    rope: ComplexRoPE.Config | None,
) -> Compressor.Config:
    """Main-KV compressor config. A reusing layer holds no weights of its own.

    ``head_dim`` sizes the projections and the norm; the rotated span of the rope is
    carried by the rope config's own ``dim``/``split``.
    """
    return Compressor.Config(
        compress_ratio=compress_ratio,
        is_source=is_source,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wkv=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        # The softmax gate only exists when there is more than one token to pool.
        wgate=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source and compress_ratio > 1
            else None
        ),
        norm=(
            RMSNorm.Config(
                normalized_shape=head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            )
            if is_source
            else None
        ),
    )


def _make_indexer_config(
    *,
    dim: int,
    q_lora_rank: int,
    head_dim: int,
    index_head_dim: int,
    num_index_heads: int,
    index_topk: int,
    compress_ratio: int,
    norm_eps: float,
    is_source: bool,
    owns_k: bool,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    rope: ComplexRoPE.Config | None,
) -> HierarchicalIndexer.Config:
    """The indexer config for one layer, in the mode its role implies.

    A layer that owns the compressed KV is Full Mode (it projects its own index keys); an
    index source without it is Reindex Mode (it rescores the shared keys); every other
    layer is Reuse Mode.  The candidate pool is built by the Full Mode source and searched
    by the Reindex Mode layers after it, so only those carry a pool capacity; every other
    layer spells the pool off with the kernels' own sentinel.
    """
    owns_index_k = owns_k and is_source
    if not is_source:
        mode = REUSE
    elif owns_index_k:
        mode = FULL
    else:
        mode = REINDEX
    wants_pool = is_candidate_source or uses_candidates
    # A layer outside the hierarchy spells the pool off with the kernels' negative sentinel;
    # only the source and the Reindex Mode layers after it carry a capacity.
    pool_capacity = candidate_topk_blocks if wants_pool else -1
    pool_block_size = candidate_block_size if wants_pool else -1
    return HierarchicalIndexer.Config(
        mode=mode,
        selector=Selector.Config(
            mode=mode,
            compress_ratio=compress_ratio,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            candidate_topk_blocks=pool_capacity,
            candidate_block_size=pool_block_size,
        ),
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wq_b=(
            Linear.Config(
                in_features=q_lora_rank,
                out_features=num_index_heads * index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        weights_proj=(
            Linear.Config(
                in_features=dim,
                out_features=num_index_heads,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        wk=(
            Linear.Config(
                in_features=head_dim,
                out_features=index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_index_k
            else None
        ),
        k_norm=(
            RMSNorm.Config(
                normalized_shape=index_head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            )
            if owns_index_k
            else None
        ),
    )


def _make_v41_attn_config(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratio: int,
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope: ComplexRoPE.Config,
    owns_compressor: bool,
    owns_indexer: bool,
    source_key: bool,
    external_key: bool,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    layer_id: int,
    index_source_layers: tuple[int, ...],
) -> Attention.Config:
    if source_key and external_key:
        raise ValueError("an indexer cannot own its source-key projection and consume an external key at the same time")
    if is_candidate_source and not owns_compressor:
        raise ValueError("the candidate-pool source must own the compressed KV it scores")

    hd = head_dim
    # Every rope config rotates the trailing ``rope_head_dim`` channels of its site's
    # head, so each site carries the un-rotated prefix width of that head.
    attention_rope = dataclasses.replace(rope, split=hd - rope_head_dim)
    indexer_rope = dataclasses.replace(rope, split=index_head_dim - rope_head_dim)
    per_group_in = (n_heads * hd) // n_groups
    per_group_out = n_groups * o_lora_rank
    softmax_scale = head_dim**-0.5
    # Every layer carries a compressor; only the sources carry its weights and its rope.
    compressor_cfg = _make_compressor_config(
        dim=dim,
        head_dim=hd,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        is_source=owns_compressor,
        rope=attention_rope if owns_compressor else None,
    )
    # Every layer carries an indexer; only a source carries its weights and its rope.
    indexer_cfg = _make_indexer_config(
        dim=dim,
        q_lora_rank=q_lora_rank,
        head_dim=hd,
        index_head_dim=index_head_dim,
        num_index_heads=index_n_heads,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        is_source=owns_indexer,
        owns_k=source_key,
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        rope=indexer_rope if owns_indexer else None,
    )

    # The inner config states only what the reference core reads: its window, its scale and
    # its own ratio, which is the divisor that turns the row's boundaries into the pooled
    # axis it addresses.  The *kernel* geometry -- heads, head dim, the indexer's numbers --
    # is not here: only a fused port's records need it, so it is stated where that port is
    # configured (``sparse_attn.asc`` and its arguments, defaulting to the Flash shape), and a
    # stack of three ratios builds three entries rather than every layer carrying the whole
    # ratio table.
    inner_attention_cfg = CompressedSparseInnerAttention2.Config(
        window_size=window_size,
        softmax_scale=softmax_scale,
        compress_ratio=compress_ratio,
    )
    return Attention.Config(
        n_heads=n_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        compress_ratio=compress_ratio,
        inner_attention=inner_attention_cfg,
        rope=attention_rope,
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=hd,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wo_a=BatchedLinear.Config(
            n_heads=n_groups,
            in_features=per_group_in,
            out_features=o_lora_rank,
            param_init=_LINEAR_INIT,
        ),
        wo_b=Linear.Config(
            in_features=per_group_out,
            out_features=dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        param_init=_SINK_INIT,
        compressor=compressor_cfg,
        indexer=indexer_cfg,
    )


def _make_v41_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
    route_norm: bool,
    load_balance_coeff: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    vision_enabled: bool,
) -> MoE.Config:
    # Every block carries the same MoE stack, so one config serves them all; the common
    # factories build it exactly as upstream's V4.1 does (the plugin patch gives them the
    # ``sqrtsoftplus`` score function and the ``swiglu_limit`` passthrough).
    router = make_router_config(
        dim=dim,
        num_experts=num_experts,
        gate_param_init=_depth_init(layer_id),
        top_k=top_k,
        score_func="sqrtsoftplus",  # pyrefly: ignore [bad-argument-type]
        route_norm=route_norm,
        route_scale=route_scale,
    )
    # The plugin's two router extensions, as two independent switches: the
    # image-masked vision bias follows the stack's modality -- a text-only stack has no
    # image token to bias, so it builds no ``bias_vl`` at all -- while the reference's
    # descending selection order holds for every V4.1 layer.
    router = dataclasses.replace(router, vision_enabled=vision_enabled, sorted_topk=True)
    return make_moe_config(
        num_experts=num_experts,
        router=router,
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=moe_inter_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_depth_experts_init(layer_id),
            comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
        ),
        shared_experts=(
            make_ffn_config(
                dim=dim,
                hidden_dim=moe_inter_dim * num_shared_experts,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
                swiglu_limit=_SWIGLU_LIMIT,  # pyrefly: ignore [unexpected-keyword]
            )
            if num_shared_experts > 0
            else None
        ),
        load_balance_coeff=load_balance_coeff,
    )


def _make_v41_config(
    *,
    n_layers: int,
    vocab_size: int,
    window_size: int,
    norm_eps: float,
    hc_mult: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
    route_norm: bool,
    load_balance_coeff: float,
    sinkhorn_iters: int,
    hc_eps: float,
    rope_theta: float,
    compress_rope_theta: float,
    rope_factor: float,
    original_seq_len: int,
    vision_layers: int,
    patch_size: int,
    downsample_ratio: int,
    compress_ratios: tuple[int, ...],
    kv_source_layers: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    candidate_source_layer: int,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    vision: bool,
    num_experts: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    moe_inter_dim: int,
    candidate_topk_blocks: int,
    vision_dim: int,
    vision_heads: int,
    vision_inter_dim: int,
    max_seq_len: int,
) -> DeepSeekV41Model.Config | DeepSeekV41MultimodalModel.Config:
    # Per-layer source invariants. A reusing layer derives its container grid and its
    # index mask from its own ``compress_ratio``, so the ratio of a layer must equal the
    # ratio of the source it consumes; a topology that breaks this would silently read
    # another ratio's plan.
    if len(compress_ratios) != n_layers:
        raise ValueError(f"compress_ratios must match n_layers ({n_layers}), got {len(compress_ratios)}")
    for name, sources in (("kv_source_layers", kv_source_layers), ("index_source_layers", index_source_layers)):
        outside = sorted({source for source in sources if source < 0 or source >= n_layers})
        if outside:
            raise ValueError(f"{name} outside the {n_layers}-layer crop: {outside}")
    for layer_id, ratio in enumerate(compress_ratios):
        kv_source = max((source for source in kv_source_layers if source <= layer_id), default=None)
        if ratio > 0 and kv_source is None:
            raise ValueError(f"layer {layer_id} has compress_ratio={ratio} but no KV source precedes it")
        if kv_source is not None and compress_ratios[kv_source] != ratio:
            raise ValueError(
                f"layer {layer_id} (compress_ratio={ratio}) consumes the compressed KV of layer "
                f"{kv_source} (compress_ratio={compress_ratios[kv_source]}); the ratios must match"
            )
        if layer_id in index_source_layers and layer_id not in kv_source_layers:
            index_source = max((source for source in index_source_layers if source < layer_id), default=None)
            if index_source is None:
                raise ValueError(f"re-indexing layer {layer_id} has no preceding index source")
            if compress_ratios[index_source] != ratio:
                raise ValueError(
                    f"layer {layer_id} (compress_ratio={ratio}) scores the shared index keys of layer "
                    f"{index_source} (compress_ratio={compress_ratios[index_source]}); the ratios must match"
                )
    if candidate_source_layer not in set(index_source_layers):
        raise ValueError(
            "the candidate-pool source must also be an index source: "
            f"layer {candidate_source_layer} is not in index_source_layers"
        )
    if candidate_topk_blocks <= 0:
        raise ValueError(f"the candidate pool capacity must be positive, got {candidate_topk_blocks}")

    source_key_indexer_layers = tuple(layer_id for layer_id in index_source_layers if layer_id in kv_source_layers)
    external_key_indexer_layers = tuple(
        layer_id for layer_id in index_source_layers if layer_id not in kv_source_layers
    )

    # The text tower's three rope sites share one public ComplexRoPE.Config;
    # reference and fused stacks pick the implementation via the public
    # override entries (workaround / asc_complex / asc_partial).
    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=rope_theta,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=rope_factor,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    layers = []
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id]
        attn_cfg = _make_v41_attn_config(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            n_groups=n_groups,
            compress_ratio=cr,
            window_size=window_size,
            norm_eps=norm_eps,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope=rope_compress if cr >= 1 else rope,
            owns_compressor=layer_id in kv_source_layers,
            owns_indexer=layer_id in index_source_layers,
            source_key=layer_id in source_key_indexer_layers,
            external_key=layer_id in external_key_indexer_layers,
            is_candidate_source=layer_id == candidate_source_layer,
            # Only an index source selects, so only it can consult the pool.
            uses_candidates=(
                layer_id in index_source_layers
                and layer_id != candidate_source_layer
                and 0 <= candidate_source_layer < layer_id
            ),
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=CANDIDATE_BLOCK_SIZE,
            layer_id=layer_id,
            index_source_layers=index_source_layers,
        )
        moe_cfg = _make_v41_moe_config(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            route_scale=route_scale,
            route_norm=route_norm,
            load_balance_coeff=load_balance_coeff,
            moe_comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            vision_enabled=vision,
        )
        layers.append(
            DeepSeekV41TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                moe=moe_cfg,
                hc_attn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=dim,
                    sinkhorn_iters=sinkhorn_iters,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=dim,
                    sinkhorn_iters=sinkhorn_iters,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_post=HcPost.Config(),
            )
        )

    common = dict(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
        kv_source_layers=kv_source_layers,
        index_source_layers=index_source_layers,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
    )
    if not vision:
        return DeepSeekV41Model.Config(**common)

    # The tower and the markers are what makes the multimodal stack a distinct class,
    # so they are imported here rather than at module scope: a text-only run never pays
    # for the vision module's dependencies.
    from .vision import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings

    return DeepSeekV41MultimodalModel.Config(
        **common,
        vision_encoder=DeepSeekV41VisionEncoder.Config(
            dim=vision_dim,
            num_layers=vision_layers,
            num_heads=vision_heads,
            inter_dim=vision_inter_dim,
            patch_size=patch_size,
            text_dim=dim,
            downsample_ratio=downsample_ratio,
        ),
        image_marker_embeddings=ImageMarkerEmbeddings.Config(
            dim=dim,
            param_init=_MARKER_INIT,
        ),
    )


def _debugmodel(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    *,
    num_experts: int = 16,
    vision: bool = True,
) -> DeepSeekV41Model.Config:
    config = _make_v41_config(
        dim=512,
        n_heads=8,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=128,
        o_lora_rank=128,
        n_groups=4,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=32,
        moe_inter_dim=256,
        candidate_topk_blocks=4,
        vision_dim=128,
        vision_heads=8,
        vision_inter_dim=256,
        max_seq_len=512,
        n_layers=40,
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        top_k=6,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        vision_layers=32,
        patch_size=14,
        downsample_ratio=3,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
        vision=vision,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    return _attach_engram(config, _ENGRAM_DEBUG_GEOMETRY)


def _deepseek_v4_1_flash(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    *,
    num_experts: int = 384,
    vision: bool = True,
    max_seq_len: int = 1_048_576,
) -> DeepSeekV41Model.Config:
    config = _make_v41_config(
        dim=5120,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        n_groups=8,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        moe_inter_dim=2304,
        candidate_topk_blocks=2048,
        vision_dim=1024,
        vision_heads=16,
        vision_inter_dim=2816,
        max_seq_len=max_seq_len,
        n_layers=40,
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        top_k=6,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        vision_layers=32,
        patch_size=14,
        downsample_ratio=3,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
        vision=vision,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    return _attach_engram(config, _flash_engram_geometry())


def _flash_engram_geometry():
    """The published V4.1-Flash Engram geometry, with the host-memory debug knob applied.

    Debug knob: shrink host-side engram tables for single-node runs.  SparseAdam
    materializes dense exp_avg/exp_avg_sq (zeros_like) on the first step, so production
    16M-row tables need ~2.4TB host RAM across 8 ranks.  TORCHTITAN_ENGRAM_TABLE_ROWS
    scales only the per-order row count; width/heads/token-map semantics are unchanged.
    Unset keeps the published V4.1-Flash geometry.

    Only the published geometry scales: the debug geometry is already small, and it is
    the one with ``require_token_id_map=False``.  Applying the row knob there would
    replace it with a compressed-vocabulary geometry no test fixture can build.  Both
    40-layer flavors read this, so a text/multimodal comparison runs on the same tables.
    """
    table_rows = os.environ.get("TORCHTITAN_ENGRAM_TABLE_ROWS")
    if table_rows is None:
        return _ENGRAM_V41_FLASH_GEOMETRY
    return dataclasses.replace(
        _ENGRAM_V41_FLASH_GEOMETRY,
        vocab_size_per_ngram=(int(table_rows),) * len(_ENGRAM_V41_FLASH_GEOMETRY.vocab_size_per_ngram),
    )


def _stripped(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    *,
    num_experts: int = 4,
) -> DeepSeekV41Model.Config:
    """A single-card DeepSeek-V4.1 that differs from the flash flavor in layers alone.

    Only the layer count and the layer-role tables are trimmed.  ``dim``, the attention
    heads and both released geometries are the flash ones untouched -- the MLA attention
    because SMLA's metadata operator requires ``head_dim`` to be exactly 512, and the
    indexer because the ``ds41`` kernels fix all four of its numbers.  The expert count is
    the only other reduction, so that six layers fit on one card.

    The topology is the flash one in miniature: ``candidate_source_layer`` sits at the first
    full-resolution layer, so the pool source is a Full Mode layer and the Reindex Mode
    layer after it searches what that source built.  Together with the ratio-2 Full Mode
    layer outside the hierarchy, all three branches of the fused selector are reachable.
    """
    return _make_v41_config(
        dim=5120,
        # The released MLA attention geometry; SMLA rejects anything but head_dim 512.
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        n_groups=8,
        # The released indexer geometry, which the ds41 QLI/QSLI pair fixes in full.
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_topk_blocks=2048,
        moe_inter_dim=2304,
        max_seq_len=4096,
        n_layers=len(V41_STRIPPED_COMPRESS_RATIOS),
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        # The router's top-k must fit the expert count, and the count is what the 32 GiB
        # container cap constrains; both are small for the same reason.
        top_k=2,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        vision_layers=32,
        patch_size=14,
        downsample_ratio=3,
        compress_ratios=V41_STRIPPED_COMPRESS_RATIOS,
        kv_source_layers=V41_STRIPPED_KV_SOURCE_LAYERS,
        index_source_layers=V41_STRIPPED_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_STRIPPED_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
        vision=False,
        vision_dim=128,
        vision_heads=8,
        vision_inter_dim=256,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )


def _attach_engram(config, engram):
    from .engram.config import _make_engram_configs

    if any(layer_id >= len(config.layers) or layer_id < 0 for layer_id in engram.layer_ids):
        raise ValueError("Engram layer IDs must lie inside the decoder")
    configs = _make_engram_configs(
        hidden_size=config.dim,
        hc_mult=config.hc_mult,
        vocab_size=config.vocab_size,
        engram=engram,
    )
    for layer_id, engram_config in configs.items():
        config.layers[layer_id].engram = engram_config
    return config


deepseek_v4_1_configs = {
    "deepseek_v4_1_flash": _deepseek_v4_1_flash,
    "deepseek_v4_1_flash_text": partial(_deepseek_v4_1_flash, vision=False),
    "deepseek_v4_1_flash_40layers_16experts_vision": partial(
        _deepseek_v4_1_flash,
        num_experts=16,
    ),
    "deepseek_v4_1_flash_40layers_16experts_text": partial(
        _deepseek_v4_1_flash,
        num_experts=16,
        vision=False,
    ),
    "deepseek_v4_1_debugmodel": _debugmodel,
    "deepseek_v4_1_debugmodel_text": partial(_debugmodel, vision=False),
    "deepseek_v4_1_stripped": _stripped,
}


def model_registry(
    flavor: str,
    *,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    from .parallelize import parallelize_deepseek_v4_1

    if flavor not in deepseek_v4_1_configs:
        raise ValueError(f"Unknown deepseek_v4_1 flavor: {flavor}. Available: {list(deepseek_v4_1_configs.keys())}")
    config = deepseek_v4_1_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    state_dict_adapter = DeepSeekV41StateDictAdapter
    if getattr(config, "lora", None) is not None:
        from .lora import DeepSeekV41LoRAStateDictAdapter

        state_dict_adapter = DeepSeekV41LoRAStateDictAdapter
    return ModelSpec(
        name="deepseek_v4_1",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v4_1,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=_register_step_pre_hooks,
        state_dict_adapter=state_dict_adapter,
    )


def _register_step_pre_hooks(optimizers, model_parts, parallel_dims) -> None:
    """Register the MoE balancing step hook."""
    from torchtitan.components.optimizer import register_moe_load_balancing_hook

    if any(getattr(model, "lora_config", None) is not None for model in model_parts):
        return
    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "V41_CANDIDATE_SOURCE_LAYER",
    "V41_FULL_COMPRESS_RATIOS",
    "V41_FULL_INDEX_SOURCE_LAYERS",
    "V41_KV_SOURCE_LAYERS",
    "Attention",
    "DeepSeekV41Model",
    "DeepSeekV41MultimodalModel",
    "DeepSeekV41StateDictAdapter",
    "deepseek_v4_1_configs",
    "model_registry",
]
