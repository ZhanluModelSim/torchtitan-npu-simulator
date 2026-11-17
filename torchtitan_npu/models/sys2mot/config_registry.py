# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native training configurations for the sys2mot VLA.

The reference geometry (``external_models/vla_physical_ai/VLA/sys2_mot``) spans
tiny (0.7M, numerically validated) through student30b/student300b. The debug
flavor is the reference's own ``tiny`` configuration verbatim (its 699,968
parameters are the golden calibration baseline): every structural path
survives — the ViT with 2D RoPE, per-layer CSA2 compress ratios ``[0, 2, 2, 2,
1, 1, 1, 0]`` with Full-Mode KV sources at layers 1/4 and indexers at layers
1/2/4/5, sqrt(softplus) routed MoE (4 experts top-2 + shared), mHC Sinkhorn
hyper-connections (n=4, 20 iterations), Engram on layers 1/3, and the
layer-paired action experts (8 pairs, width 32, joint softmax over prefix +
state/action KV).

The runner's synthetic text length (``training.seq_len``) must stay inside the
reference input contract (100..200 tokens); the debug preset pins 100.
"""

import math
from dataclasses import dataclass

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv2d
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .loss import Sys2MotJointLoss
from .model import (
    ActionExpert,
    ActionMoT,
    Block,
    Compressor,
    DeepSeekViT,
    Engram,
    GroupedExperts,
    HyperMix,
    Indexer,
    MoE,
    RMSNorm,
    RopeGeometry,
    RoutedExperts,
    Router,
    SparseAttention,
    SwiGLU,
    Sys2MotModel,
    VisionBlock,
)
from .parallelize import parallelize_sys2mot

_LIN = Linear.Config


@dataclass(frozen=True)
class _Geometry:
    """Debug geometry: the reference ``tiny`` config verbatim.

    The published student flavors reach student30b/student300b (40 layers,
    width 5120, 224 experts top-6, 84→546px inputs); the debug flavor keeps
    every structural path with the tiny capture-sized widths.
    """

    vocab_size: int = 512
    width: int = 64
    layers: int = 8
    encoder_layers: int = 4
    heads: int = 4
    head_dim: int = 16
    rope_dim: int = 4
    q_rank: int = 16
    o_rank: int = 16
    o_groups: int = 2
    experts: int = 4
    topk: int = 2
    expert_dim: int = 32
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    window: int = 16
    compress_ratios: tuple[int, ...] = (0, 2, 2, 2, 1, 1, 1, 0)
    kv_sources: tuple[int, ...] = (1, 4)
    index_sources: tuple[int, ...] = (1, 2, 4, 5)
    index_heads: int = 2
    index_dim: int = 8
    index_topk: int = 8
    candidate_source: int = 4
    candidate_blocks: int = 2
    candidate_block_size: int = 4
    hc_mult: int = 4
    sinkhorn_iters: int = 20
    eps: float = 1e-6
    engram_layers: tuple[int, ...] = (1, 3)
    engram_buckets: int = 127
    engram_heads: int = 2
    engram_dim: int = 8
    engram_max_ngram: int = 4
    vision_width: int = 32
    vision_layers: int = 2
    vision_heads: int = 4
    vision_mlp: int = 64
    patch_size: int = 14
    downsample: int = 3
    image_size: int = 84
    action_width: int = 32
    action_layers: int = 8
    action_q_rank: int = 8
    action_o_rank: int = 8
    action_mlp: int = 64
    action_dim: int = 32
    horizon: int = 50
    max_text: int = 200
    max_views: int = 16
    query_chunk: int = 64

    @property
    def engram_columns(self) -> int:
        return self.engram_heads * (self.engram_max_ngram - 1)

    @property
    def image_tokens(self) -> int:
        grid = math.ceil((self.image_size // self.patch_size) / self.downsample)
        # start + each row's features and newline + end
        return grid * grid + grid + 2

    def validate(self) -> None:
        """Reference ``Config.validate`` verbatim."""
        if len(self.compress_ratios) != self.layers or not 0 < self.encoder_layers < self.layers:
            raise ValueError("Invalid CED layer schedule")
        if self.encoder_layers not in self.kv_sources:
            raise ValueError("First decoder layer must project its global KV from final encoder states")
        if self.heads % self.o_groups or self.rope_dim % 2 or self.rope_dim > min(self.head_dim, self.index_dim):
            raise ValueError("Invalid low-rank attention dimensions")
        if self.vision_width % self.vision_heads or (self.vision_width // self.vision_heads) % 4:
            raise ValueError("2D-RoPE vision head dimension must be divisible by four")
        if self.action_layers != self.layers or self.action_width % 2 or self.topk > self.experts:
            raise ValueError("MoT requires one action expert per VLM layer and valid MoE dimensions")
        if self.image_size % self.patch_size or self.horizon != 50:
            raise ValueError("Patch-divisible images and 50-action horizon required")
        for i in range(self.layers):
            if i in self.kv_sources and i not in self.index_sources:
                raise ValueError("KV Full Mode requires an index-key owner")
            if i in self.index_sources and self.compress_ratios[i] == 0:
                raise ValueError("Index source must use global attention")


@dataclass(frozen=True)
class _GeometryMid(_Geometry):
    """Mid geometry: the reference ``student30b`` flavor with only the layer
    count reduced (40 -> 8) and the routed experts reduced (24 -> 16) to fit
    7-device fp32 training (6.3B parameters, ~14 GB/device including
    optimizer states). The budget is dominated by activation working space,
    not parameters: the indexer's candidate KV workspace alone is a single
    4.3 GB aclnnChunkCat allocation per indexer pass (candidate_blocks 2048 x
    block_size 8 x head_dim 512 x 64 heads x 4 B), and the mHC Sinkhorn
    hyper-connections keep per-layer intermediate states that at 12 layers
    still pushed reserved memory to 59.6 GB/61 GB before OOM in backward.

    Every other field keeps the published production value: the student30b
    expert/memory flavor (top-6 routing, expert_dim 1408, 32,771 engram
    buckets), the student300b dimensions (width 5120, heads 64, head_dim 512,
    low-rank 1280/1024 attention, index_topk 512, 2048 candidate blocks,
    546px vision at width 1024, 129,280 vocab). The structural schedules are
    remapped onto 8 layers with the student300b pattern: two dense layers, a
    ratio-2 compression spine, a ratio-1 tail, 2 KV sources, 4 indexers, and
    Engram on the first layer and the final encoder boundary.
    """

    vocab_size: int = 129280
    width: int = 5120
    layers: int = 8
    encoder_layers: int = 4
    heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_rank: int = 1280
    o_rank: int = 1024
    o_groups: int = 8
    experts: int = 16
    topk: int = 6
    expert_dim: int = 1408
    window: int = 128
    compress_ratios: tuple[int, ...] = (0, 0) + (2,) * 2 + (1,) * 4
    kv_sources: tuple[int, ...] = (2, 4)
    index_sources: tuple[int, ...] = (2, 4, 6, 8)
    index_heads: int = 32
    index_dim: int = 128
    index_topk: int = 512
    candidate_source: int = 4
    candidate_blocks: int = 2048
    candidate_block_size: int = 8
    engram_layers: tuple[int, ...] = (1, 4)
    engram_buckets: int = 32771
    engram_heads: int = 8
    engram_dim: int = 256
    vision_width: int = 1024
    vision_layers: int = 32
    vision_heads: int = 16
    vision_mlp: int = 2816
    image_size: int = 546
    action_width: int = 1024
    action_layers: int = 8
    action_q_rank: int = 256
    action_o_rank: int = 128
    action_mlp: int = 4096


def _norm(dim: int, g: _Geometry) -> RMSNorm.Config:
    return RMSNorm.Config(dim=dim, eps=g.eps)


def _norm_default(dim: int) -> RMSNorm.Config:
    # Reference default eps: the vision stems and block norms construct
    # ``RMSNorm(dim)`` without the model eps (1e-6 for both).
    return RMSNorm.Config(dim=dim, eps=1e-6)


def _swiglu(width: int, inner: int, limit: float) -> SwiGLU.Config:
    return SwiGLU.Config(
        gate=_LIN(in_features=width, out_features=inner, bias=False),
        up=_LIN(in_features=width, out_features=inner, bias=False),
        down=_LIN(in_features=inner, out_features=width, bias=False),
        limit=limit,
    )


def _vision_block_config(*, g: _Geometry) -> VisionBlock.Config:
    d = g.vision_width
    return VisionBlock.Config(
        vision_width=d,
        vision_heads=g.vision_heads,
        vision_mlp=g.vision_mlp,
        n1=_norm_default(d),
        n2=_norm_default(d),
        qkv=_LIN(in_features=d, out_features=3 * d, bias=True),
        out=_LIN(in_features=d, out_features=d, bias=True),
        ffn=_swiglu(d, g.vision_mlp, limit=0.0),
    )


def _vit_config(*, g: _Geometry) -> DeepSeekViT.Config:
    d = g.vision_width
    return DeepSeekViT.Config(
        vision_width=d,
        vision_layers=g.vision_layers,
        vision_heads=g.vision_heads,
        vision_mlp=g.vision_mlp,
        patch_size=g.patch_size,
        downsample=g.downsample,
        image_size=g.image_size,
        width=g.width,
        patch=Conv2d.Config(
            in_channels=3,
            out_channels=d,
            kernel_size=g.patch_size,
            stride=g.patch_size,
            bias=True,
        ),
        blocks=[_vision_block_config(g=g) for _ in range(g.vision_layers)],
        norm=_norm_default(d),
        align1=_LIN(in_features=d * g.downsample**2, out_features=g.width, bias=True),
        align2=_LIN(in_features=g.width, out_features=g.width, bias=True),
    )


def _compressor_config(*, layer_idx: int, g: _Geometry) -> Compressor.Config | None:
    if layer_idx not in g.kv_sources:
        return None
    ratio = g.compress_ratios[layer_idx]
    return Compressor.Config(
        ratio=ratio,
        width=g.width,
        head_dim=g.head_dim,
        eps=g.eps,
        kv=_LIN(in_features=g.width, out_features=g.head_dim, bias=False),
        gate=_LIN(in_features=g.width, out_features=g.head_dim, bias=False) if ratio > 1 else None,
        norm=_norm(g.head_dim, g),
    )


def _indexer_config(*, layer_idx: int, g: _Geometry) -> Indexer.Config | None:
    if layer_idx not in g.index_sources:
        return None
    owns_k = layer_idx in g.kv_sources
    return Indexer.Config(
        owns_k=owns_k,
        index_heads=g.index_heads,
        index_dim=g.index_dim,
        index_topk=g.index_topk,
        rope_dim=g.rope_dim,
        head_dim=g.head_dim,
        query_chunk=g.query_chunk,
        candidate_source=g.candidate_source,
        candidate_blocks=g.candidate_blocks,
        candidate_block_size=g.candidate_block_size,
        q_rank=g.q_rank,
        width=g.width,
        eps=g.eps,
        q=_LIN(in_features=g.q_rank, out_features=g.index_heads * g.index_dim, bias=False),
        weights=_LIN(in_features=g.width, out_features=g.index_heads, bias=False),
        key=_LIN(in_features=g.head_dim, out_features=g.index_dim, bias=False) if owns_k else None,
        norm=_norm(g.index_dim, g) if owns_k else None,
    )


def _attention_config(*, layer_idx: int, g: _Geometry) -> SparseAttention.Config:
    """CSA2 attention config for one VLM layer.

    The ``rope`` carrier only feeds the framework's sequence-length contract;
    the reference recomputes RoPE freqs inline per call.
    """
    return SparseAttention.Config(
        layer=layer_idx,
        ratio=g.compress_ratios[layer_idx],
        width=g.width,
        q_rank=g.q_rank,
        heads=g.heads,
        head_dim=g.head_dim,
        rope_dim=g.rope_dim,
        o_groups=g.o_groups,
        o_rank=g.o_rank,
        window=g.window,
        eps=g.eps,
        query_chunk=g.query_chunk,
        qa=_LIN(in_features=g.width, out_features=g.q_rank, bias=False),
        qnorm=_norm(g.q_rank, g),
        qb=_LIN(in_features=g.q_rank, out_features=g.heads * g.head_dim, bias=False),
        kv=_LIN(in_features=g.width, out_features=g.head_dim, bias=False),
        kvnorm=_norm(g.head_dim, g),
        ob=_LIN(in_features=g.o_groups * g.o_rank, out_features=g.width, bias=False),
        compressor=_compressor_config(layer_idx=layer_idx, g=g),
        indexer=_indexer_config(layer_idx=layer_idx, g=g),
        # Nominal carrier value (max_text); update_from_config resizes it to
        # training.seq_len, and the forward never reads it.
        rope=RopeGeometry(max_seq_len=g.max_text),
    )


def _moe_config(*, g: _Geometry) -> MoE.Config:
    return MoE.Config(
        router=Router.Config(
            width=g.width,
            num_experts=g.experts,
            topk=g.topk,
            route_scale=g.route_scale,
        ),
        routed_experts=RoutedExperts.Config(
            inner_experts=GroupedExperts.Config(
                width=g.width,
                expert_dim=g.expert_dim,
                num_experts=g.experts,
                swiglu_limit=g.swiglu_limit,
            ),
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=g.experts,
                top_k=g.topk,
            ),
        ),
        shared=_swiglu(g.width, g.expert_dim, g.swiglu_limit),
    )


def _engram_config(*, g: _Geometry) -> Engram.Config:
    columns = g.engram_columns
    return Engram.Config(
        buckets=g.engram_buckets,
        heads=g.engram_heads,
        dim=g.engram_dim,
        max_ngram=g.engram_max_ngram,
        width=g.width,
        hc_mult=g.hc_mult,
        eps=g.eps,
        table=Embedding.Config(num_embeddings=columns * g.engram_buckets, embedding_dim=g.engram_dim),
        project=_LIN(in_features=columns * g.engram_dim, out_features=(g.hc_mult + 1) * g.width, bias=False),
    )


def _hypermix_config(*, g: _Geometry) -> HyperMix.Config:
    n = g.hc_mult
    return HyperMix.Config(
        hc_mult=n,
        width=g.width,
        eps=g.eps,
        sinkhorn_iters=g.sinkhorn_iters,
        fn=_LIN(in_features=n * g.width, out_features=(n + 2) * n, bias=False),
    )


def _block_config(*, layer_idx: int, g: _Geometry) -> Block.Config:
    return Block.Config(
        layer=layer_idx,
        attention=_attention_config(layer_idx=layer_idx, g=g),
        moe=_moe_config(g=g),
        attn_mix=_hypermix_config(g=g),
        ffn_mix=_hypermix_config(g=g),
        n1=_norm(g.width, g),
        n2=_norm(g.width, g),
        engram=_engram_config(g=g) if layer_idx in g.engram_layers else None,
    )


def _action_expert_config(*, layer_idx: int, g: _Geometry) -> ActionExpert.Config:
    d = g.action_width
    return ActionExpert.Config(
        layer=layer_idx,
        # Shared latent K=V geometry: theta follows the VLM layer's ratio.
        theta=160000.0 if g.compress_ratios[layer_idx] else 10000.0,
        action_width=d,
        action_q_rank=g.action_q_rank,
        action_o_rank=g.action_o_rank,
        action_mlp=g.action_mlp,
        heads=g.heads,
        head_dim=g.head_dim,
        rope_dim=g.rope_dim,
        o_groups=g.o_groups,
        eps=g.eps,
        n1=_norm(d, g),
        n2=_norm(d, g),
        qa=_LIN(in_features=d, out_features=g.action_q_rank, bias=False),
        qnorm=_norm(g.action_q_rank, g),
        qb=_LIN(in_features=g.action_q_rank, out_features=g.heads * g.head_dim, bias=False),
        kv=_LIN(in_features=d, out_features=g.head_dim, bias=False),
        kvnorm=_norm(g.head_dim, g),
        ob=_LIN(in_features=g.o_groups * g.action_o_rank, out_features=d, bias=False),
        ffn=_swiglu(d, g.action_mlp, limit=0.0),
        modulation=_LIN(in_features=d, out_features=6 * d, bias=True),
    )


def _action_config(*, g: _Geometry) -> ActionMoT.Config:
    d = g.action_width
    return ActionMoT.Config(
        action_dim=g.action_dim,
        horizon=g.horizon,
        blocks=[_action_expert_config(layer_idx=i, g=g) for i in range(g.action_layers)],
        state_in=_LIN(in_features=g.action_dim, out_features=d, bias=True),
        action_in=_LIN(in_features=g.action_dim, out_features=d, bias=True),
        time_in=_LIN(in_features=d, out_features=d, bias=True),
        time_out=_LIN(in_features=d, out_features=d, bias=True),
        norm=_norm(d, g),
        out=_LIN(in_features=d, out_features=g.action_dim, bias=True),
    )


def _make_model_config(*, g: _Geometry) -> Sys2MotModel.Config:
    g.validate()
    return Sys2MotModel.Config(
        dim=g.width,
        vocab_size=g.vocab_size,
        tok_embeddings=Embedding.Config(num_embeddings=g.vocab_size, embedding_dim=g.width),
        norm=_norm(g.width, g),
        lm_head=_LIN(in_features=g.width, out_features=g.vocab_size, bias=False),
        layers=[_block_config(layer_idx=i, g=g) for i in range(g.layers)],
        vision=_vit_config(g=g),
        action=_action_config(g=g),
        encoder_layers=g.encoder_layers,
        n_index_sources=len(g.index_sources),
        hc_mult=g.hc_mult,
        image_tokens=g.image_tokens,
        image_size=g.image_size,
        max_views=g.max_views,
        max_text=g.max_text,
    )


sys2mot_configs = {
    "debugmodel_multimodal": lambda: _make_model_config(g=_Geometry()),
    "mid_multimodal": lambda: _make_model_config(g=_GeometryMid()),
}


def model_registry(flavor: str) -> ModelSpec:
    if flavor not in sys2mot_configs:
        raise ValueError(f"Unknown sys2mot flavor: {flavor}. Available: {list(sys2mot_configs)}")
    return ModelSpec(
        name="sys2mot",
        flavor=flavor,
        model=sys2mot_configs[flavor](),
        parallelize_fn=parallelize_sys2mot,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _sys2mot_trainer_config(
    flavor: str, *, num_mtp_layers: int = 0, seq_len: int = 100
) -> Trainer.Config:
    if num_mtp_layers:
        raise ValueError("sys2mot has no MTP modules; num_mtp_layers must be 0")
    model_spec = model_registry(flavor)
    model_config = model_spec.model
    assert isinstance(model_config, Sys2MotModel.Config)
    # The captured joint loss: flow MSE + 0.5*logits-KD + 0.1*hidden-MSE +
    # 0.1*text CE + 0.01*indexer aux. KD/hidden distill from a frozen teacher
    # (same capture config, mounted on the loss by the runner) so the full
    # reference objective runs inside the capture. The text span starts after
    # the synthesized 3-view vision prefix.
    loss = Sys2MotJointLoss.Config(
        text_offset=3 * model_config.image_tokens,
        action_dim=model_config.action.action_dim,
        horizon=model_config.action.horizon,
        ce_weight=0.1,
        indexer_weight=0.01,
        distill=True,
    )
    return Trainer.Config(
        loss=loss,
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_spec,
        # Reference joint training: AdamW(lr=3e-4, weight_decay=0.01).
        optimizer=default_adamw(lr=3e-4, weight_decay=0.01),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="linear",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=1,
            # Reference input contract: 100..200 text tokens; the debug preset
            # pins the minimum (embed_prefix validates the range), the mid
            # preset pins the production maximum.
            seq_len=seq_len,
            steps=10,
            # Reference ``train_dummy`` is fp32 (like the system3 capture
            # flavor); the routed experts cast to bf16 for the grouped GEMM.
            dtype="float32",
            mixed_precision_param="float32",
            mixed_precision_reduce="float32",
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


def sys2mot_debugmodel_multimodal(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _sys2mot_trainer_config("debugmodel_multimodal", num_mtp_layers=num_mtp_layers)


def sys2mot_mid_multimodal(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _sys2mot_trainer_config(
        "mid_multimodal", num_mtp_layers=num_mtp_layers, seq_len=200
    )


__all__ = ["model_registry", "sys2mot_debugmodel_multimodal", "sys2mot_mid_multimodal"]
