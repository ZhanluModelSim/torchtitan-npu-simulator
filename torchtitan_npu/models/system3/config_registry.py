# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native training configurations for the System3 Flash decoder.

The reference geometry (``external_models/system3``) is a 40T
compute-equivalent preview: 100 physical layers looped twice, 16384 hidden,
4096 routed experts top-25 with a 4096-wide latent bottleneck, 8M context.
The debug flavor is the reference's own ``smoke_test`` configuration
verbatim (its 14,148,524-parameter geometry is the golden calibration
baseline): every structural path survives — the ``[ISA, GDN2 x 3]`` unit
pattern, loop 2x with Flash KV sharing (kv/reuse/reindex), LatentMoE with
the latent bottleneck and shared expert, mHC Sinkhorn hyper-connections,
Engram conditional memory on inject layers, and (multimodal flavor only)
the four modality encoders with interleaved AR sequence assembly.
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
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, Conv2d
from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.trainer import Trainer

from .model import (
    AlternatingAttention,
    AudioEncoder,
    EncoderAttention,
    EncoderBlock,
    EngramModule,
    FlashBlock,
    GDN2Attention,
    HapticEncoder,
    HyperConnectionBlock,
    ImageEncoder,
    IndexedSparseAttention,
    ISAIndexer,
    LatentGroupedExperts,
    LatentMoELayer,
    LatentMoERouter,
    LatentRoutedExperts,
    ModalityEncoder,
    MultiHeadHashTable,
    PointCloudEncoder,
    PolynomialRollingHash,
    RopeGeometry,
    RotaryEmbedding,
    SharedLatentExpertMLP,
    SinkhornIteration,
    System3Model,
    System3RMSNorm,
)
from .parallelize import parallelize_system3

_LIN = Linear.Config


@dataclass(frozen=True)
class _Geometry:
    """Debug geometry: the reference ``smoke_test`` config verbatim.

    The published 40T preview is hidden=16384 / 100 physical layers x 2
    loops / 4096 experts top-25; the smoke flavor keeps every structural
    path with capture-sized widths (hc_branch_dim stays at the reference
    default 1024 — the smoke's 14.1M parameter count depends on it).
    """

    hidden_size: int = 128
    vocab_size: int = 512
    base_depth: int = 12
    num_recursion: int = 2
    heads: int = 4
    head_dim: int = 32
    qk_rope_head_dim: int = 8
    rope_theta: float = 1024.0
    yarn_factor: float = 4.0
    yarn_original_max: int = 256
    max_seq_len: int = 512
    norm_eps: float = 1e-6
    # Flash unit + KV sharing.
    unit_size: int = 4
    kv_groups: int = 1
    kv_reindex_per_group: int = 1
    # ISA.
    isa_compress_ratio: int = 2
    indexer_heads: int = 2
    indexer_head_dim: int = 16
    indexer_topk: int = 8
    attn_softmax_clamp: float = 50.0
    # GDN2.
    gdn2_heads: int = 4
    gdn2_d_state: int = 16
    gdn2_d_k: int = 8
    gdn2_d_v: int = 8
    gdn2_d_a: int = 3
    # mHC.
    hc_mult: int = 2
    hc_branch_dim: int = 1024
    sinkhorn_iters: int = 3
    # LatentMoE.
    num_experts: int = 4
    topk: int = 2
    moe_latent: int = 48
    moe_inter: int = 64
    route_scale: float = 2.5
    swiglu_clamp: float = 10.0
    z_loss_alpha: float = 0.001
    # Engram.
    engram_layers: tuple[int, ...] = (1, 5, 9)
    engram_orders: tuple[int, ...] = (2,)
    engram_heads: int = 2
    engram_head_dim: int = 16
    engram_rows: int = 64
    # Multimodal encoders (smoke geometry).
    image_hidden: int = 64
    image_layers: int = 2
    image_heads: int = 4
    image_patch: int = 8
    image_max_patches: int = 64
    enc_hidden: int = 48
    enc_layers: int = 2
    enc_heads: int = 4
    conv_kernel: int = 4
    conv_stride: int = 2
    max_frames: int = 64
    point3d_points: int = 16


def _norm(dim: int, g: _Geometry) -> System3RMSNorm.Config:
    return System3RMSNorm.Config(dim=dim, eps=g.norm_eps)


_FULL_GEOMETRY = _Geometry(
    # Published-preview-shaped single-card configuration: every structural
    # path (ISA/GDN2 units, latent MoE, Engram hash, mHC) at realistic
    # widths; MoE sized so bf16 params + fp32 AdamW states stay well within
    # one 910B3 (64 GB).
    hidden_size=2048,
    vocab_size=16384,
    base_depth=24,
    num_recursion=2,
    heads=16,
    head_dim=128,
    qk_rope_head_dim=64,
    rope_theta=10000.0,
    yarn_factor=1.0,
    yarn_original_max=4096,
    max_seq_len=4096,
    indexer_heads=4,
    indexer_head_dim=64,
    indexer_topk=64,
    gdn2_heads=16,
    gdn2_d_state=128,
    gdn2_d_k=64,
    gdn2_d_v=64,
    gdn2_d_a=32,
    hc_mult=2,
    hc_branch_dim=1024,
    num_experts=16,
    topk=4,
    moe_latent=512,
    moe_inter=512,
    engram_layers=(1, 7, 13, 19),
    engram_heads=4,
    engram_head_dim=64,
    engram_rows=4096,
)


def _attention_config(*, layer_idx: int, g: _Geometry):
    """ISA or GDN2 attention config for one physical layer.

    ISA layers sit at unit positions 0; their KV role follows the reference
    formula over the per-slot ISA group (p==0 -> 'kv', p in
    [2, 2+reindex) -> 'reindex', else 'reuse').
    """
    units = g.base_depth // g.unit_size
    isa_group_size = units // g.kv_groups
    unit = layer_idx // g.unit_size
    is_isa = layer_idx % g.unit_size == 0
    if not is_isa:
        d = g.hidden_size
        H, dk, dv, ds, da = g.gdn2_heads, g.gdn2_d_k, g.gdn2_d_v, g.gdn2_d_state, g.gdn2_d_a
        gdn2 = GDN2Attention.Config(
            hidden_size=d,
            num_heads=H,
            d_state=ds,
            d_k=dk,
            d_v=dv,
            d_a=da,
            rope=RopeGeometry(max_seq_len=g.max_seq_len),
            q_proj=_LIN(in_features=d, out_features=H * dk, bias=False),
            k_proj=_LIN(in_features=d, out_features=H * dk, bias=False),
            v_proj=_LIN(in_features=d, out_features=H * dv, bias=False),
            o_proj=_LIN(in_features=H * dv, out_features=d, bias=False),
            alpha_gate=_LIN(in_features=d, out_features=H * dk, bias=False),
            beta_gate=_LIN(in_features=d, out_features=2 * H * ds, bias=False),
            da_proj=_LIN(in_features=d, out_features=H * da, bias=False),
            da_norm=_norm(H * da, g),
        )
        return False, gdn2

    kv_group = unit // isa_group_size
    p = unit % isa_group_size
    role = "kv" if p == 0 else ("reindex" if 2 <= p < 2 + g.kv_reindex_per_group else "reuse")
    d = g.hidden_size
    hd = g.head_dim
    idx_dim = g.indexer_heads * g.indexer_head_dim
    attn = AlternatingAttention.Config(
        layer_idx=layer_idx,
        is_isa=True,
        hidden_size=d,
        num_heads=g.heads,
        head_dim=hd,
        qk_rope_head_dim=g.qk_rope_head_dim,
        compress_ratio=g.isa_compress_ratio,
        softmax_clamp=g.attn_softmax_clamp,
        rope=RopeGeometry(max_seq_len=g.max_seq_len),
        role=role,
        kv_group=kv_group,
        q_proj=_LIN(in_features=d, out_features=g.heads * hd, bias=False),
        o_proj=_LIN(in_features=g.heads * hd, out_features=d, bias=False),
        k_proj=_LIN(in_features=d, out_features=g.heads * hd, bias=False) if role == "kv" else None,
        v_proj=_LIN(in_features=d, out_features=g.heads * hd, bias=False) if role == "kv" else None,
        indexer=(
            ISAIndexer.Config(
                hidden_size=d,
                n_heads=g.indexer_heads,
                head_dim=g.indexer_head_dim,
                topk=g.indexer_topk,
                compress_ratio=g.isa_compress_ratio,
                q_proj=_LIN(in_features=d, out_features=idx_dim, bias=False),
                k_proj=_LIN(in_features=d, out_features=idx_dim, bias=False),
                q_norm=_norm(idx_dim, g),
                k_norm=_norm(idx_dim, g),
            )
            if role in ("kv", "reindex")
            else None
        ),
        isa=IndexedSparseAttention.Config(
            head_dim=hd,
            softmax_clamp=g.attn_softmax_clamp,
            compress_ratio=g.isa_compress_ratio,
        ),
        rotary=RotaryEmbedding.Config(
            dim=g.qk_rope_head_dim,
            theta=g.rope_theta,
            yarn_factor=g.yarn_factor,
            original_max_seq_len=g.yarn_original_max,
            max_seq_len=g.max_seq_len,
        ),
    )
    return True, attn


def _mhc_config(*, g: _Geometry) -> HyperConnectionBlock.Config:
    d = g.hidden_size
    branch = g.hc_mult * g.hc_branch_dim
    return HyperConnectionBlock.Config(
        hidden_size=d,
        hc_mult=g.hc_mult,
        branch_dim=g.hc_branch_dim,
        expand=_LIN(in_features=d, out_features=branch, bias=False),
        contract=_LIN(in_features=branch, out_features=d, bias=False),
        pre_norm=_norm(d, g),
        post_norm=_norm(d, g),
        sinkhorn=SinkhornIteration.Config(n_iters=g.sinkhorn_iters),
    )


def _moe_config(*, g: _Geometry) -> LatentMoELayer.Config:
    d = g.hidden_size
    latent, inter = g.moe_latent, g.moe_inter
    return LatentMoELayer.Config(
        hidden_size=d,
        z_loss_alpha=g.z_loss_alpha,
        gate_down=_LIN(in_features=d, out_features=latent, bias=False),
        up_down=_LIN(in_features=d, out_features=latent, bias=False),
        latent_up=_LIN(in_features=latent, out_features=d, bias=False),
        router=LatentMoERouter.Config(
            hidden_size=d,
            num_experts=g.num_experts,
            topk=g.topk,
            route_scale=g.route_scale,
            score_function="sqrtsoftplus",
        ),
        routed_experts=LatentRoutedExperts.Config(
            inner_experts=LatentGroupedExperts.Config(
                dim=latent,
                hidden_dim=inter,
                num_experts=g.num_experts,
                swiglu_limit=g.swiglu_clamp,
            ),
            token_dispatcher=LocalTokenDispatcher.Config(
                num_experts=g.num_experts,
                top_k=g.topk,
            ),
        ),
        shared_experts=[
            SharedLatentExpertMLP.Config(
                gate_inter=_LIN(in_features=latent, out_features=inter, bias=False),
                up_inter=_LIN(in_features=latent, out_features=inter, bias=False),
                inter_to_latent=_LIN(in_features=inter, out_features=latent, bias=False),
                clamp_val=g.swiglu_clamp,
            )
        ],
    )


def _engram_config(*, layer_idx: int, g: _Geometry) -> EngramModule.Config:
    d = g.hidden_size
    concat_dim = len(g.engram_orders) * g.engram_heads * g.engram_head_dim
    return EngramModule.Config(
        hidden_size=d,
        key_proj=_LIN(in_features=concat_dim, out_features=d, bias=False),
        value_proj=_LIN(in_features=concat_dim, out_features=d, bias=False),
        query_proj=_LIN(in_features=d, out_features=d, bias=False),
        q_norm=_norm(d, g),
        k_norm=_norm(d, g),
        input_norm=_norm(d, g),
        output_norm=_norm(d, g),
        hash_fns=[
            PolynomialRollingHash.Config(
                order=order,
                capacity=g.engram_rows,
                num_heads=g.engram_heads,
                layer_idx=layer_idx,
                seed_offset=order_idx,
            )
            for order_idx, order in enumerate(g.engram_orders)
        ],
        hash_tables=[
            MultiHeadHashTable.Config(
                capacity=g.engram_rows,
                num_heads=g.engram_heads,
                head_dim=g.engram_head_dim,
            )
            for _ in g.engram_orders
        ],
    )


def _layer_config(*, layer_idx: int, g: _Geometry) -> FlashBlock.Config:
    d = g.hidden_size
    _, attention = _attention_config(layer_idx=layer_idx, g=g)
    is_engram = layer_idx in g.engram_layers
    return FlashBlock.Config(
        layer_idx=layer_idx,
        layer_type="isa" if layer_idx % g.unit_size == 0 else "gdn2",
        hc_pre_attn=_mhc_config(g=g),
        hc_pre_ffn=_mhc_config(g=g),
        attention=attention,
        attn_pre_norm=_norm(d, g),
        attn_post_norm=_norm(d, g),
        moe_pre_norm=_norm(d, g),
        moe_post_norm=_norm(d, g),
        moe=None if is_engram else _moe_config(g=g),
        engram=_engram_config(layer_idx=layer_idx, g=g) if is_engram else None,
    )


def _encoder_block_config(*, hidden: int, heads: int, g: _Geometry):
    inter = hidden * 4
    return EncoderBlock.Config(
        norm1=_norm(hidden, g),
        attn=EncoderAttention.Config(
            hidden=hidden,
            heads=heads,
            qkv=_LIN(in_features=hidden, out_features=3 * hidden, bias=False),
            o=_LIN(in_features=hidden, out_features=hidden, bias=False),
        ),
        norm2=_norm(hidden, g),
        mlp_up=_LIN(in_features=hidden, out_features=inter, bias=False),
        mlp_down=_LIN(in_features=inter, out_features=hidden, bias=False),
    )


def _modality_encoder_config(*, g: _Geometry) -> ModalityEncoder.Config:
    # Order matters: the reference builds type_ids by enumerating this list
    # (text=0, image=1, audio=2, point3d=3, haptic=4), which indexes the
    # shared modality_type_embedding rows.
    modalities = ["text", "image", "audio", "point3d", "haptic"]
    return ModalityEncoder.Config(
        modalities=modalities,
        image_encoder=ImageEncoder.Config(
            max_patches=g.image_max_patches,
            patch_embed=Conv2d.Config(
                in_channels=3,
                out_channels=g.image_hidden,
                kernel_size=g.image_patch,
                stride=g.image_patch,
                bias=False,
            ),
            blocks=[
                _encoder_block_config(hidden=g.image_hidden, heads=g.image_heads, g=g) for _ in range(g.image_layers)
            ],
            norm=_norm(g.image_hidden, g),
        ),
        audio_encoder=AudioEncoder.Config(
            max_frames=g.max_frames,
            stem_conv1=Conv1d.Config(
                in_channels=1,
                out_channels=g.enc_hidden,
                kernel_size=g.conv_kernel,
                stride=g.conv_stride,
                bias=False,
            ),
            stem_conv2=Conv1d.Config(
                in_channels=g.enc_hidden,
                out_channels=g.enc_hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            blocks=[_encoder_block_config(hidden=g.enc_hidden, heads=g.enc_heads, g=g) for _ in range(g.enc_layers)],
            norm=_norm(g.enc_hidden, g),
        ),
        point3d_encoder=PointCloudEncoder.Config(
            mlp_in=_LIN(in_features=3, out_features=g.enc_hidden, bias=False),
            mlp_hidden=_LIN(in_features=g.enc_hidden, out_features=g.enc_hidden, bias=False),
            mlp_out=_LIN(in_features=g.enc_hidden, out_features=g.enc_hidden, bias=False),
            blocks=[_encoder_block_config(hidden=g.enc_hidden, heads=g.enc_heads, g=g) for _ in range(g.enc_layers)],
            norm=_norm(g.enc_hidden, g),
        ),
        haptic_encoder=HapticEncoder.Config(
            max_frames=g.max_frames,
            stem_conv=Conv1d.Config(
                in_channels=4,
                out_channels=g.enc_hidden,
                kernel_size=g.conv_kernel,
                stride=g.conv_stride,
                bias=False,
            ),
            blocks=[_encoder_block_config(hidden=g.enc_hidden, heads=g.enc_heads, g=g) for _ in range(g.enc_layers)],
            norm=_norm(g.enc_hidden, g),
        ),
        modality_projs={
            name: _LIN(in_features=hidden, out_features=g.hidden_size, bias=False)
            for name, hidden in (
                ("image", g.image_hidden),
                ("audio", g.enc_hidden),
                ("point3d", g.enc_hidden),
                ("haptic", g.enc_hidden),
            )
        },
        modality_type_embedding=Embedding.Config(
            num_embeddings=len(modalities),
            embedding_dim=g.hidden_size,
        ),
    )


def _make_model_config(*, multimodal: bool, g: _Geometry) -> System3Model.Config:
    return System3Model.Config(
        dim=g.hidden_size,
        vocab_size=g.vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=g.vocab_size,
            embedding_dim=g.hidden_size,
            # Reference init: trunc-normal(std=embedding_init_method_std=0.02).
            param_init={"weight": partial(nn.init.trunc_normal_, std=0.02)},
        ),
        norm=_norm(g.hidden_size, g),
        lm_head=_LIN(in_features=g.hidden_size, out_features=g.vocab_size, bias=False),
        # The reference ties the LM head to the token embedding; the capture
        # stack keeps two independent matrices (runner parameter-ledger
        # contract, same deviation class as the other simulator families).
        layers=[_layer_config(layer_idx=idx, g=g) for idx in range(g.base_depth)],
        num_recursion=g.num_recursion,
        kv_share_enable=True,
        kv_groups=g.kv_groups,
        unit_size=g.unit_size,
        multimodal_enable=multimodal,
        modality_encoder=_modality_encoder_config(g=g) if multimodal else None,
    )


system3_configs = {
    "debugmodel_text": lambda: _make_model_config(multimodal=False, g=_Geometry()),
    "debugmodel_multimodal": lambda: _make_model_config(multimodal=True, g=_Geometry()),
    "full_text": lambda: _make_model_config(multimodal=False, g=_FULL_GEOMETRY),
}


def model_registry(flavor: str) -> ModelSpec:
    if flavor not in system3_configs:
        raise ValueError(f"Unknown system3 flavor: {flavor}. Available: {list(system3_configs)}")
    return ModelSpec(
        name="system3",
        flavor=flavor,
        model=system3_configs[flavor](),
        parallelize_fn=parallelize_system3,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _system3_trainer_config(
    flavor: str,
    *,
    num_mtp_layers: int = 0,
    seq_len: int = 32,
    steps: int = 10,
) -> Trainer.Config:
    if num_mtp_layers:
        raise ValueError("system3 has no MTP modules; num_mtp_layers must be 0")
    model_spec = model_registry(flavor)
    # The multimodal AR sequence is the modality segments + the text span, so
    # its length is not a multiple of the chunked-loss chunk grid; run the
    # loss as a single chunk there (equivalent math, no divisibility bound).
    num_chunks = 1 if flavor == "debugmodel_multimodal" else 8
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            num_chunks=num_chunks,
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
            seq_len=seq_len,
            steps=steps,
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


def system3_debugmodel_text(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _system3_trainer_config("debugmodel_text", num_mtp_layers=num_mtp_layers)


def system3_debugmodel_multimodal(*, num_mtp_layers: int = 0) -> Trainer.Config:
    return _system3_trainer_config("debugmodel_multimodal", num_mtp_layers=num_mtp_layers)


def system3_full_text(*, num_mtp_layers: int = 0, seq_len: int = 2048, steps: int = 10) -> Trainer.Config:
    """Single-card full-size text flavor for duration-grade comparisons."""
    return _system3_trainer_config("full_text", num_mtp_layers=num_mtp_layers, seq_len=seq_len, steps=steps)
