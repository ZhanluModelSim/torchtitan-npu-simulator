# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek V4.1 backbones: the text stack and its multimodal extension.

Shape legend for this file:
    T = folded tokens, D = model dimension, hc = ``hc_mult`` residual
    branches.

:class:`DeepSeekV41Model` is the text-only backbone — a plain
:class:`torchtitan.models.common.decoder.Decoder` with no MTP depths and no vision
tower.  :class:`DeepSeekV41MultimodalModel` subclasses it and adds the vision tower, the
image markers and the modality-routing bias; the only things that differ are the input
embedding construction and whether an ``image_mask`` reaches the block, so the text
stack carries no vision parameter, no vision branch and no vision import.

The block carries the residual stream as ``hc`` parallel branches and threads the
cross-layer attention state (compressed KV, index keys, selected entries, student
logits, candidate pool) explicitly: an attention source layer returns the tensors it
produced, every other layer returns what it was handed.  That keeps the reuse structure
visible in the forward signature instead of in mutable module state, at the cost of a
longer tuple.

Single-Pass mHC means each sublayer collapses its input with the mixing coefficients
predicted by the *previous* sublayer, so the block returns the coefficients its
successor needs.  The stack has no learned output head: the model collapses the
branches with the last block's coefficients and feeds the result straight to the output
norm.

Packed documents are described by :class:`DeepSeekV41Metadata`: one frame per addressed
tensor, whose packed boundaries are the only varlen metadata there is.  Both consumers read
those frames and nothing else -- the torch-native core and selector address the row with
them, and the fused ports grid the same ones -- so a document never needs a second
description.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, cast

import torch
import torch_remat as remat
from torch import nn
from torchtitan.models.common.decoder import Decoder, TransformerBlock

from torchtitan_npu.models.common.attention_metadata import GeometryAttentionMetadataMixin
from torchtitan_npu.models.common.metadata import KernelFrame
from torchtitan_npu.models.common.restore import DOC_CHUNK_LAYOUT, doc_chunk_permutation, document_starts

from .engram import Engram  # noqa: TC001
from .indexer import IndexerMode
from .mhc import HcPost, HcPre

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE

    from torchtitan_npu.models.common.attention_metadata import AttentionMetadataMap
    from torchtitan_npu.models.common.restore import RestorePlan

    from .attention import Attention
    from .vision import DeepSeekV41VisionEncoder, ImageMarkerEmbeddings


@dataclass(frozen=True, kw_only=True, slots=True)
class DeepSeekV41Metadata:
    """The packed row's frames, built by the attention backends' config.

    Upstream's pair of cumulative boundaries -- ``cu_seq_q`` for the queries and ``cu_seq_k`` for
    the keys this entry addresses -- plus one :class:`KernelFrame` per addressed tensor (the
    queries, the sliding-window keys and, for a geometry that pools, the compressed keys), the key
    stream's own positions and, under context parallelism, its movement.  ``cu_seq_k`` is the
    **pool grid**: an entry belongs to one compression ratio (the metadata map key carries it), so
    the backend that builds the entry pools the row there and no consumer divides by its ratio at
    call time.  A layer that pools at all but addresses the row itself (ratio 0 or 1) gets
    ``cu_seq_k == cu_seq_q`` and still declares the two key frames.

    A vendor backend extends this entry with the records its own kernels launch (the AscendC
    entry adds ``aicpu_metadata``); the frames belong here, because they describe the tensors
    rather than any kernel's operator outputs.  A kernel that cannot read a frame as stated --
    arch22's TND kernels ignore ``seqused`` and anchor their mask at the frame's end -- restates it
    at its own call site (``override/a3_workaround.py``) rather than here.
    """

    cp_layout = DOC_CHUNK_LAYOUT
    """The layout these frames describe (``--parallelism.context-parallel-load-balancer``).

    A rank-local entry of this stack exists for the doc-chunk arrangement only, so the CP seam can
    resolve an unnamed layout to it instead of guessing.
    """

    cu_seq_q: torch.Tensor
    """Packed document boundaries of the row (``int32``, ``n_documents + 1``): the offsets
    Attention Gym's ``gather_attn`` takes as ``cu_seqlens`` and the kernels as ``cu_seqlens_*``."""

    cu_seq_k: torch.Tensor
    """The key side's boundaries this entry addresses: the row's own for a layer that does not
    pool, its ``compress_ratio``-folded grid for one that does.  Precomputed by the backend that
    knows the ratio, so the core reads a boundary set instead of dividing one per layer."""

    position_k: torch.Tensor
    """The key stream's own document-relative positions (``[B, C]``, ``C`` the row under one degree).

    The queries rotate at the positions carried beside the input, which are the query frame's by
    construction, so the entry states only the other side: the stream the key projections read.
    At one degree that is the input's ``positions``; under CP it is the restored window's, so a
    consumer never derives them and never learns that the stream moved.
    """

    ratio: int
    """The compression ratio this entry describes; every consumer grids this ratio."""

    q: KernelFrame
    """The query frame: the row's boundaries at one degree, this rank's chunks under CP."""

    swa_k: KernelFrame
    """The sliding-window keys' frame: the whole row at every degree, with ``seqused`` the
    per-document reach."""

    cmp_k: KernelFrame | None
    """The compressed pool's frame (``None`` for a geometry that does not pool)."""

    restore: RestorePlan | None = None
    """Under context parallelism the key side's movement: the input already holds this rank's
    chunks, and the key projections read the slab this restores them onto."""

    def key_stream(self, x: torch.Tensor, *, group) -> torch.Tensor:
        """The stream this entry's key projections read: the input's, or this rank's window.

        At one degree the layout left the row in place, so the input *is* the stream; under
        context parallelism it is the window this rank's chunks are restored onto.  A consumer
        hands over the stream as the layout produced it and never branches on the degree.
        """
        if self.restore is None:
            return x
        return self.restore.apply(x.unsqueeze(0), group=group).squeeze(0)

    @staticmethod
    def prepare_cp_metadata(
        attention_metadata: DeepSeekV41Metadata,
        *,
        permutation: torch.Tensor | None,
    ) -> DeepSeekV41Metadata:
        """Refuse context parallel: only the fused stack has a rank-local form.

        The row is the whole packed sequence, and this core hands one grid to both sides of the
        attention, so a rank that holds a chunk has no metadata it could attend with.  A CP run
        therefore stops here, before any layer runs, and the fused entry (``AscV41Metadata``)
        overrides this with the frames and records built from the rank's own chunks.
        """
        del attention_metadata, permutation
        raise ValueError(
            "the V4.1 reference stack does not support context parallel: it hands one grid to "
            "both sides of the attention, so it cannot describe a rank-local row. Enable the "
            "fused stack (override.deepseek_v4_1.sparse_attn.asc and "
            "override.deepseek_v4_1.lightning_indexer.asc) to run with "
            "--parallelism.context-parallel-degree."
        )


def build_v41_frames(
    cu_seq_q: torch.Tensor,
    cu_seq_k: torch.Tensor,
    ratio: int,
) -> tuple[KernelFrame, KernelFrame, KernelFrame | None]:
    """The row's frames: boundary differences as the prefixes, the pool folded at this ratio."""
    seqused = torch.diff(cu_seq_q).to(torch.int32)
    q = KernelFrame(cu_seqlens=cu_seq_q, seqused=seqused)
    swa_k = KernelFrame(cu_seqlens=cu_seq_q, seqused=seqused)
    cmp_k = None
    if ratio > 0:
        cmp_k = KernelFrame(
            cu_seqlens=cu_seq_k,
            seqused=(seqused // ratio).to(torch.int32),
            # A ratio of one addresses the row itself: the axis is not folded, so there is no
            # trailing group and the frame keeps ``residual`` unset.  A3's kernels take zeros
            # for it; A5's reject a value, so the frame's ``None`` passes through.
            residual=None if ratio == 1 else (seqused % ratio).to(torch.int32),
        )
    return q, swa_k, cmp_k


class DeepSeekV41TransformerBlock(TransformerBlock):
    """Transformer block with HC mixing around attention and the MoE feed-forward."""

    @dataclass(kw_only=True, slots=True)
    class Config(TransformerBlock.Config):
        # Redeclared with the V4.1 types so config builders can reach the fields the
        # sharding and MTP helpers need.
        attention: Attention.Config  # pyrefly: ignore [bad-override]
        feed_forward: MoE.Config  # pyrefly: ignore [bad-override]
        hc_attn_pre: HcPre.Config
        hc_ffn_pre: HcPre.Config
        hc_post: HcPost.Config
        engram: Engram.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        cfg = config
        self.moe_enabled = True
        self.attention = cfg.attention.build()
        self.attention_norm = cfg.attention_norm.build()
        self.ffn_norm = cfg.ffn_norm.build()
        self.feed_forward = cfg.feed_forward.build()
        self.hc_attn_pre = cfg.hc_attn_pre.build()
        self.hc_ffn_pre = cfg.hc_ffn_pre.build()
        self.hc_post = cfg.hc_post.build()
        self.engram = cfg.engram.build() if cfg.engram is not None else None
        self._remat_encode_optional_outputs = False

    def configure_remat_regions(self, save_patterns, recompute_patterns=()):
        super().configure_remat_regions(save_patterns, recompute_patterns)
        self._remat_encode_optional_outputs = True

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        attention_metadata: DeepSeekV41Metadata,
        positions: torch.Tensor | None = None,
        *,
        pre_mix: torch.Tensor,
        cmp_k: torch.Tensor | None = None,
        idx_k: torch.Tensor | None = None,
        topk_indices: torch.Tensor | None = None,
        topk_scores: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        engram_token_history: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Returns ``(x, pre_mix, cmp_k, idx_k, topk_indices, topk_scores, candidates)``.

        ``pre_mix`` is the attention-input coefficient the next block must consume, and
        the shared attention tensors are this block's contribution to the chain.
        ``image_mask`` is the block's only modality input and comes last, after the
        cross-layer attention state: the text stack's forward signature is that state
        and nothing else.  It is ``None`` whenever the caller has no modality, which is
        every text-only forward, and both consumers already read that as "no modality".
        """
        if self.engram is not None:
            x = self.engram(x, input_ids, positions, image_mask=image_mask, token_history=engram_token_history)
        residual = x
        x, attn_pre, post, comb = self.hc_attn_pre(x, pre_mix)
        x, cmp_k, idx_k, topk_indices, topk_scores, candidates = self.attention(
            self.attention_norm(x),
            positions,
            attention_metadata,
            cmp_k=cmp_k,
            idx_k=idx_k,
            topk_indices=topk_indices,
            topk_scores=topk_scores,
            candidates=candidates,
        )
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)

        residual = x
        x, ffn_pre, post, comb = self.hc_ffn_pre(x, attn_pre)
        x = self.feed_forward(
            self.ffn_norm(x),
            image_mask=image_mask,
            padding_mask_T=padding_mask,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(x)
        x = self.hc_post(x, residual, post, comb)
        outputs = (x, ffn_pre, cmp_k, idx_k, topk_indices, topk_scores, candidates)
        if self._remat_encode_optional_outputs:
            # torch_remat's checkpoint boundary only accepts tensors. Attention
            # carry tensors have rank >= 2, so a scalar bool marks an absent slot.
            return tuple(value if value is not None else x.new_zeros((), dtype=torch.bool) for value in outputs)
        return outputs


def _layer_attention_metadata(attention_metadata: AttentionMetadataMap, layer: nn.Module) -> DeepSeekV41Metadata:
    """The metadata one layer reads: its own map entry, keyed by its backend.

    Upstream #4897 hands every layer the entry its own attention backend named, which is what
    lets one stack carry several backends -- or, with a geometry key, several geometries of one
    backend -- at once.  The key is the layer's own, so a missing entry means the config built
    no metadata for a backend the layer runs: a failure, not a reason to hand the layer another
    geometry's grid.
    """
    key = layer.attention.attention_metadata_key  # pyrefly: ignore [missing-attribute]
    metadata = attention_metadata.get(key)
    if metadata is None:
        raise ValueError(f"no attention metadata was built for {key!r}")
    # The map's value type is the pinned union (a Flex/Varlen mask or a varlen metadata); this
    # stack's backends only ever publish the row contract, which is what the layer reads.
    return cast("DeepSeekV41Metadata", metadata)


# The two ports that, together, make V4.1's fused attention: the core emits the indexer's
# teacher on ``topk_scores``' gradient and the selector consumes it.  Both gates below read
# the override list a run requests -- the config stage runs before ``apply_overrides``
# installs them, so the request is what this stage has.
_FUSED_ATTENTION_OVERRIDES = (
    "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
    "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
)


def _requested_fused_attention(override) -> tuple[bool, bool]:
    """``(fused core, fused selector)``, as the run's overrides request them."""
    requested = {entry if isinstance(entry, str) else entry[0] for entry in override.imports or ()}
    core, selector = _FUSED_ATTENTION_OVERRIDES
    return core in requested, selector in requested


def _check_indexer_teacher_pair(fused_core: bool, fused_selector: bool) -> None:
    """The fused indexer teacher is a pair: one port emits it, one consumes it.

    ``topk_scores`` is carried only between the two kernel backwards.  The emitting port
    writes the raw teacher on ``topk_scores``' gradient; the consuming selector is the only
    thing that ever reads that edge.  Fusing one side alone therefore breaks the objective --
    silently, since neither half raises on its own -- so the mismatch is rejected here instead
    of being discovered from a flat training curve.

    Neither half is a usable fallback:

    - Selector without the fused core: ``topk_scores``' gradient is ``None``, so SLIKG
      returns zero gradient and the indexer never updates.
    - Fused core without the fused selector: the reference selector does pick the teacher up
      off that edge, but it applies attention-level marginals to summed per-head indexer
      logits.  The two sides are not the same quantity, so the resulting training signal is
      wrong rather than merely different.

    The check reads the *requested* overrides rather than the attention they install, because
    the request is what this stage has -- and renaming a port means renaming it here too.
    """
    if fused_core == fused_selector:
        return
    if fused_core:
        active, missing = "sparse_attn.asc", "lightning_indexer.asc"
        consequence = "the reference selector would take the teacher onto summed per-head logits"
    else:
        active, missing = "lightning_indexer.asc", "sparse_attn.asc"
        consequence = "topk_scores would carry no teacher"
    raise ValueError(
        f"A fused V4.1 attention core and a fused selector must be enabled together: {active} is "
        f"active without {missing}, so {consequence} and the indexer cannot be trained correctly. "
        "Enable both or neither. See torchtitan_npu/override/README.md#deepseek-v41."
    )


class DeepSeekV41Model(GeometryAttentionMetadataMixin, Decoder):
    """DeepSeek-V4.1 text backbone: CSA2 policy over the packed varlen contract.

    No MTP depths, no vision tower, no modality input: the stack consumes tokens,
    positions and the model-owned metadata, and nothing else.  The multimodal stack is
    :class:`DeepSeekV41MultimodalModel`, which only changes how the input embeddings are
    built and whether an ``image_mask`` reaches the blocks.
    """

    @classmethod
    def _register_optimizer_hooks(cls, optimizers, model_parts, parallelism_context):
        from torchtitan.models.common.aux_loss import register_aux_loss_zero_hook
        from torchtitan.models.common.moe import register_moe_load_balancing_hook

        from torchtitan_npu.models.common.deepseek_lora import is_lora_parameter

        if not any(is_lora_parameter(name) for model in model_parts for name, _ in model.named_parameters()):
            register_moe_load_balancing_hook(optimizers, model_parts, parallelism_context)
        register_aux_loss_zero_hook(optimizers, model_parts, parallelism_context)

    def apply_model_compile(self, model_compile):
        from .parallelize import _apply_compile_v4_1

        _apply_compile_v4_1(self, compile_config=model_compile)

    def _apply_fsdp(self, **kwargs):
        from .parallelize import apply_deepseek_v4_1_fsdp

        apply_deepseek_v4_1_fsdp(self, **kwargs)

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        # Whether this stack can run under context parallelism: the text stack is the one
        # the CP layout is being built for, and the multimodal config overrides this to
        # ``False`` until the vision path has a CP story.
        #
        # The flag lives on the config because the gate that reads it runs on the config
        # tree, before any module exists.
        accepts_context_parallel: ClassVar[bool] = True

        n_layers: int
        hc_mult: int = 4
        compress_ratios: tuple[int, ...]
        kv_source_layers: tuple[int, ...] = ()
        index_source_layers: tuple[int, ...] = ()
        candidate_source_layer: int = 20
        candidate_topk_blocks: int = 2048
        candidate_block_size: int = 8
        fp32_param_patterns: tuple[str, ...] = ()
        """Parameter name patterns retained in FP32 under FSDP mixed precision."""

        def update_from_config(self, *, config, **kwargs) -> None:
            from torchtitan.models.common.linear import Linear

            for _, linear, _, _ in self.traverse(Linear.Config):
                linear.inplace_wgrad_accum = False
            parallelism = config.parallelism
            if parallelism.tensor_parallel_degree != 1:
                raise NotImplementedError("DeepSeek V4.1 currently supports TP=1 only")
            pp = parallelism.pipeline_parallel_degree
            if pp != 1:
                raise NotImplementedError(f"DeepSeek V4.1 does not support pipeline parallelism; got PP={pp}")
            fused_core, fused_selector = _requested_fused_attention(config.override)
            _check_indexer_teacher_pair(fused_core, fused_selector)
            self._check_context_parallel(parallelism.context_parallel_degree, fused_core, fused_selector)

            if len(self.compress_ratios) != self.n_layers:
                raise ValueError(
                    f"compress_ratios must match n_layers ({self.n_layers}), got {len(self.compress_ratios)}"
                )

            if hasattr(config, "training"):
                from torchtitan.models.common.rope import RoPE

                max_context_length = config.training.max_context_length
                for _, rope_cfg, _, _ in self.traverse(RoPE.Config):
                    assert isinstance(rope_cfg, RoPE.Config)
                    rope_cfg.max_context_length = max_context_length

            engram_configs = [layer.engram for layer in self.layers if layer.engram is not None]
            ep_degree = max(1, parallelism.expert_parallel_degree)
            for engram_cfg in engram_configs:
                table_cfg = engram_cfg.table
                if (
                    table_cfg.require_token_id_map
                    and table_cfg.token_id_map_path is None
                    and table_cfg.tokenizer_path is None
                ):
                    if config.hf_assets_path is None:
                        raise ValueError(
                            "Engram tokenizer compression is required, but neither "
                            "table.token_id_map_path nor hf_assets_path is configured."
                        )
                    table_cfg.tokenizer_path = os.path.join(
                        config.hf_assets_path,
                        "tokenizer.json",
                    )
                if table_cfg.num_embeddings % ep_degree != 0:
                    raise ValueError(
                        f"Engram physical table size ({table_cfg.num_embeddings}) must "
                        f"be divisible by EP degree ({ep_degree}). Increase "
                        "EngramArgs.table_padding_multiple without changing it "
                        "between checkpoints."
                    )

            # DeepEP/HybridEP size their buffers from the training shape; v0.4.0 no longer
            # runs the upstream helper that used to fill these fields.
            from torchtitan.models.common.moe.token_dispatcher import (
                DeepEPTokenDispatcher,
                HybridEPTokenDispatcher,
            )

            buffer_dispatchers = [
                layer.feed_forward.routed_experts.token_dispatcher
                for layer in self.layers
                if isinstance(
                    layer.feed_forward.routed_experts.token_dispatcher,
                    (DeepEPTokenDispatcher.Config, HybridEPTokenDispatcher.Config),
                )
            ]
            if buffer_dispatchers:
                if parallelism.expert_parallel_degree == 1:
                    raise ValueError("DeepEP/HybridEP token dispatch requires expert_parallel_degree > 1.")
                num_token_shards = parallelism.context_parallel_degree * parallelism.tensor_parallel_degree
                num_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
                if num_tokens % num_token_shards != 0:
                    raise ValueError(
                        "training.num_tokens_per_microbatch_per_dp_rank "
                        f"({num_tokens}) must be divisible by "
                        "context_parallel_degree * tensor_parallel_degree "
                        f"({num_token_shards}) so CP and TP/SP produce equal local "
                        "token counts."
                    )
                required_capacity = num_tokens // num_token_shards
                for dispatcher in buffer_dispatchers:
                    if dispatcher.num_max_tokens_per_rank is None:
                        dispatcher.num_max_tokens_per_rank = required_capacity
                    elif dispatcher.num_max_tokens_per_rank < required_capacity:
                        raise ValueError(
                            f"{type(dispatcher).__qualname__} num_max_tokens_per_rank "
                            f"({dispatcher.num_max_tokens_per_rank}) is smaller than the "
                            f"required per-rank capacity ({required_capacity})."
                        )
                    if dispatcher.hidden_dim is None:
                        dispatcher.hidden_dim = self.dim

            from .sharding import set_deepseek_v4_1_sharding_config

            set_deepseek_v4_1_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def _check_context_parallel(self, context_parallel_degree: int, fused_core: bool, fused_selector: bool) -> None:
            """Reject a CP degree this stack cannot run.

            Context parallelism requires the fused attention: the sharded frames describe this
            rank's query chunks while the KV frames stay the gathered row, and the fused core
            and selector are the consumers that read that difference (``seqused``).  A
            torch-native core or selector grids the row instead -- silently, since every shape
            still fits -- so a run that does not ask for both ports is rejected.

            The multimodal stack cannot be sharded at all, so its config overrides
            :attr:`accepts_context_parallel`; reading the class attribute rather than branching
            on the config class keeps that half one line and lets the text stack reach the rest.
            """
            if context_parallel_degree == 1:
                return
            if not type(self).accepts_context_parallel:
                raise NotImplementedError(
                    "DeepSeek V4.1 rejects context parallelism on the multimodal stack; "
                    f"got CP={context_parallel_degree}"
                )
            missing = [
                name
                for name, requested in zip(_FUSED_ATTENTION_OVERRIDES, (fused_core, fused_selector), strict=True)
                if not requested
            ]
            if missing:
                raise ValueError(
                    "DeepSeek V4.1 runs context parallelism on the fused stack only: this run "
                    f"does not request {', '.join(missing)}. Request both "
                    "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc and "
                    "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc, or run with "
                    f"context_parallel_degree=1 (got CP={context_parallel_degree}). See "
                    "torchtitan_npu/override/README.md#deepseek-v41."
                )

        def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
            """Estimate V4.1 model FLOPs per token using TorchTitan's MFU convention.

            Engram table weights are sparse lookup storage and do not belong in the
            dense ``6P`` term. Hierarchical Reindex is charged for logical candidate
            work rather than eager-reference overcompute.
            """
            from typing import cast

            from torchtitan.models.common.flops import get_nparams_and_active_nparams

            deepseek_v4_1_model = cast("DeepSeekV41Model", model)
            first_attention = self.layers[0].attention
            excluded = []
            for layer in deepseek_v4_1_model.layers.values():
                engram = cast("DeepSeekV41TransformerBlock", layer).engram
                if engram is not None:
                    excluded.append(engram.table)
            excluded.extend(
                getattr(deepseek_v4_1_model, name, None) for name in ("vision_encoder", "image_marker_embeddings")
            )
            nparams, active_nparams = get_nparams_and_active_nparams(
                deepseek_v4_1_model, modules_excluded_from_active_params=excluded
            )
            num_flops_per_token = 6 * active_nparams

            for layer in self.layers:
                attention = layer.attention
                num_flops_per_token += (
                    6
                    * attention.n_heads
                    * (2 * attention.head_dim)
                    * min(seq_len, attention.inner_attention.window_size)
                )

                if attention.compress_ratio > 0:
                    compressed_seq_len = seq_len // attention.compress_ratio
                    num_flops_per_token += (
                        6
                        * attention.n_heads
                        * (2 * attention.head_dim)
                        * min(attention.indexer.index_topk, compressed_seq_len)
                    )

                    indexer = attention.indexer
                    if indexer.mode is not IndexerMode.REUSE:
                        indexer_seq_len = compressed_seq_len
                        # The pool's geometry lives on the selector -- the one node that
                        # builds or searches it -- so the estimate is read off the same
                        # config the layers read it from and cannot drift from it.  The
                        # positivity test is the kernels' own, matching the module's
                        # ``Selector.has_candidate_pool``.
                        selector = indexer.selector
                        if indexer.mode is IndexerMode.REINDEX and selector.candidate_topk_blocks > 0:
                            # MFU counts the model's bounded candidate search, not
                            # wider eager-reference scoring that is masked afterwards.
                            candidate_seq_len = selector.candidate_topk_blocks * selector.candidate_block_size
                            indexer_seq_len = min(indexer_seq_len, candidate_seq_len)

                        num_flops_per_token += 6 * indexer.num_index_heads * indexer.index_head_dim * indexer_seq_len
            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__(config)
        cfg = config
        self.hc_mult = cfg.hc_mult
        self.compress_ratios = tuple(cfg.compress_ratios)
        self.engram_history_size = max(
            (max(layer.engram.table.ngram_orders) - 1 for layer in cfg.layers if layer.engram is not None),
            default=0,
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallelism_context,
        parallelism,
        **kwargs,
    ):
        """Build Engram's per-row token history, then run the shared seam.

        Engram hashes a token against the ``engram_history_size`` tokens before it *inside its
        document*.  Under context parallel those predecessors are not one window of the row: rank
        ``r`` holds chunk ``r`` of every document, so ``[lo - H, lo)`` of the global row is
        another document's tail, and one window cannot state a different set of predecessors per
        row.  They are built here, before the shard, as ``[C, H]`` -- entry ``d - 1`` is the
        ``d``-th token back for each of *this rank's* rows -- and the entry's own ``positions``
        do the document mask, so a row whose document starts sooner reads none of them.  At one
        degree there is nothing to build: the model's row is its local rows.
        """
        positions = input_dict.get("positions")
        history = None
        if self.engram_history_size and parallelism_context.cp_enabled:
            if positions is None:
                raise ValueError("Engram's token history needs the row's document-relative positions.")
            history = self._engram_row_history(input_dict["input"], positions, parallelism_context)
        inputs, labels, extra_kwargs, loss_kwargs = super().preprocess_inputs(  # pyrefly: ignore [not-callable]
            input_dict,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
            **kwargs,
        )
        if history is not None:
            import spmd_types as spmd
            from torchtitan.distributed.spmd_types import annotate_input_spmd_types
            from torchtitan.models.common.decoder_sharding import dense_activation_placement

            annotate_input_spmd_types(
                parallelism_context,
                {"engram_token_history": history},
                {"engram_token_history": dense_activation_placement(tp=spmd.R, cp=spmd.S(0))},
            )
            extra_kwargs["engram_token_history"] = history
        return inputs, labels, extra_kwargs, loss_kwargs

    def _engram_row_history(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        parallelism_context,
    ) -> torch.Tensor:
        """Each of this rank's rows' predecessors, one entry per distance (``[C, H]``).

        ``doc_chunk_permutation`` states which row of the whole row each local row is, so this
        gather is the permutation the shard itself applies: a local row's predecessor is the row
        before it *in its document*, which the permutation has already placed the right number of
        slots earlier.  Nothing moves, and no rank sees another rank's rows.
        """
        cp_mesh = parallelism_context.get_mesh("cp")
        cp_size, cp_rank = cp_mesh.size(), cp_mesh.get_local_rank()
        slab_len = tokens.shape[0] // cp_size
        rows = doc_chunk_permutation(document_starts(positions), cp_size, slab_len)[0]
        local_rows = rows[cp_rank * slab_len : (cp_rank + 1) * slab_len]
        history = torch.empty(
            (slab_len, self.engram_history_size),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        for distance in range(1, self.engram_history_size + 1):
            # A row with fewer than ``distance`` predecessors in its document has none; clamping
            # keeps the gather in range and the entry's positions mask the value out.
            history[:, distance - 1] = tokens[(local_rows - distance).clamp(min=0)]
        return history

    def _embedded_inputs(
        self,
        tokens: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the input embedding stream of one forward.

        The text stack takes a caller-supplied stream or embeds the tokens itself.  The
        multimodal stack keeps this contract and derives the modality mask in its own
        override, because the mask is an input to the stack rather than part of the
        embedding stream.
        """
        tok_embeddings = self.tok_embeddings
        if input_embeds is not None:
            return input_embeds
        if tok_embeddings is None:
            return tokens
        return tok_embeddings(tokens)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        attention_metadata: AttentionMetadataMap,
        input_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        engram_token_history: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        aux_loss_denominators: torch.Tensor | None = None,
    ):
        """V4.1 forward with Single-Pass mHC on a folded token stream.

        Embeddings are ``[T, D]``; each block carries residual branches
        ``[T, hc, D]`` and input-mixing coefficients ``[T, hc]``.

        The cross-layer attention state is threaded through the stack as ordinary local
        variables: an attention source layer returns the tensors it produced, and every
        other layer returns what it was handed.  ``image_mask`` is the stack's only
        modality input: it is ``None`` on the text stack, which never has one, and the
        multimodal stack hands it the mask it derives from ``token_types``.

        ``attention_metadata`` is the seam's per-layer map -- one entry per distinct
        attention-backend key, built once per microbatch -- and every layer reads the entry
        its own backend named.
        """
        hidden = self._embedded_inputs(tokens, input_embeds=input_embeds)
        input_ids = tokens.detach().long()
        hidden = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)

        pre_mix = HcPre.identity_pre_mix(hidden, self.hc_mult)
        cmp_k: torch.Tensor | None = None
        idx_k: torch.Tensor | None = None
        topk_indices: torch.Tensor | None = None
        topk_scores: torch.Tensor | None = None
        candidates: torch.Tensor | None = None
        for layer in self.layers.values():
            (
                hidden,
                pre_mix,
                cmp_k,
                idx_k,
                topk_indices,
                topk_scores,
                candidates,
            ) = layer(
                hidden,
                input_ids,
                _layer_attention_metadata(attention_metadata, layer),
                positions,
                pre_mix=pre_mix,
                cmp_k=cmp_k,
                idx_k=idx_k,
                topk_indices=topk_indices,
                topk_scores=topk_scores,
                candidates=candidates,
                image_mask=image_mask,
                engram_token_history=engram_token_history,
                padding_mask=padding_mask,
                aux_loss_denominator=None if aux_loss_denominators is None else aux_loss_denominators[0],
            )
            if layer._remat_encode_optional_outputs:
                cmp_k, idx_k, topk_indices, topk_scores, candidates = (
                    None if value.ndim == 0 and value.dtype == torch.bool else value
                    for value in (cmp_k, idx_k, topk_indices, topk_scores, candidates)
                )

        # The stack has no learned output head: it collapses with the last block's
        # attention-input coefficients.
        main_hidden = HcPre.collapse(hidden, pre_mix)
        main_hidden = self.norm(main_hidden) if self.norm is not None else main_hidden

        if self._skip_lm_head or self.lm_head is None:
            return main_hidden
        return self.lm_head(main_hidden)


class DeepSeekV41MultimodalModel(DeepSeekV41Model):
    """DeepSeek-V4.1 backbone with the vision tower wired in.

    Everything the text stack does is inherited; the multimodal half is the three
    pieces that only exist with images: the tower and the markers, the input-embedding
    construction that scatters visual features into the token positions, and the two
    parallelization hooks that reach the tower.  The ``image_mask`` it derives is the
    single modality input the residual-stream consumers read.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV41Model.Config):
        # Context parallelism is defined for the text stack; the vision path has no CP
        # contract yet, so the multimodal stack keeps rejecting it.
        accepts_context_parallel: ClassVar[bool] = False

        # Required rather than optional: a config with neither is the text stack, and
        # that is a different class.
        vision_encoder: DeepSeekV41VisionEncoder.Config
        image_marker_embeddings: ImageMarkerEmbeddings.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.vision_encoder = config.vision_encoder.build()
        self.image_marker_embeddings = config.image_marker_embeddings.build()

    def preprocess_inputs(self, input_dict, *, parallelism_context, parallelism, **kwargs):
        # The decoder seam annotates its own inputs; the vision inputs have their
        # own DP-local layout and must be handled by the multimodal model.
        vision_names = ("pixel_values", "image_grid", "image_feature_indices", "token_types")
        vision_inputs = {name: input_dict[name] for name in vision_names if name in input_dict}
        decoder_inputs = {name: value for name, value in input_dict.items() if name not in vision_inputs}
        inputs, labels, extra_kwargs, loss_kwargs = super().preprocess_inputs(
            decoder_inputs, parallelism_context=parallelism_context, parallelism=parallelism, **kwargs
        )
        from torchtitan.distributed.spmd_types import annotate_input_spmd_types
        from torchtitan.models.common.decoder_sharding import token_id_placement
        from torchtitan.models.common.vision_encoder_sharding import multimodal_input_sharding

        vision_layout = multimodal_input_sharding()["pixel_values"]
        annotate_input_spmd_types(
            parallelism_context,
            vision_inputs,
            {
                "pixel_values": vision_layout,
                "image_grid": vision_layout,
                "image_feature_indices": token_id_placement(),
                "token_types": token_id_placement(),
            },
        )
        extra_kwargs.update(vision_inputs)
        return inputs, labels, extra_kwargs, loss_kwargs

    def apply_activation_checkpointing_extensions(self, policy) -> None:
        """Apply the shared AC policy to the V4.1 vision blocks.

        ``_wrap_block`` is private because the pinned torchtitan exposes no public hook
        for wrapping a block that is not part of a ``Decoder`` layer list.
        """
        for name, block in self.vision_encoder.blocks.named_children():
            self.vision_encoder.blocks.register_module(
                name,
                policy._wrap_block(block, base_fqn=f"vision_encoder.blocks.{name}"),
            )

    def apply_fsdp_extensions(self, *, dp_mesh, dp_mesh_dims, training, parallelism, parallelism_context) -> None:
        """FSDP-wrap the V4.1 vision tower before the shared decoder wrapper."""
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import apply_fsdp_to_multimodal_encoder

        apply_fsdp_to_multimodal_encoder(
            self.vision_encoder,
            dp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
        )

    def _modality_inputs(
        self,
        tokens: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid: torch.Tensor | None = None,
        image_feature_indices: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(embeddings, image_mask)`` for one forward.

        The mask is derived here rather than only where visual features are scattered,
        so a text-only batch of the multimodal stack still masks its Engram n-grams and
        routes with the load-balancing bias.  It is ``None`` when the batch carries no
        token types at all, which the consumers read as "no modality".
        """
        image_mask = token_types.ge(0) if token_types is not None else None
        if pixel_values is None:
            if input_embeds is not None:
                return input_embeds, image_mask
            hidden = self.tok_embeddings(tokens)
            return hidden, image_mask
        if image_grid is None or image_feature_indices is None:
            raise ValueError("pixel_values requires image_grid and image_feature_indices")
        return self._scatter_visual_features(
            tokens,
            pixel_values=pixel_values,
            image_grid=image_grid,
            image_feature_indices=image_feature_indices,
            token_types=token_types,
        ), image_mask

    def _scatter_visual_features(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor,
        image_grid: torch.Tensor,
        image_feature_indices: torch.Tensor,
        token_types: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the tower and write its features into the token embeddings."""
        from .vision.data import scatter_image_features

        if pixel_values.ndim == 4:
            pixel_values = pixel_values.flatten(0, 1)
        if image_grid.ndim == 3:
            image_grid = image_grid.flatten(0, 1)
        hidden = self.tok_embeddings(tokens)
        visual = self.vision_encoder(pixel_values, image_grid)
        if token_types is not None:
            hidden = self.image_marker_embeddings(hidden, token_types)
        ratio = self.vision_encoder.aligner.downsample_ratio
        counts = ((image_grid[:, 0].to(torch.long) + ratio - 1) // ratio) * (
            (image_grid[:, 1].to(torch.long) + ratio - 1) // ratio
        )
        flat_visual = (
            torch.cat(
                [features[: int(count)] for features, count in zip(visual, counts.tolist(), strict=True)],
                dim=0,
            )
            if counts.numel()
            else visual.new_empty((0, visual.shape[-1]))
        )
        return scatter_image_features(hidden, flat_visual, image_feature_indices)

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        *,
        attention_metadata: AttentionMetadataMap,
        pixel_values: torch.Tensor | None = None,
        image_grid: torch.Tensor | None = None,
        image_feature_indices: torch.Tensor | None = None,
        token_types: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ):
        """The text forward plus the modality inputs the vision tower consumes.

        The image tensors are keyword-only and translated into the one modality value
        the stack understands, so the rest of the forward -- the cross-layer state
        thread, the mHC collapse and the head -- is the inherited implementation.
        """
        from torchtitan.distributed.spmd_types import spmd_local_context

        # Images and their grids are independent on each data-parallel rank.
        with spmd_local_context("dp"):
            embeds, image_mask = self._modality_inputs(
                tokens,
                input_embeds=input_embeds,
                pixel_values=pixel_values,
                image_grid=image_grid,
                image_feature_indices=image_feature_indices,
                token_types=token_types,
            )
        return super().forward(
            tokens,
            positions,
            attention_metadata=attention_metadata,
            input_embeds=embeds,
            image_mask=image_mask,
            padding_mask=padding_mask,
        )
