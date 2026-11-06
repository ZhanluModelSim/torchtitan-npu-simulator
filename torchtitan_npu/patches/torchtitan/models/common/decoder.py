# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4897
# Backport upstream PR: https://github.com/pytorch/torchtitan/pull/4116
# Remove this module (and its activation import in
# ``torchtitan_npu/patches/torchtitan/__init__.py``) once the pinned
# ``torchtitan`` dependency carries both PRs natively.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-owned attention metadata over the pinned Decoder input contract.

Metadata is keyed by backend geometry, transformed with the typed CP load
balancer, then passed through the native ``shard_tensors`` and SPMD annotator.
"""

from typing import Any

import torch
from torchtitan.distributed import context_parallel
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import annotate_input_spmd_types
from torchtitan.models.common.attention import BaseAttention, FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.decoder_sharding import decoder_input_sharding

from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import (
    AttentionMetadataMap,
    cp_shard_context_set,
    prepare_cp_metadata_map,
)

__all__ = ["apply"]


def _get_attention_metadata(
    self,
    positions: torch.Tensor,
    *,
    padding_mask: torch.Tensor | None = None,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
) -> dict[Any, Any]:
    """Build one metadata entry per distinct attention-backend key.

    Upstream #4897's body.  The traversal differs only in where it finds the
    inner attention: 0.3.0 has no ``InnerAttention.Config`` base to traverse
    for, so it walks the attention wrapper configs and reads their
    ``inner_attention``.  Entries whose builder returns ``None`` are absent,
    exactly as upstream.
    """
    attention_metadata: dict[Any, Any] = {}
    # MTP depths are layers too: a decoder that keeps them in their own config list (upstream's
    # ``MTPDecoder.Config.mtp_layers``) runs attention backends whose geometries the map has to
    # cover, and with one entry per geometry a depth whose ratio no main layer pools at would
    # otherwise have no entry at all.
    for layer_config in (
        layer for layers in (self.config.layers, getattr(self.config, "mtp_layers", None) or []) for layer in layers
    ):
        traverse = getattr(layer_config, "traverse", None)
        if traverse is None:
            continue
        for _, attention_config, _, _ in traverse(BaseAttention.Config):
            inner_attention = getattr(attention_config, "inner_attention", None)
            if inner_attention is None:
                continue
            key = inner_attention.attention_metadata_key
            if key in attention_metadata:
                continue
            metadata = inner_attention.build_attention_metadata(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
            if metadata is not None:
                attention_metadata[key] = metadata
    return attention_metadata


def forward(
    self,
    tokens: torch.Tensor,
    positions: torch.Tensor | None = None,
    attention_masks: Any = None,
    *,
    attention_metadata: AttentionMetadataMap | None = None,
    padding_mask: torch.Tensor | None = None,
):
    """Upstream #4897's forward: select each layer's metadata by its key.

    ``attention_masks`` keeps its 0.3.0 meaning (one object broadcast to every
    layer) until the stacks that still build a single mask type are migrated;
    passing ``attention_metadata`` selects per layer instead.  The rest of the
    body is 0.3.0's, unchanged.
    """
    # positions is listed before attention_masks so AutoParallel's input_fn,
    # which returns (tokens, positions) and binds them positionally, maps
    # positions to the right parameter (it would otherwise land in the
    # attention_masks slot and break the maskless SDPA backend).
    # passthrough for nonexistent layers, allows easy configuration of pipeline parallel stages
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

    for layer in self.layers.values():
        layer_attention_metadata = (
            attention_masks
            if attention_metadata is None
            else attention_metadata.get(layer.attention.attention_metadata_key)
        )
        h = layer(
            h,
            layer_attention_metadata,
            positions,
            padding_mask=padding_mask,
        )

    h = self.norm(h) if self.norm is not None else h

    # _skip_lm_head is an attribute rather than a forward kwarg because PP backward
    # calls .requires_grad on all stage inputs, which fails on bool kwargs.
    # TODO: fix PP backward upstream to skip non-tensor inputs
    if self._skip_lm_head:
        return h
    output = self.lm_head(h) if self.lm_head is not None else h
    return output


def _cp_shard(
    self,
    input_dict: dict[str, Any],
    input_shardings: dict[str, Any],
    parallelism_context: Any,
    parallelism: Any,
) -> dict[str, Any]:
    """Prepare rank-local backend metadata with the native typed CP shard."""
    from torchtitan_npu.models.common.restore import DOC_CHUNK_LAYOUT
    from torchtitan_npu.patches.torchtitan.distributed.context_parallel import DocChunkCPLoadBalancer
    from torchtitan_npu.patches.torchtitan.models.common.attention_metadata import cp_layout_of

    metadata = input_dict.get("attention_metadata")
    positions = input_dict.get("positions")
    balancer_config = parallelism.context_parallel_load_balancer
    if cp_layout_of(metadata) == DOC_CHUNK_LAYOUT:
        if balancer_config is not None and not isinstance(balancer_config, DocChunkCPLoadBalancer.Config):
            raise ValueError("Document-chunk attention metadata requires DocChunkCPLoadBalancer.Config.")
        balancer_config = DocChunkCPLoadBalancer.Config()
    layout = (
        DOC_CHUNK_LAYOUT
        if isinstance(balancer_config, DocChunkCPLoadBalancer.Config)
        else "headtail"
        if isinstance(balancer_config, context_parallel.HeadTailCPLoadBalancer.Config)
        else "ptrr"
        if isinstance(balancer_config, context_parallel.PTRRFlexAttentionCPLoadBalancer.Config)
        else None
    )
    cp_mesh = parallelism_context.get_mesh(MeshAxisName.CP)
    with cp_shard_context_set(cp_mesh=cp_mesh, load_balancer_type=layout, positions=positions):
        balancer = (
            balancer_config.build(
                seq_len=context_parallel.get_cp_input_seq_len(input_dict, input_shardings=input_shardings),
                attention_metadata=metadata or input_dict.get("attention_masks"),
            )
            if balancer_config is not None
            else None
        )
        permutation = balancer.generate_permutation() if balancer is not None else None
        if metadata:
            input_dict["attention_metadata"] = prepare_cp_metadata_map(metadata, permutation=permutation)
        elif "attention_masks" in input_dict:
            # Native CP attention backends own their metadata transformation.
            from torchtitan.models.common.cp_attention import CPInnerAttention

            prepared = set()
            for _, config, _, _ in self.config.traverse(CPInnerAttention.Config, recurse=True):
                owner = config._owner
                if owner not in prepared:
                    input_dict["attention_masks"] = owner.prepare_cp_metadata(
                        input_dict["attention_masks"], permutation=permutation
                    )
                    prepared.add(owner)
        return context_parallel.shard_tensors(input_dict, input_shardings=input_shardings, permutation=permutation)


def preprocess_inputs(
    self,
    input_dict: dict[str, Any],
    *,
    parallelism_context: Any,
    parallelism: Any,
    max_num_documents: int | None = None,
    max_context_length: int | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build backend metadata, apply native CP sharding, and annotate inputs."""
    del kwargs
    positions = input_dict.get("positions")
    if positions is not None:
        metadata = self._get_attention_metadata(
            positions,
            padding_mask=input_dict.get("padding_mask"),
            max_num_documents=max_num_documents,
            max_context_length=max_context_length,
        )
        if metadata:
            input_dict["attention_metadata"] = metadata
        elif isinstance(
            self.config.first_full_attention_backend, (FlexInnerAttention.Config, VarlenInnerAttention.Config)
        ):
            input_dict["attention_masks"] = self.get_attention_masks(
                positions=positions,
                padding_mask=input_dict.get("padding_mask"),
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )
    input_shardings = decoder_input_sharding()
    if parallelism_context.cp_enabled:
        input_dict = self._cp_shard(
            input_dict,
            input_shardings=input_shardings,
            parallelism_context=parallelism_context,
            parallelism=parallelism,
        )
    input_dict = annotate_input_spmd_types(parallelism_context, input_dict, input_shardings)
    inputs, labels = input_dict.pop("input"), input_dict.pop("labels")
    return inputs, labels, input_dict


def apply() -> None:
    if getattr(Decoder, "_npu_metadata_seam", False):
        return
    Decoder._get_attention_metadata = _get_attention_metadata  # pyrefly: ignore [bad-assignment]
    Decoder._cp_shard = _cp_shard  # pyrefly: ignore [bad-assignment]
    Decoder.forward = forward  # pyrefly: ignore [bad-assignment]
    Decoder.preprocess_inputs = preprocess_inputs  # pyrefly: ignore [bad-assignment]
    Decoder._npu_metadata_seam = True


apply()
