# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Prebuild full and half-batch attention metadata for CV chunking."""

from dataclasses import dataclass
from itertools import pairwise
from typing import cast

import torch
from torchtitan.config import derive, override
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.common.metadata_extension import LightningIndexerMetadata, SparseAttentionMetadata
from torchtitan_npu.models.deepseek_v4.metadata import (
    CompressedVarlenMetadata,
    build_compressed_varlen_metadata,
    register_pytree_node_for_dataclass,
)
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedVarlenMetadata,
    AscMetadataExtension,
)

from . import _BATCH_CHUNK_PAIR_ATTR


@dataclass(kw_only=True, slots=True)
class BatchChunkAscCompressedVarlenMetadata(AscCompressedVarlenMetadata):
    batch_chunk_tensors: tuple[tuple[torch.Tensor, ...], ...] | None = None


register_pytree_node_for_dataclass(BatchChunkAscCompressedVarlenMetadata)


def _plain_batch_chunk_metadata(
    metadata: CompressedVarlenMetadata,
) -> tuple[CompressedVarlenMetadata, CompressedVarlenMetadata]:
    """Build two local plain-stream contracts at an aligned token boundary."""
    if metadata.window is not None:
        raise ValueError("batch chunk metadata does not support context parallelism")
    varlen = metadata.varlen
    if not torch.equal(varlen.cu_seq_q, varlen.cu_seq_k):
        raise ValueError("batch chunk metadata requires cu_seq_q == cu_seq_k")
    cu_host = metadata.cu_seq_q_host
    if cu_host is None:
        cu_host = tuple(int(value) for value in varlen.cu_seq_q.cpu().tolist())
    total_tokens = cu_host[-1]
    if total_tokens < 2 or total_tokens % 2:
        raise ValueError(f"batch chunk metadata requires an even positive token extent, got {total_tokens}")
    split_token = total_tokens // 2
    if split_token not in cu_host:
        raise ValueError(
            "batch chunk metadata requires documents not to cross the equal "
            f"batch boundary at token {split_token}; boundaries={cu_host}"
        )
    split_index = cu_host.index(split_token)
    ratios = tuple(metadata.plans)
    qk_share_storage = varlen.cu_seq_k is varlen.cu_seq_q
    chunks = []
    for start_index, end_index, token_offset in (
        (0, split_index, 0),
        (split_index, len(cu_host) - 1, split_token),
    ):
        local_host = tuple(boundary - token_offset for boundary in cu_host[start_index : end_index + 1])
        local_cu = (varlen.cu_seq_q[start_index : end_index + 1] - token_offset).contiguous()
        local_lengths = tuple(right - left for left, right in pairwise(local_host))
        local_max = max(local_lengths)
        local_varlen = VarlenMetadata(
            cu_seq_q=local_cu,
            # Preserve the full contract's pytree alias structure. A shared
            # q/k tensor contributes one runtime role; distinct tensors
            # contribute two roles and are remapped independently.
            cu_seq_k=local_cu if qk_share_storage else local_cu.clone(),
            max_q=local_max,
            max_k=local_max,
            cu_seq_q_host=local_host,
        )
        chunks.append(build_compressed_varlen_metadata(local_varlen, ratios))
    return cast(
        "tuple[CompressedVarlenMetadata, CompressedVarlenMetadata]",
        tuple(chunks),
    )


def _metadata_tensor_fields(
    metadata: AscCompressedVarlenMetadata,
) -> dict[str, torch.Tensor]:
    """Return tensor fields whose full/chunk values have identical roles."""
    tensors: dict[str, torch.Tensor] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, torch.Tensor):
            tensors[name] = value

    add("varlen.cu_seq_q", metadata.varlen.cu_seq_q)
    if metadata.varlen.cu_seq_k is not metadata.varlen.cu_seq_q:
        add("varlen.cu_seq_k", metadata.varlen.cu_seq_k)
    for ratio, plan in metadata.plans.items():
        for field_name in (
            "cu_seqlens_cmp_k",
            "block_remainder",
            "gather_indices",
            "block_positions",
            "first_indices",
            "compressed_rows",
            "cmp_k_global_gather_indices",
            "compressor_state_block_table",
        ):
            add(f"plans.{ratio}.{field_name}", getattr(plan, field_name))
    for ratio, plan in metadata.asc_plans.items():
        for field_name in (
            "smla_metadata",
            "smla_grad_metadata",
            "li_metadata",
            "slig_metadata",
        ):
            add(f"asc_plans.{ratio}.{field_name}", getattr(plan, field_name))
    return tensors


def _tag_batch_chunk_metadata(
    full: AscCompressedVarlenMetadata,
    chunks: tuple[AscCompressedVarlenMetadata, AscCompressedVarlenMetadata],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Tag and pack only tensor roles consumed by the full-batch graph.

    Vendor builders may produce additional optional tensors for a chunk's
    document layout. Select only full-batch roles so those extra tensors and
    data-dependent host fields cannot change the runtime pytree signature.
    """
    full_fields = _metadata_tensor_fields(full)
    chunk_fields = tuple(_metadata_tensor_fields(chunk) for chunk in chunks)
    missing_by_chunk = {
        chunk_id: sorted(set(full_fields) - set(fields))
        for chunk_id, fields in enumerate(chunk_fields)
        if set(full_fields) - set(fields)
    }
    if missing_by_chunk:
        raise RuntimeError(f"batch-chunk metadata is missing full-batch tensor roles: {missing_by_chunk}")
    for pair_id, full_tensor in full_fields.items():
        setattr(full_tensor, _BATCH_CHUNK_PAIR_ATTR, (pair_id, -1))
        for chunk_id, fields in enumerate(chunk_fields):
            setattr(
                fields[pair_id],
                _BATCH_CHUNK_PAIR_ATTR,
                (pair_id, chunk_id),
            )
    return cast(
        "tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]",
        tuple(tuple(fields[pair_id] for pair_id in full_fields) for fields in chunk_fields),
    )


class BatchChunkMetadataExtension(AscMetadataExtension):
    @dataclass(kw_only=True, slots=True)
    class Config(AscMetadataExtension.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self._chunk_li_metadata: LightningIndexerMetadata | None = None

    def __call__(self, metadata) -> BatchChunkAscCompressedVarlenMetadata:
        assert self._chunk_li_metadata is not None, "cv_parallel must bind the model's LI metadata provider"
        build_metadata = super().__call__
        full = build_metadata(metadata)
        first, second = _plain_batch_chunk_metadata(metadata)
        chunks = (
            build_metadata(self._chunk_li_metadata(first)),
            build_metadata(self._chunk_li_metadata(second)),
        )
        return BatchChunkAscCompressedVarlenMetadata(
            varlen=full.varlen,
            plans=full.plans,
            window=full.window,
            seq_len_host=full.seq_len_host,
            cu_seq_q_host=full.cu_seq_q_host,
            asc_plans=full.asc_plans,
            batch_chunk_tensors=_tag_batch_chunk_metadata(full, chunks),
        )


@override(
    target=SparseAttentionMetadata.Config,
    exact=True,
    description="Precompute AscendC metadata for two equal batch chunks",
)
def asc_metadata(cfg: SparseAttentionMetadata.Config) -> SparseAttentionMetadata.Config:
    # Register the pipeline only when this experimental override is activated.
    from . import cv_parallel  # noqa: F401

    return derive(cfg, SparseAttentionMetadata.Config, extension=BatchChunkMetadataExtension)
