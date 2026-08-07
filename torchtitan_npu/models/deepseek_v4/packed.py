# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend-independent packed-sequence metadata for DeepSeek-V4 SMLA.

The model retains its public ``[B, S, ...]`` layout. This module maps valid
tokens and complete compressed blocks to compact streams shared by eager
reference code and fused TND kernels; backend operator metadata stays outside.
"""

from itertools import count
from typing import NamedTuple

import torch
from torchtitan.models.common.attention import VarlenMetadata

_PACKED_METADATA_IDS = count()


class CompressionInfo(NamedTuple):
    """Varlen stream and storage mapping for one compression ratio.

    Queries retain the parent token boundaries, while keys use the sequence's
    complete compressed-block boundaries. The remaining fields map canonical
    packed blocks to the model's fixed ``[B, S // ratio, D]`` storage.
    """

    ratio: int
    varlen: VarlenMetadata
    lengths: torch.Tensor
    """Complete blocks per sequence; ``varlen.cu_seq_k`` is its cumsum."""
    total_blocks: int
    sequence_ranges: tuple[tuple[int, int], ...]
    """Python block ranges used by eager reference kernels."""
    residual: torch.Tensor
    """Tokens per sequence left over below one full block."""
    block_starts: torch.Tensor
    block_positions: torch.Tensor
    document_ids: torch.Tensor
    storage_indices: torch.Tensor


class DSV4PackedMetadata(NamedTuple):
    """Packed token metadata keyed by compression ratio.

    ``varlen`` describes the original token stream and each ``compressed``
    entry describes one compressed-key stream. Host values such as
    ``total_tokens`` are cached to avoid repeated device synchronization.
    """

    varlen: VarlenMetadata
    lengths: torch.Tensor
    total_tokens: int
    sequence_ranges: tuple[tuple[int, int], ...]
    """Python token ranges used by eager reference kernels."""
    compressed: dict[int, CompressionInfo]
    token_indices: torch.Tensor
    token_sequence_ids: torch.Tensor
    """Logical sequence id for every token in canonical packed-token order."""
    token_positions: torch.Tensor
    """Position within the logical sequence for every canonical packed token."""
    container_batch_size: int
    """Number of rows in the model-facing ``[B, S, ...]`` container."""
    container_seq_len: int
    """Fixed token capacity ``S`` of each model-facing container row."""
    cache_id: int

    @property
    def num_sequences(self) -> int:
        """Number of independent logical sequences in the packed container."""

        return self.lengths.numel()

    def compression_for_ratio(self, ratio: int) -> CompressionInfo:
        try:
            return self.compressed[ratio]
        except KeyError:
            raise KeyError(
                f"No compressed sequence metadata for ratio={ratio}."
            ) from None


def compact_token_tensor(
    tensor: torch.Tensor,
    metadata: DSV4PackedMetadata,
) -> torch.Tensor:
    """Gather a ``[B,S,...]`` tensor into canonical packed-token order."""

    if tensor.shape[:2] != (
        metadata.container_batch_size,
        metadata.container_seq_len,
    ):
        raise ValueError("Token tensor shape does not match DSV4 packed metadata.")
    return tensor.flatten(0, 1).index_select(0, metadata.token_indices)


def restore_token_tensor(
    compact: torch.Tensor,
    metadata: DSV4PackedMetadata,
    *,
    fill_value: int | float = 0,
) -> torch.Tensor:
    """Scatter canonical packed tokens back to their ``[B,S,...]`` container."""

    if compact.shape[0] != metadata.total_tokens:
        raise ValueError("Compact token count does not match DSV4 packed metadata.")
    output = compact.new_full(
        (
            metadata.container_batch_size * metadata.container_seq_len,
            *compact.shape[1:],
        ),
        fill_value,
    )
    output.index_copy_(0, metadata.token_indices, compact)
    return output.view(
        metadata.container_batch_size,
        metadata.container_seq_len,
        *compact.shape[1:],
    )


def compact_compressed_tensor(
    tensor: torch.Tensor,
    metadata: DSV4PackedMetadata,
    ratio: int,
) -> torch.Tensor:
    """Gather fixed physical compressed storage into canonical block order."""

    compressed = metadata.compression_for_ratio(ratio)
    expected = (
        metadata.container_batch_size,
        metadata.container_seq_len // ratio,
    )
    if tensor.shape[:2] != expected:
        raise ValueError("Compressed tensor storage shape does not match metadata.")
    return tensor.flatten(0, 1).index_select(0, compressed.storage_indices)


def _cu_seqlens(lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Return (cu_seqlens, int32 lengths, max_seqlen, total) for one stream."""

    lengths = lengths.to(dtype=torch.int32)
    cu_seqlens = torch.cat(
        [lengths.new_zeros((1,)), lengths.cumsum(dim=0, dtype=torch.int32)]
    )
    max_seqlen = int(lengths.max().item()) if lengths.numel() else 0
    total = int(cu_seqlens[-1].item()) if lengths.numel() else 0
    return cu_seqlens, lengths, max_seqlen, total


def _sequence_ranges(lengths: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """Materialize cumulative Python ranges for eager reference paths.

    Tensor lengths remain authoritative for operator ABIs. The Python form
    avoids repeated ``item()`` or ``tolist()`` calls in eager layer loops.
    """

    ranges = []
    start = 0
    for length in lengths.tolist():
        end = start + int(length)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _sequence_layout(
    positions: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return compact token indices/positions, sequence lengths, and ids."""

    batch_size, seqlen = positions.shape
    flat_indices: list[torch.Tensor] = []
    flat_positions: list[torch.Tensor] = []
    sequence_lengths: list[torch.Tensor] = []

    for batch_idx in range(batch_size):
        valid_cols = torch.nonzero(valid_tokens[batch_idx], as_tuple=False).flatten()
        if valid_cols.numel() == 0:
            continue
        row_positions = positions[batch_idx, valid_cols]
        starts = torch.nonzero(row_positions == 0, as_tuple=False).flatten()
        if starts.numel() == 0 or bool(starts[0] != 0):
            starts = torch.cat([starts.new_zeros((1,)), starts])
        ends = torch.cat([starts[1:], starts.new_tensor((valid_cols.numel(),))])

        for start, end in zip(starts.unbind(), ends.unbind(), strict=True):
            doc_positions = row_positions[start:end]
            expected = torch.arange(
                doc_positions.numel(),
                device=positions.device,
                dtype=positions.dtype,
            )
            if not torch.equal(doc_positions, expected):
                raise ValueError(
                    "DeepSeek-V4 packed positions must reset to 0 and increase "
                    "contiguously within each sequence."
                )
            sequence_lengths.append((end - start).reshape(1).to(torch.int32))

        flat_indices.append(batch_idx * seqlen + valid_cols)
        flat_positions.append(row_positions)

    if not sequence_lengths:
        raise ValueError(
            "DeepSeek-V4 packed metadata requires at least one valid token."
        )

    token_indices = torch.cat(flat_indices).to(torch.int64)
    compact_positions = torch.cat(flat_positions)
    lengths = torch.cat(sequence_lengths)
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=positions.device),
        lengths.to(torch.int64),
    )
    return token_indices, compact_positions, lengths, document_ids


def _compression_info(
    ratio: int,
    original: VarlenMetadata,
    original_lengths: torch.Tensor,
    compact_positions: torch.Tensor,
    token_indices: torch.Tensor,
    container_batch_size: int,
    container_seq_len: int,
) -> CompressionInfo:
    lengths = torch.div(original_lengths, ratio, rounding_mode="floor")
    residual = original_lengths - ratio * lengths
    original_starts = original.cu_seq_q[:-1].to(torch.int64)

    block_starts = []
    for doc_start, num_blocks in zip(
        original_starts.unbind(), lengths.unbind(), strict=True
    ):
        block_starts.append(
            doc_start
            + torch.arange(
                int(num_blocks.item()),
                device=compact_positions.device,
                dtype=torch.int64,
            )
            * ratio
        )
    starts = (
        torch.cat(block_starts)
        if block_starts
        else torch.empty((0,), dtype=torch.int64, device=compact_positions.device)
    )
    block_positions = compact_positions[starts]
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=compact_positions.device),
        lengths.to(torch.int64),
    )
    blocks_per_row = container_seq_len // ratio
    if starts.numel():
        row_ids = torch.div(
            token_indices[starts], container_seq_len, rounding_mode="floor"
        )
        row_counts = torch.bincount(row_ids, minlength=container_batch_size).to(
            torch.int64
        )
        row_offsets = torch.cat(
            [row_counts.new_zeros((1,)), row_counts.cumsum(dim=0)[:-1]]
        )
        slots = (
            torch.arange(starts.numel(), device=starts.device) - row_offsets[row_ids]
        )
        storage_indices = row_ids * blocks_per_row + slots
    else:
        storage_indices = starts.new_empty((0,))
    cu_blocks, block_lengths, max_blocks, total_blocks = _cu_seqlens(lengths)
    return CompressionInfo(
        ratio=ratio,
        # Cross-attention: original tokens query this ratio's compressed blocks.
        varlen=VarlenMetadata(
            cu_seq_q=original.cu_seq_q,
            cu_seq_k=cu_blocks,
            max_q=original.max_q,
            max_k=max_blocks,
        ),
        lengths=block_lengths,
        total_blocks=total_blocks,
        sequence_ranges=_sequence_ranges(block_lengths),
        residual=residual,
        block_starts=starts,
        block_positions=block_positions,
        document_ids=document_ids,
        storage_indices=storage_indices,
    )


def _layout_from_positions(
    positions: torch.Tensor,
    valid_tokens: torch.Tensor | None,
) -> tuple[VarlenMetadata, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build both the varlen stream and the token layout from ``[B, S]``."""

    if positions.ndim != 2:
        raise ValueError(
            f"DeepSeek-V4 positions must have shape [B, S], got {positions.shape}."
        )
    if valid_tokens is None:
        valid_tokens = torch.ones_like(positions, dtype=torch.bool)
    elif valid_tokens.shape != positions.shape:
        raise ValueError("valid_tokens must have the same shape as positions.")
    else:
        valid_tokens = valid_tokens.to(dtype=torch.bool)

    token_indices, compact_positions, lengths, sequence_ids = _sequence_layout(
        positions, valid_tokens
    )
    cu_tokens, lengths, max_tokens, total_tokens = _cu_seqlens(lengths)
    # Q and K share the same tensor object. CP uses identity to recognize
    # self-attention without a device-to-host equality check.
    varlen = VarlenMetadata(
        cu_seq_q=cu_tokens, cu_seq_k=cu_tokens, max_q=max_tokens, max_k=max_tokens
    )
    return varlen, token_indices, compact_positions, lengths, sequence_ids, total_tokens


def _layout_from_varlen(
    cu_seq_q: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Derive an unpadded token layout from varlen boundaries and positions.

    ``cu_seq_q`` is authoritative for document boundaries, including after CP
    sharding. ``positions`` supplies the ``[B, S]`` container shape and each
    token's document-relative position.
    """

    container_batch_size, container_seq_len = positions.shape
    total_tokens = int(cu_seq_q[-1].item())
    if total_tokens != container_batch_size * container_seq_len:
        raise ValueError(
            f"varlen stream covers {total_tokens} tokens but positions describe "
            f"a {container_batch_size}x{container_seq_len} container."
        )

    device = positions.device
    lengths = torch.diff(cu_seq_q).to(torch.int32)
    repeats = lengths.to(torch.int64)
    # Cumulative lengths must start at zero and cover the full local container;
    # otherwise the repeated sequence ids would not align with token storage.
    if int(repeats.sum().item()) != total_tokens:
        raise ValueError(
            f"varlen stream accounts for {int(repeats.sum().item())} of "
            f"{total_tokens} tokens; the first document must start at token 0. "
            "Context-parallel shards are not supported yet."
        )
    token_indices = torch.arange(total_tokens, device=device, dtype=torch.int64)
    compact_positions = positions.reshape(-1)
    sequence_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device), repeats
    )

    # Validate the same contiguous-position invariant as ``_sequence_layout``:
    # a token's document-relative position is its index minus document start.
    expected_positions = token_indices - torch.repeat_interleave(
        cu_seq_q[:-1].to(torch.int64), repeats
    )
    if not torch.equal(compact_positions.to(torch.int64), expected_positions):
        raise ValueError(
            "DeepSeek-V4 packed positions must reset to 0 and increase "
            "contiguously within each sequence."
        )
    return token_indices, compact_positions, lengths, sequence_ids, total_tokens


def build_dsv4_packed_metadata(
    attention_masks,
    compress_ratios: tuple[int, ...] | list[int],
    *,
    positions: torch.Tensor | None = None,
    valid_tokens: torch.Tensor | None = None,
) -> DSV4PackedMetadata:
    """Build the compression plan for one rank-local token stream.

    The mask handler calls this once per batch after optional CP sharding, so
    every token and block mapping describes the model's local container.

    Args:
        attention_masks: Rank-local varlen metadata, or a ``[B, S]`` positions
            tensor for direct construction without sharding.
        compress_ratios: Compression ratios to plan; values at most 1 are ignored.
        positions: Rank-local positions, required with varlen metadata.
        valid_tokens: Optional padding mask for direct positions input only.
    """

    if isinstance(attention_masks, torch.Tensor):
        if positions is not None:
            raise ValueError(
                "Pass positions either as attention_masks or as the positions "
                "keyword, not both."
            )
        positions = attention_masks
        (
            varlen,
            token_indices,
            compact_positions,
            lengths,
            sequence_ids,
            total_tokens,
        ) = _layout_from_positions(positions, valid_tokens)
    else:
        if not hasattr(attention_masks, "cu_seq_q"):
            raise TypeError(
                "build_dsv4_packed_metadata expects a varlen stream or a "
                f"positions tensor, got {type(attention_masks)}."
            )
        if positions is None:
            raise ValueError(
                "positions is required when building from a varlen stream: it "
                "carries the [B, S] container shape, which VarlenMetadata does "
                "not."
            )
        if valid_tokens is not None:
            raise ValueError(
                "valid_tokens is only accepted when building metadata from positions."
            )
        varlen = attention_masks
        (
            token_indices,
            compact_positions,
            lengths,
            sequence_ids,
            total_tokens,
        ) = _layout_from_varlen(varlen.cu_seq_q, positions)

    container_batch_size, container_seq_len = positions.shape

    ratios = sorted({int(ratio) for ratio in compress_ratios if ratio > 1})
    compressed = {
        ratio: _compression_info(
            ratio,
            varlen,
            lengths,
            compact_positions,
            token_indices,
            container_batch_size,
            container_seq_len,
        )
        for ratio in ratios
    }
    metadata = DSV4PackedMetadata(
        varlen=varlen,
        lengths=lengths,
        total_tokens=total_tokens,
        sequence_ranges=_sequence_ranges(lengths),
        compressed=compressed,
        token_indices=token_indices,
        token_sequence_ids=sequence_ids,
        token_positions=compact_positions,
        container_batch_size=container_batch_size,
        container_seq_len=container_seq_len,
        cache_id=next(_PACKED_METADATA_IDS),
    )
    # A fixed ``[B, S]`` container may hold different numbers of documents,
    # valid tokens, and compressed blocks. Mark only those leading dimensions
    # dynamic so Dynamo does not specialize on batch contents.
    dynamic_tensors = [
        metadata.varlen.cu_seq_q,
        metadata.lengths,
        metadata.token_indices,
        metadata.token_sequence_ids,
        metadata.token_positions,
    ]
    for compressed_info in metadata.compressed.values():
        dynamic_tensors.extend(
            [
                compressed_info.varlen.cu_seq_k,
                compressed_info.lengths,
                compressed_info.residual,
                compressed_info.block_starts,
                compressed_info.block_positions,
                compressed_info.document_ids,
                compressed_info.storage_indices,
            ]
        )
    for tensor in dynamic_tensors:
        torch._dynamo.maybe_mark_dynamic(tensor, 0)
    return metadata
