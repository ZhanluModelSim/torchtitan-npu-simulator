# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Document-packed compression metadata for DeepSeek-V4.

Built once per batch by the model's ``build_attention_masks`` from a
``VarlenMetadata`` stream.  The container grid is ``[1, S]`` (the DSV4 packed
scenario runs with ``local_batch_size == 1``; raise ``seq_len`` instead of
``local_batch_size``), so ``batch_size == 1`` and ``seq_len`` is the total
token count ``cu_seq_q[-1]``.

This module carries only the **common** contract — the kernel-contract
``plans`` (``cu_seqlens_cmp_k`` / ``block_remainder`` / ``gather_indices`` /
``block_positions`` / ``first_indices`` per ratio) consumed by the
Compressor, the kernels, and every attention path.  Two further layers live
outside this module:

- the **reference tier** (per-token document ids/positions, the dense
  attendability mask, the container-slot scatter, the static block
  listing) is delivered by the ``metadata_extension`` seam
  (``reference.py``'s ``ReferenceMetadataExtension``, the default);
- the **context-parallel layer** (the plan builder + the dispatcher) lives
  in ``token_dispatcher.py``: a ratio-independent ``WindowPlan`` on the
  metadata (``window``) and the per-ratio block plans at ``plans[ratio]``,
  whose part 1 is derived directly over the plan blocks.

The kernel-layout derivation is the **plain-stream** contract:

- ``build_kernel_layout`` derives the per-document plans from the document
  boundaries alone (each document contributes its complete leading blocks,
  gathered contiguously; the ``len % ratio`` tail produces no entry).  It
  refuses context-parallel-shaped streams — those plans come from
  ``build_cp_plan`` (``token_dispatcher.py``).

Documents never span rows, and complete blocks never cross documents: a row's
compressed region is the concatenation of its documents' complete blocks,
padded to ``S // ratio`` slots.
"""

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

import torch
from torchtitan.models.common.attention import VarlenMetadata

if TYPE_CHECKING:
    from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import (
        CPVarlenMetadata,
    )

    from .token_dispatcher import ExchangePlan, WindowPlan

__all__ = [
    "CompressedBlockLayout",
    "CompressedKernelContract",
    "CompressedVarlenMetadata",
    "build_compressed_varlen_metadata",
    "build_kernel_layout",
]


@dataclass(kw_only=True, slots=True)
class CompressedBlockLayout:
    """Kernel contract for one compression ratio (the key of ``plans``).

    All tensors are built once per batch by the plan build.  Their leading
    dimensions are marked dynamic by the fused path so ``torch.compile``
    does not specialize on batch contents.

    The plan has two parts:

    - **part 1 — the unified contract** (CP and non-CP): the kernel tensors
      and the compressor contract (``gather_indices`` / ``block_positions``
      / ``first_indices``), consumed identically by the Compressor's
      dispatcher and the kernels regardless of context parallel;
    - **part 2 — the dispatcher fields** (CP only, ``None`` without): the
      block exchange routing (``exchange``), the container packing
      (``compressed_rows`` / ``out_width``) and the compressed-level
      gather (``cmp_k_global_gather_indices``).
    """

    # ---- part 1: the unified contract (CP and non-CP) ----
    cu_seqlens_cmp_k: torch.Tensor | None = None
    """Cumulative compressed-block lengths over the packed stream (int32),
    ``[n_seqs + 1]``.  Entry ``i`` is the global start index of sequence
    ``i``'s compressed blocks.  ``None`` for ratio-1 plans."""

    n_cmp_blocks_host: int | None = None
    """Host-cached ``cu_seqlens_cmp_k[-1]`` (total compressed blocks), set at
    the eager plan-build boundary so ``_assemble_tnd`` avoids a per-layer
    ``.item()`` D2H sync inside the compiled region."""

    block_remainder: torch.Tensor | None
    """Per-sequence block remainder (int32), ``[n_seqs]``: ``len[i] % ratio``
    trailing tokens of sequence ``i`` fall short of one full block and
    produce no compressed KV entry.  ``None`` for ratio-1 plans."""

    gather_indices: torch.Tensor | None
    """The pooled block-row indices (int64) the dispatcher's ``gather``
    applies.  Without context parallel they are the contiguous doc-major
    block-token indices into the local stream (``x.flatten(0, 1)``) —
    the plain local gather; under context parallel they are the assembly
    into ``cat([x_local, exchange_recv_rows])`` in pooled order — the
    remote gather + permute.  Reshape to ``[n_blocks, ratio, D]`` after
    the gather.  ``None`` for ratio-1 plans."""

    block_positions: torch.Tensor | None = None
    """Document-relative block starts (int32) ``[n_blocks]`` — the RoPE
    positions of the pooled keys (block ``b`` of a document sits at
    ``b * ratio``).  The compressor contract: precomputed once per batch so
    no per-layer derivation is needed.  ``None`` for ratio-1 plans."""

    first_indices: torch.Tensor | None = None
    """Document-first block ids (int64) ``[n_segs]`` — the positions of the
    first block of every segment in the ``gather_indices`` block order,
    i.e. ``cu_seqlens_cmp_k[:-1]``.  The overlap validity mask: the borrowed
    (previous-block) rows of these blocks are masked to zero weight.  The
    compressor contract: precomputed once per batch.  ``None`` for ratio-1
    plans."""

    # ---- part 2: the dispatcher fields (CP only) ----
    exchange: "ExchangePlan | None" = None
    """The block exchange routing (alltoallv input/output splits) of the
    projected ``kv``/``score`` rows completing the plan blocks'
    ``[A, B)`` sub-range.  ``None`` without context parallel."""

    compressed_rows: torch.Tensor | None = None
    """Container-packing selection (int64 ``[n_kept]``): the positions of
    the kept blocks in the pooled key stream (the borrow-source blocks are
    dropped).  ``None`` without context parallel (all blocks are kept)."""

    out_width: int | None = None
    """The container grid width: ``seq_len // ratio`` without context
    parallel, the uniform ``max_kept`` shard width under CP (every rank's
    container is a valid ``S(1)`` shard).  ``None`` for ratio-1 plans."""

    cmp_k_global_gather_indices: torch.Tensor | None = None
    """The compressed-K global gather (int64 ``[sum seqlen_k // ratio]``) —
    the compressed analogue of ``k_global_gather_indices``: per-segment
    full-prefix blocks as offsets into the all-gathered
    ``[cp * out_width, D]`` container (the ShardingConfig all-gather's
    output).  ``None`` without context parallel."""


@dataclass(kw_only=True, slots=True)
class CompressedVarlenMetadata:
    """The DeepSeek-V4 varlen attention contract (the common part).

    Carried as ``attention_masks`` through the DSA layers.  Built by
    the model's ``build_attention_masks`` from a ``VarlenMetadata``
    stream: the kernel contract (``plans``).  The ``metadata_extension``
    (e.g. the AscendC kernel metadata, or the reference tier) post-processes
    it into the concrete per-path contract.

    The container grid is ``[1, S]`` with ``S`` equal to the total token
    count (``local_batch_size == 1``): ``batch_size`` is always ``1`` and
    ``seq_len`` is ``cu_seq_q[-1]`` — both derived as properties, not
    stored.  Without context parallel, ``cu_seq_q`` and ``cu_seq_k`` are
    the same tensor (consumers read ``varlen.cu_seq_q`` / ``varlen.cu_seq_k``
    directly; under context parallel the ori cumsum is the ``window``
    plan's ``cu_seqlens_ori_kv``).
    """

    varlen: "VarlenMetadata | CPVarlenMetadata"
    """Token-stream boundaries (``cu_seq_q`` / ``cu_seq_k``).  Under context
    parallel the rank-local ``CPVarlenMetadata`` (the shard path's own
    builder output)."""

    plans: dict[int, CompressedBlockLayout]
    """Kernel contract for each ratio present in the model.  Ratio-1 plans
    describe no compressed region."""

    window: "WindowPlan | None" = None
    """The sliding-window plan (CP only, ``None`` without): the ratio-
    independent window exchange + ori-stream assembly the Attention's
    ``swa_k`` gather consumes (``token_dispatcher.WindowPlan``)."""

    seq_len_host: int | None = None
    """Host-cached total token count (``cu_seq_q[-1]``), set once at the eager
    ``build_compressed_varlen_metadata`` boundary so the ``seq_len`` property
    avoids a per-layer ``.item()`` D2H sync inside the compiled region."""

    @property
    def batch_size(self) -> int:
        """Container batch size (``1`` for the current packed scenario)."""
        return 1

    @property
    def seq_len(self) -> int:
        """Container sequence length (the total token count)."""
        if self.seq_len_host is not None:
            return self.seq_len_host
        return int(self.varlen.cu_seq_q[-1].item())


@runtime_checkable
class CompressedKernelContract(Protocol):
    """The per-ratio plan shape shared by the model-dir metadata and the
    NPU slim type: the compressor-contract plans (``gather_indices``,
    ``block_positions``, ``first_indices``) plus the kernel tensors.

    Consumers that read only the plans (e.g. the Compressor) accept this
    protocol instead of the concrete metadata type, so the fused path's
    slim metadata (which carries no reference tier) satisfies the contract
    without the model directory knowing about it.
    """

    plans: dict[int, CompressedBlockLayout]


def build_kernel_layout(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> dict[int, CompressedBlockLayout]:
    """The kernel-contract tier: the per-ratio plans for a plain stream.

    Called once per batch by ``build_compressed_varlen_metadata``; the
    AscendC path materializes no reference tier (its extension only consumes
    the plans).  The container grid is ``[1, S]`` with ``S`` equal to the
    total token count, so the layout is derived purely from the document
    boundaries and the ratios.  Context-parallel streams must use
    ``build_cp_plan`` instead (guarded below).
    """
    if not hasattr(varlen, "cu_seq_q"):
        raise TypeError(f"build_kernel_layout expects a varlen stream, got {type(varlen)}.")

    cu_seq_q = varlen.cu_seq_q
    if int(cu_seq_q[0].item()) != 0:
        raise ValueError(f"varlen stream must start at token 0, got cu_seq_q[0]={cu_seq_q[0]}.")
    if not torch.equal(cu_seq_q, varlen.cu_seq_k):
        raise ValueError(
            "build_kernel_layout requires a plain stream (cu_seq_q == "
            "cu_seq_k); context-parallel plans come from build_cp_plan."
        )
    # The packed scenario runs with local_batch_size == 1; raise seq_len
    # instead of local_batch_size.
    seq_len = int(cu_seq_q[-1].item())

    # The plain per-document derivation: each document contributes its
    # complete leading blocks (the ``len % ratio`` tail produces no
    # compressed entry), gathered contiguously document by document.  The
    # plans carry no dispatcher fields — the dispatcher's gather degrades
    # to the plain local gather, so the forward path never special-cases
    # context parallel.
    cu = cu_seq_q.cpu().tolist()
    lengths = [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]
    distinct_ratios = sorted({int(r) for r in compress_ratios})
    device = cu_seq_q.device
    plans: dict[int, CompressedBlockLayout] = {}
    for ratio in distinct_ratios:
        if ratio == 1:
            plans[1] = CompressedBlockLayout(
                cu_seqlens_cmp_k=None,
                block_remainder=None,
                gather_indices=None,
            )
            continue
        if ratio not in (4, 128):
            raise NotImplementedError(f"CompressedBlockLayout does not support ratio={ratio}; expected 1, 4, or 128.")
        c_lens = [length // ratio for length in lengths]
        cu_seqs = torch.cat(
            [
                torch.zeros((1,), dtype=torch.int32, device=device),
                torch.tensor(c_lens, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32),
            ]
        )
        pieces = [
            torch.arange(k_start, k_start + ratio * cnt, dtype=torch.int64, device=device)
            for k_start, cnt in zip(cu[:-1], c_lens, strict=True)
            if cnt
        ]
        positions = [torch.arange(0, ratio * cnt, ratio, dtype=torch.int32, device=device) for cnt in c_lens if cnt]
        gather = torch.cat(pieces, dim=0) if pieces else torch.empty((0,), dtype=torch.int64, device=device)
        block_positions = (
            torch.cat(positions, dim=0) if positions else torch.empty((0,), dtype=torch.int32, device=device)
        )
        # Doc-start block ids of the docs that actually have blocks
        # (``cu[i] < cu[i+1]``); zero-block docs contribute no blocks to
        # mask, and a trailing zero-block doc's boundary would be an
        # out-of-range index.
        first_indices = cu_seqs[:-1][torch.diff(cu_seqs) > 0].to(torch.int64)
        plans[ratio] = CompressedBlockLayout(
            cu_seqlens_cmp_k=cu_seqs,
            n_cmp_blocks_host=sum(length // ratio for length in lengths),
            block_remainder=torch.tensor(
                [length % ratio for length in lengths],
                dtype=torch.int32,
                device=device,
            ),
            gather_indices=gather,
            block_positions=block_positions,
            first_indices=first_indices,
            compressed_rows=None,
            out_width=seq_len // ratio,
        )
    return plans


def build_compressed_varlen_metadata(
    varlen: VarlenMetadata,
    compress_ratios: tuple[int, ...] | list[int],
) -> CompressedVarlenMetadata:
    """Build the DSV4 varlen contract for one rank-local token stream.

    Called once per batch by the model's ``build_attention_masks``: the
    common kernel contract (``build_kernel_layout``).  The reference tier
    and the vendor kernel tensors are filled by the ``metadata_extension``
    (``reference.py`` / the AscendC override).

    Args:
        varlen: Rank-local ``VarlenMetadata``.  ``cu_seq_q`` is
            authoritative for document boundaries.
        compress_ratios: Compression ratios present in the model.
    """
    plans = build_kernel_layout(varlen, compress_ratios)
    # Cache the total token count on the host so the ``seq_len`` property
    # avoids a per-layer ``.item()`` D2H sync inside the compiled region.
    # Built once here (eager boundary) from ``cu_seq_q[-1]``.
    return CompressedVarlenMetadata(
        varlen=varlen,
        plans=plans,
        seq_len_host=int(varlen.cu_seq_q[-1].item()),
    )


def register_pytree_node_for_dataclass(cls: type) -> None:
    """Register a kw-only dataclass as a pytree node (idempotent).

    The graph_trainer's ``minimal_fx_tracer`` requires every ``attention_masks``
    leaf to be a tensor/primitive; an unregistered dataclass is rejected as a
    single non-primitive leaf.
    """
    from torch.utils._pytree import SUPPORTED_NODES, GetAttrKey, KeyEntry, register_pytree_node

    if cls in SUPPORTED_NODES:
        return
    field_names = [f.name for f in fields(cls)]

    def flatten(obj):
        return [getattr(obj, name) for name in field_names], None

    def flatten_with_keys(obj) -> tuple[list[tuple[KeyEntry, Any]], None]:
        # ``GetAttrKey`` is structurally assignable to the ``KeyEntry`` protocol
        # at runtime, but pyrefly treats list as invariant, so cast the list.
        keys = cast(
            "list[tuple[KeyEntry, Any]]",
            [(GetAttrKey(name), getattr(obj, name)) for name in field_names],
        )
        return keys, None

    def unflatten(values, context):
        return cls(**dict(zip(field_names, values, strict=True)))

    register_pytree_node(
        cls,
        flatten,
        unflatten,
        flatten_with_keys_fn=flatten_with_keys,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
    )


register_pytree_node_for_dataclass(CompressedBlockLayout)
register_pytree_node_for_dataclass(CompressedVarlenMetadata)
