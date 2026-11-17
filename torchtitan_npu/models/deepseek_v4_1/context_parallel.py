# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The context-parallel query plan of one packed row, and the exchange that runs it.

Shape legend: ``R = cp_size``; ``r = rank``; ``N`` the row's tokens, ``C = N // R`` the slab
width; ``L_d`` document ``d``'s length and ``Q_d = L_d // R`` its chunk.

Rank ``r`` holds the plain slab ``[r·C, (r+1)·C)`` of the row, and the queries move onto
**chunk ``r`` of every document**: one chunk of every document is what makes the per-document
reach ``K_d = (r+1)·Q_d`` the same kind of problem on every rank.

The KV side is **not** moved.  It is gathered declaratively at the kernel boundary (``cp: S(1)
-> R`` on the fused core's inputs) and ``seqused = K_d`` is what limits a rank's view, which is
also what keeps the kernels' end-aligned reading -- the sliding window and the compressed
causal limit are anchored at this rank's reach instead of at the row's end -- so the KV streams
stay the whole row on every rank.

``cp_plan`` derives the movement from the global document boundaries alone: pure, on host
values, zero communication.  ``QueryExchange`` is the module that runs it, owning the process
group that ``parallelize`` wires, and :class:`ExchangeMetadata` is one batch's tables, carried
on the attention metadata so no consumer needs a new argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_to_all_single
from torchtitan.protocols.module import Module

__all__ = [
    "ExchangeMetadata",
    "QueryExchange",
    "cp_plan",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class ExchangeMetadata:
    """One batch's query plan: the movement, and the positions that come with it."""

    send_index: torch.Tensor
    """Reorder of this rank's slab (int64, ``[C]``): the rows addressed to peer ``i`` form one
    contiguous slice, peers in order and ascending slot order within a peer."""

    send_split: list[int]
    """``input_split_sizes``: rows this rank addresses to peer ``i``, one entry per peer.  A list,
    because that is what the collective takes -- the plan builds it as one."""

    recv_split: list[int]
    """``output_split_sizes``: rows this rank receives from peer ``i``.  The receiver's buffer
    is already in held-row order -- sources ascend and each source's rows ascend with them --
    so there is no scatter index and no second reorder."""

    positions_q: torch.Tensor
    """The held rows' own document-relative positions (int64, ``[C]``): the positions the
    moved rows rotate at, not the ones the tensor they came from was addressed by.  The
    metadata exposes them shaped like the input's own positions, so no consumer reshapes."""


class QueryExchange(Module):
    """The query side's movement: the slab frame to this rank's chunks, and back.

    The process group is the module's only state, wired once by the framework's
    ``Module.parallelize`` recursion (the V4 ``token_dispatcher`` mirror); everything a
    particular batch decides arrives as an :class:`ExchangeMetadata`, so the same instance
    serves every forward and every layer.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """No fields: the movement is the same on every layer, the tables are the batch's."""

    def __init__(self, config: Config):
        super().__init__()
        self.group = None

    def parallelize(self, parallel_dims) -> None:
        """Take the CP mesh the framework hands down to every module."""
        self.group = parallel_dims.get_optional_mesh("cp")
        super().parallelize(parallel_dims)

    def permute(self, tensor: torch.Tensor, metadata: ExchangeMetadata | None) -> torch.Tensor:
        """Move ``tensor``'s rows from the slab frame onto this rank's chunks.

        The row axis is dim 0: a caller holding ``[1, C, ...]`` squeezes the batch axis and puts
        it back, because the row axis is the one the exchange addresses.

        Without a plan there is no movement to make -- the forward is unsharded, and the slab
        *is* the frame -- so the tensor comes back unchanged.  That case returns before the group
        is consulted, which is what lets one forward path serve both degrees without a branch at
        the call site and without a process group at ``cp = 1``.
        """
        if metadata is None:
            return tensor
        if self.group is None:
            raise RuntimeError(
                "the query exchange has no process group; QueryExchange.parallelize must run before a sharded forward"
            )
        return all_to_all_single(
            tensor[metadata.send_index],
            output_split_sizes=metadata.recv_split,
            input_split_sizes=metadata.send_split,
            group=self.group,
        )

    def unpermute(self, tensor: torch.Tensor, metadata: ExchangeMetadata | None) -> torch.Tensor:
        """Its inverse: the chunks' rows back onto the slab.

        A permutation, so the adjoint *is* the inverse: the reply is scattered through
        ``send_index`` and nothing accumulates.  Without a plan this is the identity too, for the
        same reason as :meth:`permute`.
        """
        if metadata is None:
            return tensor
        if self.group is None:
            raise RuntimeError(
                "the query exchange has no process group; QueryExchange.parallelize must run before a sharded forward"
            )
        received = all_to_all_single(
            tensor,
            output_split_sizes=metadata.send_split,
            input_split_sizes=metadata.recv_split,
            group=self.group,
        )
        out = torch.empty_like(received)
        out[metadata.send_index] = received
        return out


def cp_plan(cu_seqlens: torch.Tensor, cp_size: int, rank: int, seq_len: int) -> ExchangeMetadata:
    """Derive one rank's plan, locally, from the global document boundaries.

    ``seq_len`` is the slab width ``C``.  Every document length must be a positive multiple of
    ``cp_size`` so that a chunk is a whole number of rows; it is checked here because a
    violation would otherwise come back as a silently wrong split rather than an error.
    """
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths % cp_size).any()) or bool((lengths <= 0).any()):
        raise ValueError(f"every document must be a positive multiple of cp_size={cp_size}, got {lengths.tolist()}")
    chunk = lengths // cp_size
    device = cu_seqlens.device

    # Send side: where each row of this rank's slab goes.  ``dst`` is needed for the sort
    # anyway, and its histogram is the split.
    pos = torch.arange(rank * seq_len, (rank + 1) * seq_len, device=device)
    doc = torch.bucketize(pos, cu_seqlens, right=True) - 1
    dst = (pos - cu_seqlens[doc]) // chunk[doc]
    send_index = torch.argsort(dst, stable=True)
    send_split = torch.bincount(dst, minlength=cp_size)

    # Receive side, locally: each document's chunk ``rank`` is one global interval and spans
    # at most two slabs (``chunk <= seq_len``), so it contributes a first segment to the slab
    # it starts in and -- when it crosses -- the remainder to the next one.  The clamp is the
    # crossing test, so no mask and no second pass are needed.
    start = cu_seqlens[:-1] + rank * chunk
    edge = (start // seq_len + 1) * seq_len
    recv_split = torch.bincount(
        torch.cat((start // seq_len, (start // seq_len + 1).clamp(max=cp_size - 1))),
        weights=torch.cat((torch.minimum(start + chunk, edge) - start, (start + chunk - edge).clamp(min=0))).float(),
        minlength=cp_size,
    ).long()

    # The held rows' own positions: the slot index plus (this rank's chunk start minus where
    # that document's block starts in the buffer), since RoPE restarts at every document.
    positions_q = torch.arange(seq_len, device=device) + torch.repeat_interleave(
        rank * chunk - (cu_seqlens // cp_size)[:-1], chunk
    )
    return ExchangeMetadata(
        send_index=send_index,
        send_split=send_split.tolist(),
        recv_split=recv_split.tolist(),
        positions_q=positions_q,
    )
