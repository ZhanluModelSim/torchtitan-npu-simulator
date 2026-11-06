# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The ratio-4 overlap's cross-rank block: an autograd-capable ``batch_send_recv``.

Under context parallelism a rank's slab usually starts inside a document, and the ratio-4
compressor's first local block pools tokens from the predecessor rank's last block.  This
module rings that block over -- the raw tokens -- and returns the augmented stream together with
the document grid clipped to the augmented window and the number of output blocks the carry owns.
It is the *only* part of the stack that knows the compressor's overlap continues on another rank:
the compressor itself just compresses ``(x, positions, cu_seqlens)`` at every degree.

The prepended block's positions are **computed**, not exchanged: they are the predecessor slab's
last block, which the row's own boundaries already describe (``slab_positions``).  They are also
numerically inert -- the carry block is always trimmed out of the pooled stream, and the overlap
reads the *projected* a-series before RoPE -- so the exchange carries no second collective for
them.

Like :meth:`~torchtitan_npu.models.common.restore.RestorePlan.apply`, the process group is
the module's only state (wired by the framework's ``Module.parallelize`` recursion) and the
collectives are functional, so the backward of the prepended block is the reverse exchange.
At one degree the module is a no-op: it hands its inputs back, and the caller runs the very
same compressor instance it runs under CP.

The augmented boundaries come from the row's grid clipped to ``[slab_start - ratio, slab_end)``:
when the slab starts inside a document no boundary falls at the carry/slab seam, so the carry
completes that document's first local segment; when it starts at a document edge the row's own
boundary is kept and the carry is a segment of its own.  "Mask every segment start" is the same
rule in both cases, and the carry's output blocks are dropped by the caller, so nothing
special-cases the carry.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributed._functional_collectives import all_to_all_single
from torchtitan.protocols.module import Module

from torchtitan_npu.models.common.restore import doc_relative_positions

__all__ = ["CarryExchange", "CarryStream"]


@dataclass(frozen=True, slots=True)
class CarryStream:
    """One rank's stream after the carry: what to compress, and what to trim afterwards."""

    x: torch.Tensor
    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    trim: int | torch.SymInt
    """Compressed blocks at the head of the output that belong to the carry (0 or 1): zero on the
    rank that starts the row, one where a block was prepended, so a caller narrows by it rather
    than branching on it."""


def _ring_split(peer: int, rows: int, cp_size: int) -> list[int]:
    """``rows`` addressed to ``peer`` and to nobody else (one entry per peer, empty out of range).

    The ring's split lists are host values: the two neighbours are the only peers a rank ever
    talks to, and a neighbour that does not exist (the row's first and last rank) simply gets an
    empty split.  Every rank still enters both collectives, so the ring cannot deadlock on them.
    """
    return [rows if index == peer else 0 for index in range(cp_size)]


class CarryExchange(Module):
    """The predecessor's last block, prepended to this rank's slab (V4's ratio-4 overlap)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ratio: int
        """The layer's compression ratio; only a ratio-4 geometry has an overlapping block."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ratio = config.ratio
        self.group = None

    def _parallelize(self, parallelism_context) -> None:
        """Take the CP mesh the framework hands down to every module."""
        self.group = parallelism_context.get_optional_mesh("cp")
        super()._parallelize(parallelism_context)

    @property
    def active(self) -> bool:
        """Whether this run shards the sequence: at one degree the exchange is a no-op."""
        return self.group is not None and self.group.size() > 1

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> CarryStream:
        """This rank's augmented stream (``cu_seqlens`` are the **row's** boundaries)."""
        group = self.group
        if group is None or group.size() <= 1:
            return CarryStream(x=x, positions=positions, cu_seqlens=cu_seqlens, trim=0)
        if x.size(0) > 1:
            raise ValueError("CP>1 暂不支持 local_batch_size>1")
        ratio = self.ratio
        cp_size = group.size()
        slab_len = x.size(1)
        # This rank is a host value of the run, so the ring's two ends are stated as the split
        # lists the collective takes -- one host list per direction, derived from the rank
        # itself -- instead of a device mask read back with ``tolist()``, which would drain the
        # pipeline twice per layer.
        rank = group.get_local_rank()
        carried_rows = 0
        if ratio == 4:
            # Every rank enters the ring, and every rank prepends what it received: on rank 0
            # that is an empty block, but the concat keeps the collective's output in the graph.
            # Dropping the empty case instead would prune the collective's *backward* on rank 0
            # only, and the rank would then wait for a gradient exchange its peer still issues.
            x_carry = self._exchange(
                x,
                group=group,
                send_split=_ring_split(rank + 1, ratio, cp_size),
                recv_split=_ring_split(rank - 1, ratio, cp_size),
            )
            # The block's positions are the predecessor slab's last block: the row's boundaries
            # say where it sits, so nothing has to travel with the tokens.  The slice is by the
            # rows actually received, so rank 0 (an empty block) needs no branch of its own --
            # the rank is a symbolic value under the single-process simulator, and a python
            # comparison on it would have to guard.
            flat = rank * slab_len - ratio + torch.arange(ratio, device=cu_seqlens.device)
            positions_carry = doc_relative_positions(cu_seqlens, flat)[None, :][:, : x_carry.size(1)].to(
                positions.dtype
            )
            x = torch.cat([x_carry, x], dim=1)
            positions = torch.cat([positions_carry, positions], dim=1)
            carried_rows = x_carry.size(1)
        # The window starts one block earlier when the carry completed it (never on rank 0, whose
        # block is empty) and is that much wider, so the grid is clipped to what this rank holds.
        bounds = self._bounds(
            cu_seqlens,
            first=rank * slab_len - carried_rows,
            width=slab_len + carried_rows,
        )
        return CarryStream(x=x, positions=positions, cu_seqlens=bounds, trim=carried_rows // ratio)

    def _exchange(
        self,
        x: torch.Tensor,
        *,
        group,
        send_split: list[int],
        recv_split: list[int],
    ) -> torch.Tensor:
        """Ring the predecessor's last block over, empty on the rank that starts the row.

        Every rank enters the collective with an explicit split (the ranks that have nothing to
        send or receive pass an empty one), so the ring cannot deadlock on it.  The block is cut
        down to the rows this rank's own split covers, which is what lets the end of the ring and
        its middle run the same call.
        """
        ratio = self.ratio
        send_rows = sum(send_split)
        # ``all_to_all_single`` splits **dim 0**, so the batch axis has to come off: the ring
        # moves one block of rows, not one row of blocks.
        send = x[:, -ratio:, :].flatten(0, 1).narrow(0, 0, send_rows).contiguous()
        received = all_to_all_single(
            send,
            output_split_sizes=recv_split,
            input_split_sizes=send_split,
            group=group,
        )
        return received.unsqueeze(0)

    def _bounds(self, cu_seqlens: torch.Tensor, *, first: int, width: int) -> torch.Tensor:
        """The row's document boundaries clipped to ``[first, first + width)``, rebased onto it.

        Device arithmetic on the row's grid -- the clip is a mask and the rebase a subtraction --
        so a window that starts inside a document and one that starts on a boundary are the same
        call, and no degree reads the grid back to the host.
        """
        inside = cu_seqlens[(cu_seqlens > first) & (cu_seqlens < first + width)] - first
        return torch.cat((inside.new_zeros(1), inside, inside.new_full((1,), width))).to(torch.int32)
