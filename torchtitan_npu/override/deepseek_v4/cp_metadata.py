# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Packed document permutations and KV exchange plans."""

from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torchtitan.distributed.context_parallel import ContextParallelLoadBalancer
from torchtitan.distributed.spmd_types import spmd_mesh_size

from torchtitan_npu.models.common.alignment import compression_alignment
from torchtitan_npu.models.common.metadata import KernelFrame


class DocumentCPLoadBalancer(ContextParallelLoadBalancer):
    """Give each rank one chunk or one head/tail pair per packed document."""

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelLoadBalancer.Config):
        head_tail: bool = False

    def __init__(self, config: Config, *, seq_len, attention_metadata, cp_size=None):
        self.head_tail = config.head_tail
        self.cp_size = spmd_mesh_size("cp") if cp_size is None else cp_size
        cu_seqlens = attention_metadata.row.q.cu_seqlens
        self.bounds = cu_seqlens.tolist()
        self.device = cu_seqlens.device
        chunks = self.cp_size * (2 if self.head_tail else 1)
        if self.bounds[-1] != seq_len or any((end - start) % chunks for start, end in pairwise(self.bounds)):
            raise ValueError(f"Each packed document must split evenly into {chunks} CP chunks.")

    def generate_permutation(self):
        parts = self.cp_size * (2 if self.head_tail else 1)
        chunks = torch.cat(
            [
                torch.arange(start, end, device=self.device).reshape(parts, (end - start) // parts)
                for start, end in pairwise(self.bounds)
            ],
            dim=1,
        )
        if self.head_tail:
            chunks = torch.stack((chunks[: self.cp_size], chunks[self.cp_size :].flip(0)), dim=1)
        return chunks.reshape(1, -1)


def document_alignment(ratios, parallelism):
    """Keep every shard and compression group within its document."""
    layout = parallelism.context_parallel_load_balancer
    parts = parallelism.context_parallel_degree
    if parts > 1 and isinstance(layout, DocumentCPLoadBalancer.Config) and layout.head_tail:
        parts *= 2
    return parts * compression_alignment(ratios)


@dataclass(frozen=True, slots=True)
class FlattenExchange:
    send_index: torch.Tensor | None
    send_split: list[int]
    recv_split: list[int]
    recv_index: torch.Tensor


@dataclass(frozen=True, slots=True)
class WindowExchange:
    send_index: torch.Tensor
    send_split: list[int]
    recv_split: list[int]
    key_indices: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class DocumentCPMetadata:
    entries: tuple
    flat: FlattenExchange
    window: WindowExchange
    position_k: torch.Tensor
    compressor_cu_seqlens: torch.Tensor

    def query_parts(self, num_tokens):
        if len(self.entries) == 1:
            return (slice(None),)
        middle = num_tokens // 2
        return slice(None, middle), slice(middle, None)


def _token_ids(starts, ends, device):
    """Expand host-side document intervals without a dynamic device allocation."""
    lengths = ends - starts
    offsets = lengths.cumsum(0) - lengths
    rows = torch.arange(int(lengths.sum()), device="cpu")
    return (rows + torch.repeat_interleave(starts - offsets, lengths)).to(device)


@dataclass(slots=True)
class DocumentCPPlan:
    """One batch's layout, shared by all attention geometries."""

    cu_seqlens: torch.Tensor
    boundaries: torch.Tensor
    chunk_lengths: torch.Tensor
    query_starts: tuple[torch.Tensor, ...]
    locations: torch.Tensor
    owners: torch.Tensor
    cp_size: int
    cp_rank: int
    width: int
    flat: FlattenExchange
    _windows: dict[int, WindowExchange] = field(default_factory=dict)

    @classmethod
    def build(cls, permutation, cu_seqlens, *, cp_size, cp_rank, head_tail, boundaries=None):
        width = permutation.numel() // cp_size
        locations = torch.argsort(permutation.flatten())
        owners = locations // width
        if boundaries is None:
            boundaries = cu_seqlens.tolist()
        bounds = torch.tensor(boundaries, device="cpu")
        lengths = bounds.diff() // (2 * cp_size if head_tail else cp_size)
        ranks = torch.arange(cp_size, device="cpu")[:, None]
        starts = (bounds[:-1] + ranks * lengths,)
        if head_tail:
            starts += (bounds[1:] - (ranks + 1) * lengths,)

        slabs = torch.arange(permutation.numel(), device=permutation.device).split(width)
        send, send_split, recv_split, incoming = cls._route(slabs, owners, locations, cp_rank, width)
        flat = FlattenExchange(
            None if torch.equal(send, torch.arange(width, device=permutation.device)) else send,
            send_split,
            recv_split,
            torch.argsort(incoming),
        )
        return cls(cu_seqlens, bounds, lengths, starts, locations, owners, cp_size, cp_rank, width, flat)

    @staticmethod
    def _route(requests, owners, locations, rank, width):
        send = [tokens[owners[tokens] == rank] for tokens in requests]
        wanted = requests[rank]
        receive = [wanted[owners[wanted] == peer] for peer in range(len(requests))]
        return (
            locations[torch.cat(send)] % width,
            [rows.numel() for rows in send],
            [rows.numel() for rows in receive],
            torch.cat(receive),
        )

    def _window_starts(self, window_size):
        return tuple(torch.maximum(self.boundaries[:-1], start - window_size + 1) for start in self.query_starts)

    def _window_exchange(self, window_size):
        if window_size in self._windows:
            return self._windows[window_size]
        device = self.cu_seqlens.device
        window_starts = self._window_starts(window_size)
        windows = tuple(zip(window_starts, self.query_starts, strict=True))
        requests = []
        for peer in range(self.cp_size):
            prefixes = [_token_ids(lo[peer], hi[peer], device) for lo, hi in windows]
            tokens = torch.cat(prefixes).unique(sorted=True)
            requests.append(tokens[self.owners[tokens] != peer])
        send, send_split, recv_split, incoming = self._route(
            requests, self.owners, self.locations, self.cp_rank, self.width
        )
        key_slots = self.locations % self.width
        key_slots[incoming] = torch.arange(incoming.numel(), device=device) + self.width
        key_indices = tuple(
            key_slots[_token_ids(lo[self.cp_rank], hi[self.cp_rank] + self.chunk_lengths, device)] for lo, hi in windows
        )
        result = WindowExchange(send, send_split, recv_split, key_indices)
        self._windows[window_size] = result
        return result

    def prepare(self, row, *, window_size, overlap, build_entry):
        chunks = self.cp_size * len(self.query_starts)
        alignment = chunks * max(row.ratio, 1)
        if (self.boundaries.diff() % alignment).any():
            raise ValueError(f"Document CP with ratio {row.ratio} requires lengths divisible by {alignment}.")
        if row.position_k.ndim != 1:
            raise ValueError(f"Document CP expects a folded row of positions, got {tuple(row.position_k.shape)}.")

        rank = self.cp_rank
        device = self.cu_seqlens.device
        start = rank * self.width
        end = start + self.width
        carry_start = start - (row.ratio if overlap and rank else 0)
        internal_bounds = (bound - carry_start for bound in self.boundaries.tolist() if carry_start < bound < end)
        compressor_bounds = [0, *internal_bounds]
        compressor_bounds.append(end - carry_start)
        q = KernelFrame(
            cu_seqlens=self.cu_seqlens // chunks,
            seqused=self.chunk_lengths.to(device=device, dtype=torch.int32),
        )
        entries = []
        for window_lo, query_lo in zip(self._window_starts(window_size), self.query_starts, strict=True):
            query_end = query_lo[rank] + self.chunk_lengths
            window_lengths = (query_end - window_lo[rank]).to(device=device, dtype=torch.int32)
            swa_k = KernelFrame(
                cu_seqlens=torch.cat((window_lengths.new_zeros(1), window_lengths.cumsum(0, dtype=torch.int32))),
                seqused=window_lengths,
            )
            cmp_k = None
            if row.cmp_k is not None:
                reach = (query_end - self.boundaries[:-1]).to(device=device, dtype=torch.int32)
                cmp_k = KernelFrame(
                    cu_seqlens=row.cmp_k.cu_seqlens,
                    seqused=reach // row.ratio,
                    residual=None if row.ratio == 1 else reach % row.ratio,
                )
            entries.append(build_entry(q=q, swa_k=swa_k, cmp_k=cmp_k))
        return DocumentCPMetadata(
            tuple(entries),
            self.flat,
            self._window_exchange(window_size),
            row.position_k.narrow(0, start, self.width),
            torch.tensor(compressor_bounds, device=device, dtype=torch.int32),
        )
