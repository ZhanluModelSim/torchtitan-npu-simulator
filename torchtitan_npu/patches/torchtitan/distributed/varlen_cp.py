# Pending upstream PR: https://github.com/pytorch/torchtitan/pull/3430

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context-parallel metadata for variable-length attention.

This mirrors ``torchtitan/distributed/varlen_cp.py`` from upstream PR #3430.
"""

from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._context_parallel._load_balancer import (
    _LoadBalancer,
)
from torchtitan.models.common.attention import VarlenMetadata


@dataclass(frozen=True, eq=False)
class CPVarlenMetadata:
    """Rank-local varlen metadata with K/V compaction indices.

    ``k_global_gather_indices`` selects visible regions after the CP all-gather
    and reverses the load-balancer permutation when present.
    """

    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int
    k_global_gather_indices: torch.Tensor

    @classmethod
    def from_global(
        cls,
        global_metadata: VarlenMetadata,
        device_mesh: DeviceMesh,
        batch_size: int,
        seq_length: int,
        load_balancer: _LoadBalancer | None = None,
    ) -> "CPVarlenMetadata":
        """Build rank-local metadata from global self-attention boundaries.

        Q is sharded and K/V are assumed to be all-gathered. Each contiguous
        Q run becomes a causal segment. Construction uses one host transfer.
        """
        if isinstance(global_metadata, CPVarlenMetadata):
            raise ValueError(
                "from_global received a CPVarlenMetadata; pass the "
                "unsharded global VarlenMetadata instead."
            )
        # Identity avoids a device-to-host equality check on every forward.
        if global_metadata.cu_seq_q is not global_metadata.cu_seq_k:
            raise ValueError(
                "CP varlen sharding currently supports only self-attention; "
                "cu_seq_q and cu_seq_k must be the same tensor object."
            )
        cu_seq_q_global = global_metadata.cu_seq_q

        if device_mesh.ndim != 1:
            raise ValueError(
                f"CPVarlenMetadata.from_global expects a 1-D CP mesh, "
                f"got ndim={device_mesh.ndim}."
            )
        cp_world_size = device_mesh.size()
        cp_rank = device_mesh.get_local_rank()
        required_divisor = (
            2 * cp_world_size if load_balancer is not None else cp_world_size
        )
        if seq_length % required_divisor != 0:
            reason = (
                "2 * cp world size (load balancers chunk each shard into 2 halves)"
                if load_balancer is not None
                else "cp world size"
            )
            raise ValueError(
                f"seq_length {seq_length} must be divisible by {required_divisor} "
                f"({reason}); got cp world size {cp_world_size}."
            )
        shard_len = seq_length // cp_world_size
        device = cu_seq_q_global.device
        dtype = cu_seq_q_global.dtype

        # Map each batch slot to its pre-balancing token index.
        tok_indices_per_batch: torch.Tensor
        restore_per_batch: torch.Tensor | None = None
        if load_balancer is None:
            tok_indices_per_batch = (
                torch.arange(seq_length, device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
        else:
            rearrange_indices = load_balancer._generate_indices(restore=False)
            if rearrange_indices is None:
                raise ValueError(
                    "load_balancer._generate_indices() returned None; "
                    "a load balancer must return a tensor."
                )
            if rearrange_indices.ndim != 2 or rearrange_indices.shape[0] not in (
                1,
                batch_size,
            ):
                raise ValueError(
                    "load balancer indices must have shape (1, seq_length) "
                    "or (batch_size, seq_length); got "
                    f"{tuple(rearrange_indices.shape)}."
                )
            rearrange_indices = rearrange_indices.to(dtype)
            if rearrange_indices.shape[0] == 1:
                tok_indices_per_batch = rearrange_indices.expand(batch_size, -1)
                restore_per_batch = torch.argsort(rearrange_indices, dim=-1).expand(
                    batch_size, -1
                )
            else:
                tok_indices_per_batch = rearrange_indices
                restore_per_batch = torch.argsort(rearrange_indices, dim=-1)

        # Map rank-local Q slots to sequence positions.
        rank_q_indices = tok_indices_per_batch[
            :, cp_rank * shard_len : (cp_rank + 1) * shard_len
        ]

        # Convert sequence positions to the row-major packed layout.
        batch_offsets = (
            torch.arange(batch_size, device=device, dtype=dtype).unsqueeze(1)
            * seq_length
        )
        packed_local_to_global = (batch_offsets + rank_q_indices).reshape(-1)
        total_local = batch_size * shard_len

        doc_id = (
            torch.searchsorted(
                cu_seq_q_global,
                packed_local_to_global,
                right=True,
                out_int32=(dtype == torch.int32),
            )
            - 1
        )

        # Split on document changes or non-contiguous packed positions.
        diff_doc = doc_id[1:] != doc_id[:-1]
        diff_global = packed_local_to_global[1:] != packed_local_to_global[:-1] + 1
        is_break = diff_doc | diff_global
        seg_starts_inner = (is_break.nonzero(as_tuple=False).squeeze(-1) + 1).to(dtype)
        seg_starts = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), seg_starts_inner]
        )
        seg_ends = torch.cat(
            [seg_starts[1:], torch.tensor([total_local], dtype=dtype, device=device)]
        )

        seqlen_q = seg_ends - seg_starts
        # seqlen_k = (last global pos in segment) - (doc global start) + 1.
        last_local_idx = seg_ends - 1
        last_global = packed_local_to_global[last_local_idx]
        seg_doc_id = doc_id[seg_starts]
        doc_global_start = cu_seq_q_global[seg_doc_id]
        seqlen_k = last_global - doc_global_start + 1

        cu_seq_q = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), seqlen_q.cumsum(0).to(dtype)]
        )
        cu_seq_k = torch.cat(
            [torch.zeros(1, dtype=dtype, device=device), seqlen_k.cumsum(0).to(dtype)]
        )

        # Read all host scalars in one device-to-host transfer.
        max_q, max_k, total_k = (
            torch.stack([seqlen_q.max(), seqlen_k.max(), seqlen_k.sum()]).cpu().tolist()
        )

        # Gather each segment's K range from its document start.
        bases = torch.repeat_interleave(doc_global_start, seqlen_k)
        seg_starts_repeated = torch.repeat_interleave(cu_seq_k[:-1], seqlen_k)
        within_seg = (
            torch.arange(total_k, device=device, dtype=dtype) - seg_starts_repeated
        )
        k_global_gather_indices = bases + within_seg

        # Compose with the inverse load-balancer permutation.
        if restore_per_batch is not None:
            batch_id = k_global_gather_indices // seq_length
            pos_in_batch = k_global_gather_indices % seq_length
            restored_pos = restore_per_batch[batch_id, pos_in_batch]
            k_global_gather_indices = batch_id * seq_length + restored_pos

        return cls(
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            k_global_gather_indices=k_global_gather_indices.to(torch.long),
        )
