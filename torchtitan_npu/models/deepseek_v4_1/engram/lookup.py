# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Torch Host lookup with optional EP All-to-All routing.

The table is padded so that every rank in ``group`` owns the same number of
contiguous rows. Forward sends deduplicated row requests to their owners and
returns the gathered rows. Backward reverses that exchange and accumulates owner-local sparse gradients
in the Host table. The complete table gradient is never materialized.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _lookup_rows(
    ctx,
    weight: torch.Tensor,
    row_ids: torch.Tensor,
    group,
):
    rows_per_rank, row_dim = weight.shape
    world_size = dist.get_world_size(group)

    flat_ids = row_ids.reshape(-1)
    unique_ids, inverse = torch.unique(
        flat_ids,
        sorted=True,
        return_inverse=True,
    )
    owners = torch.div(unique_ids, rows_per_rank, rounding_mode="floor")
    if unique_ids.numel() and int(owners[-1]) >= world_size:
        raise IndexError(
            f"Engram row ID {int(unique_ids[-1])} exceeds the padded table "
            f"capacity ({rows_per_rank * world_size} rows)."
        )
    local_ids = unique_ids - owners * rows_per_rank
    send_counts = torch.bincount(owners, minlength=world_size)

    recv_counts = torch.empty_like(send_counts)
    dist.all_to_all_single(recv_counts, send_counts, group=group)
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()

    received_ids = torch.empty(
        int(recv_counts.sum()),
        dtype=torch.long,
        device=row_ids.device,
    )
    dist.all_to_all_single(
        received_ids,
        local_ids.contiguous(),
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=group,
    )

    owned_rows = weight.index_select(0, received_ids.to(device=weight.device)).to(device=row_ids.device)
    unique_rows = torch.empty(
        int(send_counts.sum()),
        row_dim,
        dtype=weight.dtype,
        device=row_ids.device,
    )
    dist.all_to_all_single(
        unique_rows,
        owned_rows,
        output_split_sizes=send_splits,
        input_split_sizes=recv_splits,
        group=group,
    )

    ctx.save_for_backward(inverse, received_ids)
    ctx.send_splits = send_splits
    ctx.recv_splits = recv_splits
    ctx.group = group
    return unique_rows.index_select(0, inverse).view(*row_ids.shape, row_dim)


def _lookup_grad_rows(ctx, grad_output: torch.Tensor):  # pyrefly: ignore [bad-override]
    inverse, received_ids = ctx.saved_tensors
    row_dim = grad_output.shape[-1]

    # Aggregate duplicate hits in fp32 before communication. This keeps the
    # A2A payload proportional to the number of unique rows and avoids a
    # low-precision local reduction.
    grad_unique = torch.zeros(
        sum(ctx.send_splits),
        row_dim,
        dtype=torch.float32,
        device=grad_output.device,
    )
    grad_unique.index_add_(0, inverse, grad_output.reshape(-1, row_dim).float())

    received_grads = torch.empty(
        sum(ctx.recv_splits),
        row_dim,
        dtype=torch.float32,
        device=grad_output.device,
    )
    dist.all_to_all_single(
        received_grads,
        grad_unique.contiguous(),
        output_split_sizes=ctx.recv_splits,
        input_split_sizes=ctx.send_splits,
        group=ctx.group,
    )

    return received_grads, received_ids


class HostEngramLookup(torch.autograd.Function):
    """Use device collectives and CPU row access with owner-local sparse grads.

    The sparse gradient is attached immediately before SparseAdam, just as for
    the fused backend. The anchor keeps this autograd node live without exposing
    an ordinary CPU gradient to the trainer's DTensor gradient clipper.
    """

    @staticmethod
    def forward(ctx, weight, row_ids, table, keepalive):  # pyrefly: ignore [bad-override]
        ctx.table = table
        ctx.anchor_device = keepalive.device
        ctx.anchor_dtype = keepalive.dtype
        ctx.distributed = table.lookup_mesh is not None and table.lookup_mesh.size() > 1
        if ctx.distributed:
            return _lookup_rows(ctx, weight, row_ids, table.lookup_mesh.get_group())
        local_ids = row_ids.reshape(-1).to(device="cpu")
        ctx.save_for_backward(local_ids)
        return weight.index_select(0, local_ids).to(device=row_ids.device).view(*row_ids.shape, weight.shape[1])

    @staticmethod
    def backward(ctx, grad_output):  # pyrefly: ignore [bad-override]
        if ctx.distributed:
            grad_rows, local_ids = _lookup_grad_rows(ctx, grad_output)
        else:
            (local_ids,) = ctx.saved_tensors
            grad_rows = grad_output.reshape(-1, grad_output.shape[-1])
        ctx.table.accumulate_sparse_gradient(local_ids, grad_rows)
        return None, None, None, torch.zeros((), device=ctx.anchor_device, dtype=ctx.anchor_dtype)


__all__ = ["HostEngramLookup"]
