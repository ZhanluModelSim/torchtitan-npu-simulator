# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compile boundary for CANN Host lookup and sparse-gradient accumulation.

Communication and CPU work execute at runtime inside the custom ops. This is
torch.compile integration, not device-graph capture of the Host optimizer.
"""

from typing import TYPE_CHECKING, cast

import torch
from torch._library.effects import EffectType

from torchtitan_npu.models.deepseek_v4_1.engram import host

if TYPE_CHECKING:
    from .ascendc import HostOffloadEngramTable


def compiled_lookup(table: "HostOffloadEngramTable", row_ids: torch.Tensor) -> torch.Tensor:
    if not host._HAS_OPAQUE_OBJECT_API:
        raise RuntimeError("Compiled AscendC Host Engram requires the PyTorch opaque-object API.")
    return _lookup(
        table.weight,
        row_ids.reshape(-1),
        table._grad_keepalive,
        table._ep_size * table._efsdp_size,
        table.num_max_tokens_per_rank,
        table._lookup_dtype(),
        table._host_lookup_handle,
    )[0].view(*row_ids.shape, table.embedding_dim)


if host._HAS_OPAQUE_OBJECT_API:
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTableHandle

    @torch.library.custom_op("torchtitan_npu::ascendc_engram_lookup", mutates_args=())
    def _lookup(
        weight: torch.Tensor,
        row_ids: torch.Tensor,
        keepalive: torch.Tensor,
        rank_size: int,
        capacity: int,
        output_dtype: torch.dtype,
        table: HostEngramTableHandle,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rows, fetch = cast("HostOffloadEngramTable", table.value)._fetch_rows(row_ids)
        return rows, fetch.perm, fetch.send_counts, fetch.recv_counts, fetch.recv_local_entry, fetch.num_recv

    # Fetch is a read of the input weight and can be recomputed by AC, as in
    # eager mode. Each call returns its own backward metadata; each table has
    # a dedicated communicator. Only backward mutates accumulated Host state.

    @_lookup.register_fake
    def _lookup_fake(weight, row_ids, keepalive, rank_size, capacity, output_dtype, table):
        # CANN ElasticBuffer's EngramFetchCtx / engram_fetch_train meta kernel
        # define these as 1D int32 tensors on the indices device. send_counts
        # uses a 32-byte slot (8 int32s) per rank; received entries reserve
        # capacity rows per source rank. ElasticBuffer validates this layout
        # before FetchGrad. See ops-transformer df1d648, elastic_buffer.py.
        def metadata(size):
            return row_ids.new_empty((size,), dtype=torch.int32)

        return (
            row_ids.new_empty((row_ids.numel(), weight.shape[1]), dtype=output_dtype),
            metadata(row_ids.numel()),
            metadata(rank_size * 8),
            metadata(rank_size),
            metadata(capacity * rank_size),
            metadata(1),
        )

    def _setup_context(ctx, inputs, output):
        ctx.table = inputs[-1]
        ctx.save_for_backward(inputs[2], *output[1:])
        ctx.mark_non_differentiable(*output[1:])

    def _backward(ctx, grad_rows, *unused):
        keepalive, perm, send_counts, recv_counts, recv_local_entry, num_recv = ctx.saved_tensors
        anchor_grad = _lookup_backward(
            grad_rows, keepalive, perm, send_counts, recv_counts, recv_local_entry, num_recv, ctx.table
        )
        # The returned gradient also keeps sparse accumulation connected to
        # autograd: the CPU table receives its sparse gradient out of band.
        return None, None, anchor_grad, None, None, None, None

    torch.library.register_autograd(_lookup, _backward, setup_context=_setup_context)

    @torch.library.custom_op("torchtitan_npu::ascendc_engram_lookup_backward", mutates_args=())
    def _lookup_backward(
        grad_rows: torch.Tensor,
        keepalive: torch.Tensor,
        perm: torch.Tensor,
        send_counts: torch.Tensor,
        recv_counts: torch.Tensor,
        recv_local_entry: torch.Tensor,
        num_recv: torch.Tensor,
        table: HostEngramTableHandle,
    ) -> torch.Tensor:
        from cann_ops_transformer.ops.mc2.common.elastic_buffer import EngramFetchCtx

        fetch = EngramFetchCtx(
            perm=perm,
            send_counts=send_counts,
            recv_counts=recv_counts,
            recv_local_entry=recv_local_entry,
            num_recv=num_recv,
        )
        cast("HostOffloadEngramTable", table.value)._accumulate_fetch_gradient(grad_rows, fetch)
        return torch.zeros_like(keepalive)

    _lookup_backward.register_effect(EffectType.ORDERED)

    @_lookup_backward.register_fake
    def _lookup_backward_fake(grad_rows, keepalive, perm, send_counts, recv_counts, recv_local_entry, num_recv, table):
        return torch.zeros_like(keepalive)
