# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context Parallel for DeepSeek-V4 Attention — ParallelStyle with hooks."""

from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.distributed._functional_collectives as ft_c
import torch.distributed.distributed_c10d as c10d
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ParallelStyle

from .registry import register_cp_strategy


def _assert_seq_not_sharded(placements, seq_dim: int, where: str) -> None:
    """Reject a DTensor sharded on the sequence dim: CP manipulates that dim on the
    local tensor and cannot preserve a ``Shard(seq)`` placement on re-wrap. Today
    the activation is ``Replicate``; fails loudly if a future TP plan shards it.
    """
    from torch.distributed.tensor import Shard

    for placement in placements:
        if isinstance(placement, Shard) and placement.dim == seq_dim:
            raise ValueError(
                f"{where}: input DTensor is sharded on the sequence dim "
                f"(Shard({seq_dim})); CP manipulates this dim on the local tensor "
                f"and cannot preserve that placement (placements={placements})."
            )


def _call_local_fn(
    tensor: torch.Tensor,
    seq_dim: int,
    where: str,
    local_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Run ``local_fn`` on the local tensor, adapting a TP DTensor input
    (``to_local -> local_fn -> from_local`` with the same mesh / placements). CP's
    c10d / view ops have no DTensor sharding strategy, so the real work stays on the
    local tensor; a ``Shard(seq)`` input is rejected (CP manipulates that dim and
    cannot preserve it on re-wrap). Under autograd, ``from_local``'s backward
    redistributes the gradient to the forward placements.
    """
    from torch.distributed.tensor import DTensor

    if not isinstance(tensor, DTensor):
        return local_fn(tensor)

    _assert_seq_not_sharded(tensor.placements, seq_dim, where)
    local = tensor.to_local(grad_placements=tensor.placements)
    out = local_fn(local)
    return DTensor.from_local(
        out,
        device_mesh=tensor.device_mesh,
        placements=tensor.placements,
        run_check=False,
    )


def _materialize_no_alias(tensor: torch.Tensor | None, seq_dim: int, where: str) -> torch.Tensor | None:
    if tensor is None:
        return None
    return _call_local_fn(tensor, seq_dim, where, lambda local: local.contiguous())


class _WindowExchangeLocal(torch.autograd.Function):
    """P2P exchange of a sequence window between adjacent CP ranks on local tensors:
    each rank sends its last ``window`` tokens to rank+1 and prepends those received
    from rank-1. DTensor (TP) adaptation is handled by :func:`_window_exchange`.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        tensor: torch.Tensor,
        window: int,
        group: c10d.ProcessGroup,
    ) -> torch.Tensor:
        rank = group.rank()
        world_size = group.size()

        send_buf = tensor[:, -window:].contiguous()
        recv_buf = torch.empty_like(send_buf) if rank > 0 else torch.empty(0, device=tensor.device)

        ctx.rank = rank
        ctx.world_size = world_size
        ctx.group = group
        ctx.window = window
        ctx.forward_sent = rank + 1 < world_size
        ctx.forward_recvd = rank > 0

        recv_req = None
        send_req = None

        if ctx.forward_recvd:
            recv_req = c10d.irecv(recv_buf, group_src=rank - 1, group=group)
        if ctx.forward_sent:
            send_req = c10d.isend(send_buf, group_dst=rank + 1, group=group)

        if recv_req is not None:
            recv_req.wait()
        if send_req is not None:
            send_req.wait()

        tensor = torch.cat([recv_buf, tensor], dim=1) if ctx.forward_recvd else tensor.clone()

        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        rank = ctx.rank
        window = ctx.window
        group = ctx.group

        recv_req = None
        send_req = None

        if ctx.forward_recvd:
            grad_send = grad_output[:, :window].contiguous()
            send_req = c10d.isend(grad_send, group_dst=rank - 1, group=group)

        if ctx.forward_sent:
            grad_recv = torch.empty_like(grad_output[:, :window])
            recv_req = c10d.irecv(grad_recv, group_src=rank + 1, group=group)

        if recv_req is not None:
            recv_req.wait()
        if send_req is not None:
            send_req.wait()

        if ctx.forward_sent:
            # pyrefly: ignore [unbound-name]
            grad_output[:, -window:] = grad_output[:, -window:] + grad_recv

        if ctx.forward_recvd:
            grad_output = grad_output[:, window:]

        return grad_output, None, None


def _window_exchange(tensor: torch.Tensor, window: int, group: c10d.ProcessGroup) -> torch.Tensor:
    return _call_local_fn(
        tensor,
        1,
        "_window_exchange",
        lambda local: _WindowExchangeLocal.apply(local, window, group),
    )


def _allgather_seq_local(local_tensor: torch.Tensor, mesh: DeviceMesh, seq_dim: int = 1) -> torch.Tensor:
    """All-gather a local tensor along the sequence dim across the CP mesh."""
    group = mesh.get_group()
    group_size = group.size()
    gathered = ft_c.all_gather_tensor_autograd(local_tensor.contiguous(), gather_dim=0, group=group)
    if isinstance(gathered, ft_c.AsyncCollectiveTensor):
        gathered = ft_c.wait_tensor(gathered)
    return torch.cat(torch.chunk(gathered, group_size, dim=0), dim=seq_dim)


def _allgather_seq(tensor: torch.Tensor, mesh: DeviceMesh, seq_dim: int = 1) -> torch.Tensor:
    return _call_local_fn(
        tensor,
        seq_dim,
        "_allgather_seq",
        lambda local: _allgather_seq_local(local, mesh, seq_dim),
    )


class CompressorAttentionCP(ParallelStyle):
    """Unified CP for DS-V4 Attention.

    W = max(compress_ratio, 128) covers all cases:
        SWA   (r=1):   W=128  (SWA window tokens only)
        C128A (r=128): W=128  (SWA window, no compressor overlap needed)
        C4A   (r=4):   W=128  (SWA window + compressor overlap)

    Args:
        compress_ratio: 1 (SWA), 4 (C4A), or 128 (C128A)
    """

    def __init__(self, compress_ratio: int) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.window = max(compress_ratio, 128)

    def _apply(self, module: torch.nn.Module, device_mesh: DeviceMesh) -> torch.nn.Module:
        module.register_forward_pre_hook(partial(self._pre_hook, mesh=device_mesh), with_kwargs=True)
        module.register_forward_hook(partial(self._post_hook, mesh=device_mesh))
        return module

    def _pre_hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        mesh: DeviceMesh,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """WindowExchange via P2P isend/irecv to prepend previous rank's
        tokens for compressor overlap and SWA."""
        x, freqs_cis, hadamard_mat = args
        local_s = x.size(1)
        rank = mesh.get_local_rank()
        W = self.window

        x = _window_exchange(x, W, mesh.get_group())

        # Compute positions for the expanded window (including prepended tokens).
        #   rank=0: [0, local_s)                            -- no prepended tokens
        #   rank>0: [rank*local_s - W, (rank+1)*local_s)    -- includes W prepended
        # NOTE: Current CP scheme does NOT support any load-balancing (e.g. head-tail rearrangement).
        start = max(0, rank * local_s - W) if rank > 0 else 0
        end = (rank + 1) * local_s
        kwargs["positions"] = torch.arange(start, end, dtype=torch.int32, device=x.device).unsqueeze(0)

        return (x, freqs_cis, hadamard_mat), kwargs

    def _post_hook(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        outputs: tuple[Any, ...],
        mesh: DeviceMesh,
    ) -> Any:
        """Strip extra tokens, allgather compressed tensors, causal slice
        compressed KV and k_indexer, left-zero-pad ori_kv."""
        q, kv, kv_compress, q_indexer, k_indexer, weights = outputs
        W = self.window
        R = self.compress_ratio
        ceil_div = (W + R - 1) // R
        rank = mesh.get_local_rank()

        if rank > 0:
            q = q[:, W:]
            if q_indexer is not None:
                q_indexer = q_indexer[:, W:]
            if weights is not None:
                weights = weights[:, W:]

            if kv_compress is not None:
                kv_compress = kv_compress[:, ceil_div:]
            if k_indexer is not None:
                k_indexer = k_indexer[:, ceil_div:]

        if kv_compress is not None:
            kv_compress = _allgather_seq(kv_compress, mesh)
        if k_indexer is not None:
            k_indexer = _allgather_seq(k_indexer, mesh)

        # NOTE: This can be optimized:
        # Remove slicing after NPU kernels support `seqused_k`
        # Remove padding after NPU kernels support `cmp_residual_k`
        local_s = q.size(1)
        slice_blocks = (rank + 1) * local_s // R
        target_ori_len = slice_blocks * R

        if kv_compress is not None:
            kv_compress = kv_compress[:, :slice_blocks]
        if k_indexer is not None:
            k_indexer = k_indexer[:, :slice_blocks]

        if kv.size(1) < target_ori_len:
            kv = torch.nn.functional.pad(kv, (0, 0) * (kv.ndim - 2) + (target_ori_len - kv.size(1), 0))

        materialized = (
            _materialize_no_alias(q, 1, "_post_hook.q"),
            _materialize_no_alias(kv, 1, "_post_hook.kv"),
            _materialize_no_alias(kv_compress, 1, "_post_hook.kv_compress"),
            _materialize_no_alias(q_indexer, 1, "_post_hook.q_indexer"),
            _materialize_no_alias(k_indexer, 1, "_post_hook.k_indexer"),
            _materialize_no_alias(weights, 1, "_post_hook.weights"),
        )
        return materialized


def _detect_dsv4(module: torch.nn.Module) -> bool:
    return hasattr(module, "compress_ratio") and hasattr(module, "pre_attention")


def _apply_dsv4(module: torch.nn.Module, cp_mesh: DeviceMesh) -> None:
    from torch.distributed.tensor.parallel import parallelize_module

    parallelize_module(
        # pyrefly: ignore [bad-argument-type]
        module.pre_attention,
        cp_mesh,
        # pyrefly: ignore [bad-argument-type]
        CompressorAttentionCP(compress_ratio=module.compress_ratio),
    )


register_cp_strategy(_detect_dsv4, _apply_dsv4)
