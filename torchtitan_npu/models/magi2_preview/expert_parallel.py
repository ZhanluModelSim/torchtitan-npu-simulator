# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Head-parallel MoE sharding utilities for MAGI-2-preview.

MAGI-2's routed MoE routes per head (``CoreMultiHeadMoE``), so expert
parallelism shards the HEAD axis: the fused expert tensors have leading dim
``H * E`` with head-major rows (head ``h`` owns rows ``[h * E, (h + 1) * E)``),
and every parallel rank owns whole heads. Two communication regimes exist,
mirroring the official inference model
(``magi2-preview/inference/model/magi2_preview.py``):

* Regime (a), tokens replicated (no CP): each rank routes and computes its
  local heads over all tokens and the zero-padded partial outputs are
  all-reduced over the expert mesh (see :func:`pad_head_partial` and
  :func:`all_reduce_head_parallel_output`). This is the regime wired by
  ``parallelize._apply_moe_parallel`` today.
* Regime (b), tokens sequence-sharded (CP enabled): a Ulysses-style
  all-to-all swaps the sequence and head axes so each rank sees the full
  sequence for its heads, then swaps back. :func:`ep_dispatch` /
  :func:`ep_undispatch` port the official ``ep_dispatch`` / ``ep_undispatch``
  (lines 3277-3330 of the reference file, used by
  ``CoreMultiHeadMoE._forward_impl`` at lines 2624-2655) with autograd
  support. They are currently function-level primitives: wiring them needs
  sequence-sharded tokens, which arrives with context parallelism.
"""

import torch
import torch.distributed as dist

from .feed_forward import CoreMultiHeadMoE

__all__ = [
    "EXPERT_PARAM_NAMES",
    "ROUTER_BUFFER_NAMES",
    "head_range_for_rank",
    "slice_expert_state_by_head",
    "shard_moe_core_by_head",
    "pad_head_partial",
    "all_reduce_head_parallel_output",
    "all_reduce_head_parallel_input_grad",
    "head_seq_dispatch_permute",
    "head_seq_undispatch_permute",
    "ep_dispatch",
    "ep_undispatch",
]

# Fused expert params of CoreMultiHeadMoE, all shaped (H * E, ...) leading.
EXPERT_PARAM_NAMES = ("gate", "W_gate", "W_up", "W_down")
# Router buffers sharing the same leading H * E dim.
ROUTER_BUFFER_NAMES = ("expert_bias", "expert_bias_ema")


def head_range_for_rank(
    rank: int, degree: int, num_heads: int
) -> tuple[int, int]:
    """Contiguous global-head range owned by ``rank`` of an ``degree``-way
    head-parallel split; raises unless ``degree`` divides ``num_heads``."""
    if num_heads % degree != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by the "
            f"head-parallel degree={degree}"
        )
    heads_per_rank = num_heads // degree
    return rank * heads_per_rank, (rank + 1) * heads_per_rank


def slice_expert_state_by_head(
    state: dict[str, torch.Tensor],
    head_range: tuple[int, int],
    num_experts: int,
) -> dict[str, torch.Tensor]:
    """Head-granular leading-dim slice of the fused expert tensors.

    ``state`` is a mapping keyed relative to a ``CoreMultiHeadMoE`` module
    (``gate``, ``W_gate``, ``W_up``, ``W_down``, ``router.expert_bias``,
    ``router.expert_bias_ema``); the expert entries are narrowed from the
    full ``H * E`` leading dim to the rows of ``head_range`` and every
    other entry is passed through unchanged.
    """
    head_start, head_end = head_range
    row_start = head_start * num_experts
    row_end = head_end * num_experts
    expert_keys = set(EXPERT_PARAM_NAMES) | {
        f"router.{name}" for name in ROUTER_BUFFER_NAMES
    }
    sliced = dict(state)
    for key, tensor in state.items():
        if key not in expert_keys:
            continue
        if tensor.shape[0] % num_experts != 0:
            raise ValueError(
                f"expert tensor {key} leading dim {tensor.shape[0]} is not a "
                f"multiple of num_experts={num_experts}"
            )
        sliced[key] = tensor[row_start:row_end].contiguous()
    return sliced


def shard_moe_core_by_head(
    core: CoreMultiHeadMoE, head_range: tuple[int, int]
) -> None:
    """Turn a full-head ``CoreMultiHeadMoE`` into its head-range shard.

    Replaces the expert params and router buffers with their head-granular
    slices in place and sets ``core.head_range``; used to materialize a
    single rank's view of a head-parallel split (e.g. in tests).
    """
    sliced = slice_expert_state_by_head(
        core.state_dict(), head_range, core.num_experts
    )
    for param_name in EXPERT_PARAM_NAMES:
        param = getattr(core, param_name)
        core.register_parameter(
            param_name,
            torch.nn.Parameter(
                sliced[param_name], requires_grad=param.requires_grad
            ),
        )
    for buffer_name in ROUTER_BUFFER_NAMES:
        core.router.register_buffer(buffer_name, sliced[f"router.{buffer_name}"])
    core.set_head_range(head_range)


class _PadHeadPartial(torch.autograd.Function):
    """Zero-pad a local-head partial MoE output to the full hidden width."""

    @staticmethod
    def forward(ctx, partial, head_start, head_end, num_heads):
        ctx.head_start = head_start
        ctx.head_end = head_end
        ctx.num_heads = num_heads
        tokens = partial.shape[0]
        local_num_heads = head_end - head_start
        ctx.d_head = partial.shape[-1] // local_num_heads
        full = partial.new_zeros(tokens, num_heads, ctx.d_head)
        full[:, head_start:head_end] = partial.view(
            tokens, local_num_heads, ctx.d_head
        )
        return full.reshape(tokens, num_heads * ctx.d_head)

    @staticmethod
    def backward(ctx, grad_full):
        grad_partial = grad_full.view(
            grad_full.shape[0], ctx.num_heads, ctx.d_head
        )[:, ctx.head_start : ctx.head_end]
        return (
            grad_partial.reshape(grad_full.shape[0], -1),
            None,
            None,
            None,
        )


def pad_head_partial(
    partial: torch.Tensor, head_range: tuple[int, int], num_heads: int
) -> torch.Tensor:
    """Place a ``(T, local_heads * d_head)`` partial output into a zero
    ``(T, num_heads * d_head)`` tensor at its global head columns.

    The backward slices the upstream gradient back to the owned head
    columns, matching the sum-assembly of regime (a).
    """
    head_start, head_end = head_range
    return _PadHeadPartial.apply(partial, head_start, head_end, num_heads)


class _AllReduceHeadParallel(torch.autograd.Function):
    """Zero-pad + all-reduce forward; plain column-slice backward."""

    @staticmethod
    def forward(ctx, partial, head_start, head_end, num_heads, group):
        ctx.head_start = head_start
        ctx.head_end = head_end
        ctx.num_heads = num_heads
        tokens = partial.shape[0]
        local_num_heads = head_end - head_start
        ctx.d_head = partial.shape[-1] // local_num_heads
        full = partial.new_zeros(tokens, num_heads, ctx.d_head)
        full[:, head_start:head_end] = partial.view(
            tokens, local_num_heads, ctx.d_head
        )
        full = full.reshape(tokens, num_heads * ctx.d_head)
        dist.all_reduce(full, group=group)
        return full

    @staticmethod
    def backward(ctx, grad_full):
        grad_partial = grad_full.view(
            grad_full.shape[0], ctx.num_heads, ctx.d_head
        )[:, ctx.head_start : ctx.head_end]
        return (
            grad_partial.reshape(grad_full.shape[0], -1),
            None,
            None,
            None,
            None,
        )


def all_reduce_head_parallel_output(
    partial: torch.Tensor,
    head_range: tuple[int, int],
    num_heads: int,
    group: "dist.ProcessGroup",
) -> torch.Tensor:
    """Regime (a) assembly: zero-pad the local-head partial output and
    all-reduce it over the expert mesh so every rank holds the full-width
    result (tokens are replicated, so the local partials occupy disjoint
    head columns and the sum is the unsharded output).

    The backward is the plain slice of the upstream gradient to the owned
    head columns: tokens are replicated across the mesh, the downstream
    gradient is identical on every rank, and this rank's partial only feeds
    its own columns, so re-reducing the gradient (as
    ``torch.distributed.nn.functional.all_reduce`` would) over-counts by
    the mesh degree. This keeps sharded gradients exactly equal to the
    unsharded ones.
    """
    head_start, head_end = head_range
    return _AllReduceHeadParallel.apply(
        partial, head_start, head_end, num_heads, group
    )


class _AllReduceInputGrad(torch.autograd.Function):
    """Identity forward; all-reduce the gradient backward."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, group=ctx.group)
        return grad, None


def all_reduce_head_parallel_input_grad(
    x: torch.Tensor, group: "dist.ProcessGroup"
) -> torch.Tensor:
    """Regime-(a) input conjugate of :func:`all_reduce_head_parallel_output`.

    The MoE input is replicated across the expert mesh and every rank's
    local-head backward only produces its own heads' contribution, so the
    input gradient must be summed across the mesh before it propagates into
    the modules upstream of the MoE (which are replicated and expect the
    full, unsharded gradient).
    """
    return _AllReduceInputGrad.apply(x, group)


# ---------------------------------------------------------------------------
# Regime (b): Ulysses-style seq<->head all-to-all dispatch/combine.
# ---------------------------------------------------------------------------


def head_seq_dispatch_permute(x: torch.Tensor, ep_size: int) -> torch.Tensor:
    """Reshape ``(S, H, D)`` into ``(ep_size, S, H // ep_size, D)`` so dim 0
    holds the all-to-all chunk addressed to each rank (official
    ``ep_dispatch`` pre-all-to-all layout)."""
    seq_len, num_heads, d_head = x.shape
    return (
        x.contiguous()
        .view(seq_len, ep_size, num_heads // ep_size, d_head)
        .permute(1, 0, 2, 3)
        .contiguous()
    )


def head_seq_undispatch_permute(x: torch.Tensor, ep_size: int) -> torch.Tensor:
    """Reshape ``(ep_size, S, H // ep_size, D)`` (all-to-all output chunks
    indexed by source rank) back into ``(S, H, D)`` (official
    ``ep_undispatch`` post-all-to-all layout)."""
    _, seq_len, heads_per_ep, d_head = x.shape
    return (
        x.permute(1, 0, 2, 3).contiguous().view(seq_len, heads_per_ep * ep_size, d_head)
    )


class _EpDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        ep_size = dist.get_world_size(group)
        permuted = head_seq_dispatch_permute(x, ep_size)
        out = torch.empty_like(permuted)
        dist.all_to_all_single(out, permuted, group=group)
        seq_len = x.shape[0]
        return out.view(ep_size * seq_len, x.shape[1] // ep_size, x.shape[2])

    @staticmethod
    def backward(ctx, grad_output):
        return _ep_undispatch_raw(grad_output, ctx.group), None


class _EpUndispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return _ep_undispatch_raw(x, group)

    @staticmethod
    def backward(ctx, grad_output):
        ep_size = dist.get_world_size(ctx.group)
        permuted = head_seq_dispatch_permute(grad_output, ep_size)
        out = torch.empty_like(permuted)
        dist.all_to_all_single(out, permuted, group=ctx.group)
        seq_len = grad_output.shape[0]
        return out.view(ep_size * seq_len, grad_output.shape[1] // ep_size, grad_output.shape[2]), None


def _ep_undispatch_raw(x: torch.Tensor, group: "dist.ProcessGroup") -> torch.Tensor:
    ep_size = dist.get_world_size(group)
    seq_len = x.shape[0] // ep_size
    chunks = x.contiguous().view(
        ep_size, seq_len, x.shape[1], x.shape[2]
    )
    out = torch.empty_like(chunks)
    dist.all_to_all_single(out, chunks, group=group)
    return head_seq_undispatch_permute(out, ep_size)


def ep_dispatch(x: torch.Tensor, group: "dist.ProcessGroup | None") -> torch.Tensor:
    """Dispatch tokens to the ranks owning each head.

    Input ``(S, H, D)`` (this rank's sequence shard, all heads); output
    ``(S * ep_size, H // ep_size, D)`` (the full sequence concatenated in
    rank order, only this rank's heads). Port of the official
    ``ep_dispatch`` (magi2-preview inference/model/magi2_preview.py:3277),
    extended with autograd: the backward is the undispatch all-to-all.
    """
    ep_size = dist.get_world_size(group) if group is not None else 1
    if ep_size == 1:
        return x
    if x.shape[1] % ep_size != 0:
        raise ValueError(
            f"Number of heads H ({x.shape[1]}) must be divisible by "
            f"ep_size ({ep_size})"
        )
    return _EpDispatch.apply(x, group)


def ep_undispatch(x: torch.Tensor, group: "dist.ProcessGroup | None") -> torch.Tensor:
    """Undispatch head-local results back to the sequence-owning ranks.

    Input ``(S * ep_size, H // ep_size, D)``; output ``(S, H, D)`` with the
    head axis re-assembled. Port of the official ``ep_undispatch``
    (magi2-preview inference/model/magi2_preview.py:3303), extended with
    autograd: the backward is the dispatch all-to-all.
    """
    ep_size = dist.get_world_size(group) if group is not None else 1
    if ep_size == 1:
        return x
    if x.shape[0] % ep_size != 0:
        raise ValueError(
            f"Total sequence length ({x.shape[0]}) must be divisible by "
            f"ep_size ({ep_size})"
        )
    return _EpUndispatch.apply(x, group)
