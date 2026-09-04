# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoE dispatch/combine adapter for ``cann_ops_transformer.ElasticBuffer``.

The adapter deliberately implements only the communication primitives consumed by
TorchTitan's compact ``DeepEPTokenDispatcher`` training path. The dispatcher creates
and owns the ``ElasticBuffer`` instance; this module only transports that reference
through the custom-op boundary. Engram storage and the DeepEP expand/cudagraph
inference layout are outside its scope.
"""

__all__ = ["DispatchHandle", "ElasticBufferHandle"]

from typing import Any

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the Ascend device type
from cann_ops_transformer import ElasticBuffer, EPHandle
from torch._library.opaque_object import (
    CustomClassBase,  # pyrefly: ignore [missing-module-attribute]
    register_opaque_type,
)


class ElasticBufferHandle(CustomClassBase):
    """Opaque reference to an ``ElasticBuffer`` owned by the dispatcher."""

    def __init__(self, value: ElasticBuffer):
        self.value: ElasticBuffer = value

    def __eq__(self, other):
        return isinstance(other, ElasticBufferHandle) and self.value is other.value

    def __hash__(self):
        return id(self.value)


register_opaque_type(ElasticBufferHandle, typ="reference")


class DispatchHandle(CustomClassBase):
    """Opaque wrapper carrying a CANN ``EPHandle`` across custom-op boundaries."""

    def __init__(self, value: EPHandle | None = None):
        self.value: Any = value

    def __eq__(self, other):
        return isinstance(other, DispatchHandle) and self.value is other.value

    def __hash__(self):
        return 0 if self.value is None else id(self.value)

    def __fx_repr__(self):
        return "DispatchHandle()", {"DispatchHandle": DispatchHandle}


register_opaque_type(DispatchHandle, typ="reference")


@torch.library.custom_op("deepep::dispatch", mutates_args=(), device_types="npu")
def _dispatch_op_impl(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    buffer: ElasticBufferHandle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchHandle]:
    """Dispatch into CANN's compact expert-major layout (non-cached mode)."""
    x = x.contiguous()
    topk_idx = topk_idx.contiguous()
    topk_weights = topk_weights.contiguous()
    elastic_buffer = buffer.value

    recv_x, _recv_topk_idx, recv_scores, handle = elastic_buffer.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=num_experts,
        num_max_tokens_per_rank=x.shape[0],
        # In non-cached dispatch, there is no handle with a previously known receive count.
        # ``do_cpu_sync=True`` resets a host-pinned counter to -1; after the NPU
        # finishes computing the number of expert-routed token rows received by
        # local rank (the sum across its local experts), its dispatch kernel writes
        # that value to the counter. ``dispatch`` then CPU-spins until it observes
        # a non-negative value before allocating the exact receive tensors and routed metadata.
        do_cpu_sync=True,
    )
    return (
        recv_x,
        recv_scores,
        handle.num_recv_tokens_per_expert,
        DispatchHandle(value=handle),
    )


@_dispatch_op_impl.register_fake
def _dispatch_fake(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    num_local_experts: int,
    buffer: ElasticBufferHandle,
):
    # recv_x.shape[0] is data-dependent (actual tokens received at runtime), so a
    # static upper bound would fail inductor's assert_size_stride. This dynamic
    # symbol represents the actual number of rows produced by dispatch.
    num_recv = torch.library.get_ctx().new_dynamic_size()
    return (
        x.new_empty((num_recv, x.shape[1])),
        x.new_empty((num_recv,), dtype=torch.float32),
        x.new_empty((num_local_experts,), dtype=torch.int64),
        DispatchHandle(),
    )


@torch.library.custom_op("deepep::combine", mutates_args=(), device_types="npu")
def _combine_op_impl(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    buffer: ElasticBufferHandle,
) -> torch.Tensor:
    """Pure-reduction combine; routing scores are applied outside this op."""
    x = x.contiguous()
    elastic_buffer = buffer.value
    combined, _ = elastic_buffer.combine(x, handle.value, topk_weights=None)
    # Workaround for a CANN ElasticBuffer combine issue observed with the
    # dedicated DeepEP communication domain: without a process-group
    # synchronization, one rank can observe non-finite values from the
    # combine result, so synchronize once here.
    dist.barrier(elastic_buffer._group)
    return combined


@_combine_op_impl.register_fake
def _combine_fake(
    x: torch.Tensor,
    handle: DispatchHandle,
    num_tokens: int,
    buffer: ElasticBufferHandle,
):
    # CANN combine permutes the routed tokens back to the original token order,
    # so its output has num_tokens rows, NOT the combine input's num_recv rows.
    # Let torch.compile keep the true (dynamic) routed count.
    return x.new_empty((num_tokens, x.shape[1]))


@torch.library.custom_op("deepep::dispatch_backward", mutates_args=(), device_types="npu")
def _dispatch_backward_op_impl(
    grad_hidden: torch.Tensor,
    grad_scores: torch.Tensor,
    has_grad_scores: bool,
    topk_idx: torch.Tensor,
    handle: DispatchHandle,
    buffer: ElasticBufferHandle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward of ``dispatch`` implemented by CANN's cached ``combine``."""
    grad_hidden = grad_hidden.contiguous()
    grad_scores = grad_scores.contiguous()
    elastic_buffer = buffer.value
    grad_x, grad_topk_weights = elastic_buffer.combine(
        grad_hidden,
        handle.value,
        topk_weights=(grad_scores.reshape(-1).float().contiguous() if has_grad_scores else None),
    )
    if not has_grad_scores:
        # The custom-op schema always returns a Tensor for this slot; the
        # autograd wrapper discards it when the forward score output is unused.
        grad_topk_weights = grad_scores.new_empty(topk_idx.shape)
    # Workaround for a CANN ElasticBuffer combine issue observed with the
    # dedicated DeepEP communication domain: without a process-group
    # synchronization, one rank can observe non-finite values from the
    # combine result, so synchronize once here.
    dist.barrier(elastic_buffer._group)
    return grad_x, grad_topk_weights


@_dispatch_backward_op_impl.register_fake
def _dispatch_backward_fake(
    grad_hidden: torch.Tensor,
    grad_scores: torch.Tensor,
    has_grad_scores: bool,
    topk_idx: torch.Tensor,
    handle: DispatchHandle,
    buffer: ElasticBufferHandle,
):
    return (
        grad_hidden.new_empty((topk_idx.shape[0], grad_hidden.shape[1])),
        grad_hidden.new_empty(topk_idx.shape, dtype=torch.float32),
    )


@torch.library.custom_op("deepep::combine_backward", mutates_args=(), device_types="npu")
def _combine_backward_op_impl(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    buffer: ElasticBufferHandle,
) -> torch.Tensor:
    """Backward of ``combine`` implemented by CANN's cached ``dispatch``."""
    grad_combined = grad_combined.contiguous()
    elastic_buffer = buffer.value
    grad_x, _recv_topk_idx, _recv_scores, _new_handle = elastic_buffer.dispatch(
        grad_combined,
        handle=handle.value,
        # Cached dispatches reuse the count stored in the handle
        # and explicitly disable this synchronization.
        do_cpu_sync=False,
    )
    return grad_x


@_combine_backward_op_impl.register_fake
def _combine_backward_fake(
    grad_combined: torch.Tensor,
    handle: DispatchHandle,
    num_permuted_tokens: int,
    buffer: ElasticBufferHandle,
):
    return grad_combined.new_empty((num_permuted_tokens, grad_combined.shape[1]))


def _dispatch_setup_context(ctx, inputs, output) -> None:
    x, topk_idx, _topk_weights, _num_experts, _num_local_experts, buffer = inputs
    (
        _recv_x,
        _recv_scores,
        _num_recv_tokens_per_expert,
        handle,
    ) = output
    ctx.save_for_backward(topk_idx)
    ctx.handle = handle
    ctx.buffer = buffer
    ctx.input_dtype = x.dtype
    ctx.hidden = x.shape[1]


def _dispatch_backward(
    ctx,
    grad_recv_x,
    grad_recv_scores,
    _grad_num_recv_per_expert,
    _grad_handle,
):
    """Route data and routing-score gradients through the CANN combine path."""
    if grad_recv_x is None and grad_recv_scores is None:
        return None, None, None, None, None, None

    has_grad_x = grad_recv_x is not None
    has_grad_scores = grad_recv_scores is not None
    if grad_recv_x is None:
        grad_recv_x = torch.zeros(
            (grad_recv_scores.numel(), ctx.hidden),
            dtype=ctx.input_dtype,
            device=grad_recv_scores.device,
        )
    if grad_recv_scores is None:
        grad_recv_scores = grad_recv_x.new_empty((0,), dtype=torch.float32)
    (topk_idx,) = ctx.saved_tensors
    grad_x, grad_topk_weights = torch.ops.deepep.dispatch_backward(
        grad_recv_x,
        grad_recv_scores,
        has_grad_scores,
        topk_idx,
        ctx.handle,
        ctx.buffer,
    )
    return (
        grad_x.to(ctx.input_dtype) if has_grad_x else None,
        None,
        grad_topk_weights if has_grad_scores else None,
        None,
        None,
        None,
    )


def _combine_setup_context(ctx, inputs, output) -> None:
    x, handle, _, buffer = inputs
    ctx.handle = handle
    ctx.buffer = buffer
    ctx.num_permuted_tokens = x.shape[0]


def _combine_backward(ctx, grad_combined):
    """Scatter combined gradients through the cached CANN dispatch path."""
    grad_x = torch.ops.deepep.combine_backward(
        grad_combined,
        ctx.handle,
        ctx.num_permuted_tokens,
        ctx.buffer,
    )
    return grad_x, None, None, None


torch.library.register_autograd("deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context)
torch.library.register_autograd("deepep::combine", _combine_backward, setup_context=_combine_setup_context)
