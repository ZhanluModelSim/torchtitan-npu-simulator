# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2006 Idiap Research Institute (Samy Bengio)
# Copyright (c) 2013 the respective contributors
# Licensed under the BSD 3-Clause License (see LICENSE for details).

"""NPU-side accumulation for CPU-offloaded FSDP gradients.

FSDP owns the canonical gradient object and its host allocation. On the
second and later microbatch, this module only changes where the addition is
performed: a bounded scratch slice is filled from the pinned host gradient,
updated with the NPU reduce-scatter result, and copied back to that same host
slice. The upstream first-microbatch path and all DTensor metadata remain
unchanged.

Patching strategy
-----------------
Instead of rewriting upstream source text, three thin wrappers replace the
upstream entry points and *delegate to the untouched originals* whenever CPU
offload is disabled or not applicable, so non-offload training runs pristine
upstream code:

* ``FSDPParamGroup.unshard`` / ``FSDPParamGroup.wait_for_unshard``: singleton
  (world-size 1) groups whose parameters are CPU-offloaded prefetch their
  parameters to the NPU through the all-gather copy-in stream, replacing
  upstream's synchronous copy in ``wait_for_unshard``.
* ``foreach_reduce``: reduced shards that must be accumulated into a
  CPU-canonical gradient are added on the NPU (see ``accumulate_cpu_grad``)
  instead of through upstream's blocking D2H copy plus host add.

Each wrapper carries a marker attribute so installation is idempotent, keeps
``__wrapped__`` pointing at the original for debugging, and the upstream
signatures are validated at install time so incompatible PyTorch releases
fail loudly instead of silently misbehaving.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from inspect import signature
from itertools import chain
from typing import Any

import torch
import torch.distributed as dist
import torch_npu
from torch import Tensor
from torch.distributed.fsdp._fully_shard import _fsdp_collectives as _collectives
from torch.distributed.fsdp._fully_shard import _fsdp_param_group as _param_group
from torch.distributed.fsdp._fully_shard._fsdp_collectives import (
    AllGatherResult,
    _div_if_needed,
    _get_all_gather_input_metadatas,
    _get_device_handle,
    _get_gradient_divide_factors,
    _get_param_all_gather_inputs,
    foreach_all_gather_copy_out,
    foreach_reduce_scatter_copy_in,
)
from torch.distributed.fsdp._fully_shard._fsdp_common import (
    FSDPMeshInfo,
    TrainingState,
    _disable_functorch_if_active,
    _get_dim0_padded_size,
    _to_dtype_if_needed,
)
from torch.distributed.tensor import DTensor
from torch.profiler import record_function

from torchtitan_npu.extensions.cpu_offload.runtime import local_tensor

_PATCH_MARKER = "_torchtitan_npu_fsdp_grad_accum_v1"
_CHUNK_BYTES = 64 * 1024 * 1024  # NPU bounce-buffer granularity for grad accumulation
_PREFETCH_MARKER = "_torchtitan_npu_cpu_offload_prefetch_v2"
_CPU_OFFLOAD_ENABLED = False


@dataclass(slots=True)
class _AccumSlot:
    buffer: Tensor
    release: Any | None = None


class _AccumPipeline:
    """Bounded H2D/compute/D2H pipeline for one NPU device."""

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)  # pyrefly: ignore [read-only]
        rank = os.environ.get("RANK", "0")
        from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging

        self.staging = CpuStaging(self.device, owner=f"grad-accum-rank{rank}")
        self.compute_stream = torch_npu.npu.Stream(device=self.device)
        self._slots: dict[tuple[torch.dtype, int], tuple[_AccumSlot, _AccumSlot]] = {}
        self._next_slot: dict[tuple[torch.dtype, int], int] = {}

    def slots(
        self, dtype: torch.dtype, chunk_numel: int
    ) -> tuple[tuple[_AccumSlot, _AccumSlot], tuple[torch.dtype, int]]:
        key = (dtype, chunk_numel)
        slots = self._slots.get(key)
        if slots is None:
            slots = (
                _AccumSlot(torch.empty(chunk_numel, dtype=dtype, device=self.device)),
                _AccumSlot(torch.empty(chunk_numel, dtype=dtype, device=self.device)),
            )
            self._slots[key] = slots
        return slots, key

    def next_slot(self, key: tuple[torch.dtype, int]) -> int:
        index = self._next_slot.get(key, 0)
        self._next_slot[key] = index + 1
        return index

    def close(self) -> None:
        self.staging.close()
        self._slots.clear()
        self._next_slot.clear()


_PIPELINES_BY_DEVICE: dict[torch.device, _AccumPipeline] = {}


def register_cpu_offload_hooks() -> None:
    """Enable the CPU-offload FSDP hooks owned by this module."""
    global _CPU_OFFLOAD_ENABLED
    _CPU_OFFLOAD_ENABLED = True


def unregister_cpu_offload_hooks() -> None:
    """Disable the CPU-offload FSDP hooks (symmetric to register).

    The wrappers stay installed; they delegate to the untouched upstream
    functions while disabled.
    """
    global _CPU_OFFLOAD_ENABLED
    _CPU_OFFLOAD_ENABLED = False


def _cpu_offload_enabled() -> bool:
    return _CPU_OFFLOAD_ENABLED


def _pipeline_for(device: torch.device) -> _AccumPipeline:
    pipeline = _PIPELINES_BY_DEVICE.get(device)
    if pipeline is None:
        pipeline = _AccumPipeline(device)
        _PIPELINES_BY_DEVICE[device] = pipeline
    return pipeline


@torch.no_grad()
def accumulate_cpu_grad(
    fsdp_param: Any,
    incoming: Tensor,
    compute_device: torch.device,
    *,
    stream: Any,
    synchronous: bool = False,
) -> None:
    """Accumulate one reduced shard in CPU canonical storage via NPU.

    incoming is already the post-reduce dtype and is a view of FSDP's
    reduce-scatter output. The operation is stream ordered: the scratch
    slice is reused only after its queued D2H copy on the same stream.
    """
    grad = getattr(getattr(fsdp_param, "sharded_param", None), "grad", None)
    if grad is None:
        raise RuntimeError("FSDP gradient accumulation requires an existing canonical CPU gradient")
    canonical = local_tensor(grad)
    compute = torch.device(compute_device)
    if canonical.device.type != "cpu":
        raise RuntimeError(f"FSDP gradient accumulation requires CPU canonical storage, got {canonical.device}")
    if incoming.device != compute:
        raise RuntimeError(f"FSDP gradient accumulation input is on {incoming.device}, expected {compute}")
    if incoming.numel() != canonical.numel():
        raise RuntimeError(
            "FSDP gradient accumulation shape mismatch: "
            f"incoming={tuple(incoming.shape)} canonical={tuple(canonical.shape)}"
        )
    if incoming.dtype != canonical.dtype:
        raise RuntimeError(
            f"FSDP gradient accumulation dtype mismatch: incoming={incoming.dtype} canonical={canonical.dtype}"
        )
    if incoming.numel() == 0:
        return
    if not canonical.is_contiguous() or not incoming.is_contiguous():
        raise RuntimeError("FSDP NPU gradient accumulation requires contiguous canonical and reduced shards")

    canonical_flat = canonical.view(-1)
    incoming_flat = incoming.view(-1)
    chunk_numel = max(1, _CHUNK_BYTES // canonical.element_size())
    previous_offload = getattr(fsdp_param, "grad_offload_event", None)
    if synchronous or not canonical.is_pinned():
        if previous_offload is not None:
            previous_offload.synchronize()
        canonical.add_(incoming.to(device="cpu"))
        return

    pipeline = _pipeline_for(compute)
    transfer_stream = pipeline.staging.stream
    if previous_offload is not None:
        # A second microbatch can enter before FSDP's finalization fence.  The
        # H2D must observe the first microbatch's D2H before reading the same
        # canonical slice.
        transfer_stream.wait_event(previous_offload)
    slots, slot_key = pipeline.slots(
        canonical.dtype,
        min(chunk_numel, canonical.numel()),
    )
    producer_ready = torch_npu.npu.Event()
    producer_ready.record(stream)
    pipeline.compute_stream.wait_event(producer_ready)
    incoming_flat.record_stream(pipeline.compute_stream)
    last_complete = None
    for offset in range(0, canonical.numel(), chunk_numel):
        count = min(chunk_numel, canonical.numel() - offset)
        slot = slots[pipeline.next_slot(slot_key) % 2]
        destination = canonical_flat[offset : offset + count]
        source = incoming_flat[offset : offset + count]
        work = slot.buffer[:count]
        with torch_npu.npu.stream(transfer_stream):
            if slot.release is not None:
                transfer_stream.wait_event(slot.release.event)
        h2d = pipeline.staging.submit_h2d(
            destination,
            work,
            stream=transfer_stream,
            track=False,
        )
        h2d.wait_on(pipeline.compute_stream)
        with torch_npu.npu.stream(pipeline.compute_stream):
            work.add_(source)
        d2h = pipeline.staging.submit_d2h(
            work,
            destination,
            producer_stream=pipeline.compute_stream,
            stream=transfer_stream,
            track=False,
        )
        slot.release = d2h
        last_complete = d2h
    if last_complete is None:
        raise AssertionError("gradient accumulation did not enqueue a copy")
    fsdp_param.grad_offload_event = last_complete.event


def clear() -> None:
    """Release process-local staging resources at shutdown or test teardown."""
    for pipeline in _PIPELINES_BY_DEVICE.values():
        pipeline.close()
    _PIPELINES_BY_DEVICE.clear()


# ---------------------------------------------------------------------------
# Prefetch for singleton, CPU-offloaded parameter groups
# ---------------------------------------------------------------------------


def _is_single_rank_cpu_offload(group: Any) -> bool:
    if not _cpu_offload_enabled():
        return False
    if not isinstance(group.mesh_info, FSDPMeshInfo):
        return False
    return bool(
        group._all_gather_process_group.size() == 1
        and group.fsdp_params
        and all(fsdp_param.offload_to_cpu for fsdp_param in group.fsdp_params)
    )


@torch.no_grad()
def _copy_only_all_gather(
    fsdp_params: list[Any],
    group: Any,
    async_op: bool,
    all_gather_copy_in_stream: Any,
    all_gather_stream: Any,
    device: torch.device,
    all_gather_comm: Any,
) -> Any:
    """Stage singleton CPU-offloaded parameters without issuing a collective.

    Copy-in section derived from PyTorch ``_fsdp_collectives.foreach_all_gather``
    (BSD 3-Clause, Copyright (c) Facebook, Inc. and its affiliates).
    """
    del async_op, all_gather_stream
    if group.size() != 1:
        raise RuntimeError("copy-only all-gather requires a singleton process group")

    device_handle = _get_device_handle(device.type)
    if device_handle is None:
        raise RuntimeError(f"no device module is registered for {device.type}")
    with device_handle.stream(all_gather_copy_in_stream):
        param_inputs = _get_param_all_gather_inputs(fsdp_params)
        input_dtypes, input_numels, dtype = _get_all_gather_input_metadatas(param_inputs)
        if dtype == torch.uint8:
            inputs = [tensor.view(torch.uint8) for tensors in param_inputs for tensor in tensors]
        else:
            inputs = [*chain.from_iterable(param_inputs)]
        input_split_sizes = [tensor.numel() for tensor in inputs]
        input_numel = sum(input_split_sizes)
        output = all_gather_comm.allocate((input_numel,), dtype=dtype, device=device)
        _, output = torch.ops.fsdp.all_gather_copy_in(
            inputs,
            output,
            input_split_sizes,
            input_numel,
            0,
        )
        ready_event = all_gather_copy_in_stream.record_event()

    return AllGatherResult(
        output,
        ready_event,
        None,
        input_dtypes,
        input_numels,
        input_split_sizes,
    )


def _is_copy_only_result(group: Any) -> bool:
    result = group._all_gather_result
    return bool(
        _cpu_offload_enabled()
        and result is not None
        and result.all_gather_work is None
        and result.all_gather_input_split_sizes
    )


def _unshard_already_satisfied(group: Any) -> bool:
    """Mirror upstream ``unshard``'s early returns before its main body."""
    if group._all_gather_result is not None:  # already called, pending wait
        return True
    if group.is_unsharded:
        return True  # no-op
    return not group.unshard_in_backward and group._training_state == TrainingState.PRE_BACKWARD


# Wraps PyTorch _fsdp_param_group.FSDPParamGroup.unshard (BSD 3-Clause).
@_disable_functorch_if_active
def _patched_unshard(self: Any, async_op: bool = False) -> None:
    """Wrap ``FSDPParamGroup.unshard`` with a copy-only prefetch path.

    Upstream's singleton path stages nothing in ``unshard`` and pays a
    synchronous copy in ``wait_for_unshard``. For CPU-offloaded parameters we
    reuse the regular copy-in machinery on the all-gather copy-in stream so
    the H2D transfer overlaps with compute; every other case delegates to the
    original method.
    """
    if not _is_single_rank_cpu_offload(self) or _unshard_already_satisfied(self):
        return _ORIGINAL_UNSHARD(self, async_op)
    if self._reshard_after_forward_event is not None:
        # Resharded parameter data is allocated in the default stream and
        # used in the all-gather streams
        self._wait_all_gather_streams_on_event(self._reshard_after_forward_event)
        self._reshard_after_forward_event = None
    with record_function(self._with_fqn("FSDP::all_gather_copy_in")):
        copy_in_stream, gather_stream = self.comm_ctx.get_all_gather_streams(async_op, self._training_state)
        self._all_gather_result = _copy_only_all_gather(
            self.fsdp_params,
            self._all_gather_process_group,
            async_op,
            copy_in_stream,
            gather_stream,
            self.device,
            self._all_gather_comm,
        )


def _wait_for_copy_only_unshard(group: Any) -> None:
    """Consume a copy-only all-gather result like the multi-rank copy-out."""
    result = group._all_gather_result
    if group._training_state == TrainingState.FORWARD and (  # implicit prefetch
        prev_all_gather_state := group.comm_ctx.all_gather_state
    ):
        group._wait_all_gather_streams_on_event(prev_all_gather_state.event)
        group.comm_ctx.all_gather_state = None  # free the all-gather result
    with record_function(group._with_fqn("FSDP::all_gather_copy_out")):
        foreach_all_gather_copy_out(
            result,
            group.fsdp_params,
            group._all_gather_process_group,
        )
    for fsdp_param in group.fsdp_params:
        fsdp_param.init_unsharded_param()

    group._to_unsharded()
    all_gather_copy_out_event = group.device_handle.Event()
    all_gather_copy_out_event.record()
    # Upstream only defers the free for multi-rank all-gathers; copy-only
    # groups are single-rank, so always wait on the copy-out event.
    group._wait_all_gather_streams_on_event(all_gather_copy_out_event)
    group._all_gather_result = None


# Wraps PyTorch _fsdp_param_group.FSDPParamGroup.wait_for_unshard (BSD 3-Clause).
@_disable_functorch_if_active
def _patched_wait_for_unshard(self: Any) -> None:
    """Wrap ``FSDPParamGroup.wait_for_unshard`` for copy-only results."""
    if _is_copy_only_result(self):
        _wait_for_copy_only_unshard(self)
        return None
    return _ORIGINAL_WAIT_FOR_UNSHARD(self)


# ---------------------------------------------------------------------------
# NPU accumulation of reduced shards into CPU-canonical gradients
# ---------------------------------------------------------------------------


def _uses_cpu_canonical_grads(fsdp_params: list[Any], device: torch.device) -> bool:
    """Whether this reduction feeds CPU-canonical gradients on the NPU."""
    return (
        _cpu_offload_enabled() and device.type == "npu" and any(fsdp_param.offload_to_cpu for fsdp_param in fsdp_params)
    )


# Mirrors PyTorch _fsdp_collectives.foreach_reduce (BSD 3-Clause).
def _cpu_offload_foreach_reduce(
    fsdp_params: list[Any],
    unsharded_grads: list[Tensor],
    reduce_scatter_group: Any,
    reduce_scatter_stream: torch.Stream,
    reduce_scatter_comm: Any,
    orig_dtype: torch.dtype | None,
    reduce_dtype: torch.dtype | None,
    device: torch.device,
    gradient_divide_factor: float | None,
    all_reduce_group: Any,  # not `None` iff HSDP
    all_reduce_stream: torch.Stream,
    all_reduce_grads: bool,
    partial_reduce_output: Tensor | None,  # only used for HSDP
    all_reduce_hook: Any,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    Tensor,
    Any,
    torch.Stream,
    Any,
    Tensor | None,
    Any,
    Tensor | None,
]:
    """Upstream ``foreach_reduce`` with NPU accumulation for CPU-canonical grads.

    This mirrors
    ``torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_reduce``
    with a single change in the post-reduce loop: a reduced shard that must be
    accumulated into an existing CPU-canonical gradient is added on the NPU
    through :func:`accumulate_cpu_grad` instead of upstream's blocking D2H copy
    plus host-side add. Everything else, including the HSDP all-reduce,
    all-reduce-hook, and partial-reduction paths, follows upstream.
    """
    grad_dtype = unsharded_grads[0].dtype
    reduce_dtype = reduce_dtype or grad_dtype
    (predivide_factor, postdivide_factor, reduce_scatter_op, all_reduce_op) = _get_gradient_divide_factors(
        reduce_scatter_group,
        all_reduce_group,
        reduce_dtype,
        device.type,
        gradient_divide_factor,
        force_sum_reduction_for_comms,
    )

    world_size = 1 if reduce_scatter_group is None else reduce_scatter_group.size()
    device_handle = _get_device_handle(device.type)
    if device_handle is None:
        raise RuntimeError("NPU is the only supported backend")
    current_stream = device_handle.current_stream()

    # CPU-offloaded models can carry hardcoded fp32 parameters alongside
    # bf16 mixed-precision weights, producing mixed-dtype gradients within
    # one param group. Upstream FSDP expects uniform dtype at the copy-in;
    # cast everything to the reduce dtype (fp32 under mixed-precision-reduce)
    # before entering the reduce-scatter pipeline.
    if len({grad.dtype for grad in unsharded_grads}) > 1:
        unsharded_grads = [grad.to(reduce_dtype) if grad.dtype != reduce_dtype else grad for grad in unsharded_grads]

    if world_size > 1:
        for i, (fsdp_param, unsharded_grad) in enumerate(zip(fsdp_params, unsharded_grads, strict=False)):
            if (shard_dim := fsdp_param.fsdp_placement.dim) == 0:
                continue
            if unsharded_grad.size(shard_dim) % world_size != 0:
                raise AssertionError(
                    f"Shard({shard_dim}) requires even sharding: {unsharded_grad.size()=} {world_size=}"
                )
            chunks = torch.chunk(unsharded_grad, world_size, dim=shard_dim)
            unsharded_grads[i] = torch.cat(chunks, dim=0)

    padded_unsharded_sizes = tuple(_get_dim0_padded_size(grad.size(), world_size) for grad in unsharded_grads)
    reduce_scatter_input_numel = sum(s.numel() for s in padded_unsharded_sizes)
    reduce_scatter_output_numel = reduce_scatter_input_numel // world_size
    reduce_scatter_input = reduce_scatter_comm.allocate(
        (reduce_scatter_input_numel,),
        dtype=reduce_dtype,
        device=device,
    )

    foreach_reduce_scatter_copy_in(unsharded_grads, reduce_scatter_input, world_size)

    # Only after the copy-in finishes can we free the gradients
    unsharded_grads.clear()
    reduce_scatter_stream.wait_stream(current_stream)
    all_reduce_input = None
    all_reduce_event = None

    with device_handle.stream(reduce_scatter_stream):
        reduce_output = reduce_scatter_comm.allocate(
            (reduce_scatter_output_numel,),
            dtype=reduce_dtype,
            device=device,
        )
        _div_if_needed(reduce_scatter_input, predivide_factor)
        if world_size > 1:
            reduce_scatter_comm(
                output_tensor=reduce_output,
                input_tensor=reduce_scatter_input,
                group=reduce_scatter_group,
                op=reduce_scatter_op,
            )
        else:
            # For single GPU, just copy the input to output (no actual reduce-scatter needed), and
            # account for a possible gradient_divide_factor.
            if gradient_divide_factor is not None:
                reduce_output.copy_(reduce_scatter_input / gradient_divide_factor)
            else:
                reduce_output.copy_(reduce_scatter_input)
        reduce_scatter_event = reduce_scatter_stream.record_event()
        post_reduce_stream = reduce_scatter_stream
        if all_reduce_group is not None:  # HSDP or DDP/replicate
            # Accumulations must run in the reduce-scatter stream
            if not all_reduce_grads:
                if partial_reduce_output is not None:
                    partial_reduce_output += reduce_output
                else:
                    partial_reduce_output = reduce_output
                return (
                    reduce_scatter_input,
                    reduce_scatter_event,
                    post_reduce_stream,
                    post_reduce_stream.record_event(),
                    all_reduce_input,
                    all_reduce_event,
                    partial_reduce_output,
                )
            if partial_reduce_output is not None:
                reduce_output += partial_reduce_output
            post_reduce_stream = all_reduce_stream
            if world_size >= 1:
                all_reduce_stream.wait_stream(reduce_scatter_stream)
            else:
                all_reduce_stream.wait_stream(current_stream)
            with device_handle.stream(all_reduce_stream):
                dist.all_reduce(
                    reduce_output,
                    group=all_reduce_group,
                    op=all_reduce_op,
                )
                # Keep refs to the reduce-dtype AR buffer + completion event
                # so FSDPParamGroup._all_reduce_state can hold them across
                # layers; see upstream PR #140044 / #180900.
                all_reduce_input = reduce_output
                all_reduce_event = all_reduce_stream.record_event()
    # -- END: ops in reduce_scatter stream

    if all_reduce_hook is not None:
        # Execute user-specified all reduce hook.
        # If native HSDP is used, this is executed after the HSDP all reduce.
        # If 1-d FSDP is used, this is executed post reduce-scatter.
        post_reduce_stream = all_reduce_stream
        all_reduce_stream.wait_stream(reduce_scatter_stream)
        with device_handle.stream(all_reduce_stream):
            all_reduce_hook(reduce_output)
    # -- END: ops post reduce_scatter

    with device_handle.stream(post_reduce_stream):
        _div_if_needed(reduce_output, postdivide_factor)
        # Rebinds to a new orig_dtype tensor when reduce_dtype != orig_dtype
        reduce_output = _to_dtype_if_needed(reduce_output, orig_dtype)
        # View out and accumulate sharded gradients
        flat_grad_offset = 0  # [0, reduce_scatter_output_numel - 1]
        for padded_unsharded_size, fsdp_param in zip(padded_unsharded_sizes, fsdp_params, strict=False):
            # Assume even sharding for Shard(i), i > 0; otherwise would require
            # copy-out for contiguous strides
            new_sharded_grad = torch.as_strided(
                reduce_output,
                size=fsdp_param.sharded_size,
                stride=fsdp_param.contiguous_sharded_stride,
                storage_offset=flat_grad_offset,
            )
            to_accumulate_grad = fsdp_param.sharded_param.grad is not None
            # In-backward CPU consumers of the grad must observe completed
            # transfers, hence the synchronous mode below.
            has_post_acc_grad_hook = bool(getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", None))
            if fsdp_param.offload_to_cpu:
                if to_accumulate_grad:
                    # Accumulate into CPU-canonical storage on the NPU:
                    # chunked H2D, add, and D2H through pinned bounce
                    # buffers, all stream ordered.
                    accumulate_cpu_grad(
                        fsdp_param,
                        new_sharded_grad,
                        device,
                        stream=post_reduce_stream,
                        synchronous=has_post_acc_grad_hook,
                    )
                else:
                    # First microbatch for this parameter: keep upstream's
                    # D2H staging and install it as the canonical gradient.
                    non_blocking = fsdp_param.pin_memory and not has_post_acc_grad_hook
                    grad_dest = getattr(fsdp_param, "_pooled_grad_dest", None)
                    if (
                        non_blocking
                        and grad_dest is not None
                        and grad_dest.shape == new_sharded_grad.shape
                        and grad_dest.dtype == new_sharded_grad.dtype
                    ):
                        # Pre-carved pinned slot from the flat pool: avoids the
                        # per-gradient allocation inside ``.to(non_blocking=True)``,
                        # which the host pinned allocator rounds up to a
                        # power-of-two size class (~5 MB average per gradient).
                        grad_dest.copy_(new_sharded_grad, non_blocking=True)
                        new_sharded_grad = grad_dest
                    else:
                        new_sharded_grad = new_sharded_grad.to(torch.device("cpu"), non_blocking=non_blocking)
                    if non_blocking:
                        # Record an event on which to block the CPU thread to
                        # ensure that the D2H copy finishes before the optimizer
                        fsdp_param.grad_offload_event = post_reduce_stream.record_event()
                    fsdp_param.sharded_param.grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            elif to_accumulate_grad:
                if not isinstance(fsdp_param.sharded_param.grad, DTensor):
                    raise AssertionError(
                        "Expected fsdp_param.sharded_param.grad to be DTensor, "
                        f"got {type(fsdp_param.sharded_param.grad)}"
                    )
                fsdp_param.sharded_param.grad._local_tensor += new_sharded_grad
            else:
                fsdp_param.sharded_param.grad = fsdp_param.to_sharded_dtensor(new_sharded_grad)
            for hook in (getattr(fsdp_param.sharded_param, "_post_accumulate_grad_hooks", {}) or {}).values():
                hook(fsdp_param.sharded_param)
            padded_sharded_numel = padded_unsharded_size.numel() // world_size
            flat_grad_offset += padded_sharded_numel
        post_reduce_event = post_reduce_stream.record_event()
    return (
        reduce_scatter_input,
        reduce_scatter_event,
        post_reduce_stream,
        post_reduce_event,
        all_reduce_input,
        all_reduce_event,
        None,
    )


def _patched_foreach_reduce(*args: Any, **kwargs: Any) -> Any:
    """Wrap ``foreach_reduce`` to accumulate CPU-canonical grads on the NPU.

    ``args[0]`` is ``fsdp_params`` and ``args[7]`` is ``device`` (the
    upstream positional signature); keyword calls are forwarded verbatim.
    """
    fsdp_params = args[0] if args else kwargs["fsdp_params"]
    device = args[7] if len(args) > 7 else kwargs["device"]
    if _uses_cpu_canonical_grads(fsdp_params, device):
        return _cpu_offload_foreach_reduce(*args, **kwargs)
    return _ORIGINAL_FOREACH_REDUCE(*args, **kwargs)


# Installation
# ---------------------------------------------------------------------------

_ORIGINAL_FOREACH_REDUCE: Any = _collectives.foreach_reduce
_ORIGINAL_UNSHARD: Any = _param_group.FSDPParamGroup.unshard
_ORIGINAL_WAIT_FOR_UNSHARD: Any = _param_group.FSDPParamGroup.wait_for_unshard

_EXPECTED_FOREACH_REDUCE_PARAMS = (
    "fsdp_params",
    "unsharded_grads",
    "reduce_scatter_group",
    "reduce_scatter_stream",
    "reduce_scatter_comm",
    "orig_dtype",
    "reduce_dtype",
    "device",
    "gradient_divide_factor",
    "all_reduce_group",
    "all_reduce_stream",
    "all_reduce_grads",
    "partial_reduce_output",
    "all_reduce_hook",
    "force_sum_reduction_for_comms",
)


# Validates against PyTorch _fsdp_collectives/_fsdp_param_group signatures (BSD 3-Clause).
def _validate_upstream_signatures() -> None:
    """Fail loudly on PyTorch releases the wrappers were not built against."""
    foreach_reduce_params = tuple(signature(_ORIGINAL_FOREACH_REDUCE).parameters)
    if foreach_reduce_params != _EXPECTED_FOREACH_REDUCE_PARAMS:
        raise RuntimeError(
            "Unsupported PyTorch FSDP foreach_reduce signature "
            f"{foreach_reduce_params}; CPU-canonical NPU gradient accumulation "
            "was not installed"
        )
    unshard_params = tuple(signature(_ORIGINAL_UNSHARD).parameters)
    if unshard_params != ("self", "async_op"):
        raise RuntimeError(
            f"Unsupported PyTorch FSDP unshard signature {unshard_params}; CPU-offload prefetch was not installed"
        )
    wait_for_unshard_params = tuple(signature(_ORIGINAL_WAIT_FOR_UNSHARD).parameters)
    if wait_for_unshard_params != ("self",):
        raise RuntimeError(
            f"Unsupported PyTorch FSDP wait_for_unshard signature {wait_for_unshard_params}; "
            "CPU-offload prefetch was not installed"
        )


def _wrap_metadata(wrapper: Any, original: Any, marker: str) -> None:
    """Give the wrapper the original's identity plus a marker and escape hatch."""
    functools.update_wrapper(
        wrapper,
        original,
        assigned=("__module__", "__name__", "__qualname__"),
        updated=(),
    )
    setattr(wrapper, marker, True)


_wrap_metadata(_patched_foreach_reduce, _ORIGINAL_FOREACH_REDUCE, _PATCH_MARKER)
_wrap_metadata(_patched_unshard, _ORIGINAL_UNSHARD, _PREFETCH_MARKER)
_wrap_metadata(_patched_wait_for_unshard, _ORIGINAL_WAIT_FOR_UNSHARD, _PREFETCH_MARKER)


def install() -> None:
    """Install the wrappers once per process, leaving originals reachable.

    Called explicitly by the CPU-offload optimizer container
    (:class:`torchtitan_npu.override.common.optimizer.CpuOffloadOptimizersContainer`)
    when the user selects the ``cpu_offload`` override; importing this module
    alone has no side effects. Without the installation, upstream PyTorch
    FSDP and the CPU-offload gradients are not patched.
    """
    foreach_reduce_patched = getattr(_collectives.foreach_reduce, _PATCH_MARKER, False)
    unshard_patched = getattr(_param_group.FSDPParamGroup.unshard, _PREFETCH_MARKER, False)
    wait_for_unshard_patched = getattr(_param_group.FSDPParamGroup.wait_for_unshard, _PREFETCH_MARKER, False)
    if foreach_reduce_patched and unshard_patched and wait_for_unshard_patched:
        return
    _validate_upstream_signatures()
    if not foreach_reduce_patched:
        _collectives.foreach_reduce = _patched_foreach_reduce
        # ``_fsdp_param_group`` binds the function with a from-import.
        _param_group.foreach_reduce = _patched_foreach_reduce
    if not unshard_patched:
        _param_group.FSDPParamGroup.unshard = _patched_unshard
    if not wait_for_unshard_patched:
        _param_group.FSDPParamGroup.wait_for_unshard = _patched_wait_for_unshard
