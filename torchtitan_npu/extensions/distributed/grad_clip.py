# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run clipping for CPU-canonical FSDP gradients on the NPU."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.distributed as dist
import torch_npu
import torchtitan.distributed.utils as distributed_utils
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _is_shard_like

from torchtitan_npu.extensions.cpu_offload import runtime as clip_state

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from torchtitan_npu.extensions.cpu_offload.staging import TransferHandle


_ORIGINAL_CLIP_GRAD_NORM = distributed_utils.clip_grad_norm_
_PATCHED = "_torchtitan_npu_cpu_clip_patched"
_CHUNK_BYTES = 64 * 1024 * 1024  # NPU bounce-buffer granularity for clip staging


@dataclass(slots=True)
class _NormSlot:
    buffer: torch.Tensor
    done: Any | None = None


@dataclass(slots=True)
class _NormGroup:
    mesh: Any
    reduce_dims: tuple[int, ...]
    value: torch.Tensor


@dataclass(slots=True)
class _CachedNormItem:
    npu_gradient: torch.Tensor
    group: _NormGroup
    handles: tuple[TransferHandle, ...]


class _NormPipeline:
    """Streams and bounded buffers shared by CPU-gradient clipping steps."""

    def __init__(self, device: torch.device) -> None:
        rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = device  # pyrefly: ignore [read-only]
        from torchtitan_npu.extensions.cpu_offload.staging import CpuStaging

        self.staging = CpuStaging(device, owner=f"clip-rank{rank}")
        self.compute_stream = torch_npu.npu.Stream(device=device)
        self.slots: dict[tuple[torch.dtype, int], tuple[_NormSlot, _NormSlot]] = {}

    def buffers(self, dtype: torch.dtype, chunk_numel: int) -> tuple[_NormSlot, _NormSlot]:
        key = (dtype, chunk_numel)
        slots = self.slots.get(key)
        if slots is None:
            slots = (
                _NormSlot(torch.empty(chunk_numel, dtype=dtype, device=self.device)),
                _NormSlot(torch.empty(chunk_numel, dtype=dtype, device=self.device)),
            )
            self.slots[key] = slots
        return slots

    def close(self) -> None:
        """Release the staging lane, pinned bounces, and NPU slot buffers."""
        self.staging.close()
        self.slots.clear()


_PIPELINES: dict[torch.device, _NormPipeline] = {}


def _pipeline(device: torch.device) -> _NormPipeline:
    pipeline = _PIPELINES.get(device)
    if pipeline is None:
        pipeline = _NormPipeline(device)
        _PIPELINES[device] = pipeline
    return pipeline


def clear() -> None:
    """Release process-local clip pipelines at teardown or re-configuration."""
    for pipeline in _PIPELINES.values():
        pipeline.close()
    _PIPELINES.clear()


def _norm_group_key(gradient: DTensor) -> tuple[object, tuple[int, ...]]:
    """Return mesh axes that contribute distinct shards to the global norm."""
    reduce_dims = []
    for mesh_dim, placement in enumerate(gradient.placements):
        if getattr(placement, "is_partial", lambda: False)():
            raise NotImplementedError(
                "CPU gradient clipping does not support Partial placements; "
                "FSDP must materialize a sharded or replicated gradient first"
            )
        if _is_shard_like(placement) and gradient.device_mesh.size(mesh_dim) > 1:
            reduce_dims.append(mesh_dim)
        elif not placement.is_replicate() and not _is_shard_like(placement):
            raise NotImplementedError(f"CPU gradient clipping does not support {placement!r} placements")
    return gradient.device_mesh, tuple(reduce_dims)


def _compute_device(parameters: list[torch.Tensor]) -> torch.device | None:
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        local = clip_state.local_tensor(gradient)
        if local.device.type != "cpu":
            return local.device
        if isinstance(gradient, DTensor) and gradient.device_mesh.device_type == "npu":
            npu = getattr(torch, "npu", None)
            if npu is not None and npu.is_available():
                return torch.device("npu", npu.current_device())
    return None


def _get_norm_group(
    groups: dict[tuple[int, tuple[int, ...]], _NormGroup],
    gradient: DTensor,
    device: torch.device,
    *,
    infinite: bool,
) -> _NormGroup:
    mesh, reduce_dims = _norm_group_key(gradient)
    key = (id(mesh), reduce_dims)
    group = groups.get(key)
    if group is None:
        initial = float("-inf") if infinite else 0.0
        group = _NormGroup(
            mesh,
            reduce_dims,
            torch.full((), initial, dtype=torch.float32, device=device),
        )
        groups[key] = group
    return group


def _finish_norm(
    groups: dict[tuple[int, tuple[int, ...]], _NormGroup],
    device: torch.device,
    norm_type: float,
    pp_mesh,
    stream,
) -> tuple[torch.Tensor, Any]:
    infinite = math.isinf(norm_type)
    # HCCL collectives may not honor the surrounding NPU stream context:
    # order the all_reduce after the local norm accumulation by making the
    # current stream (where the collective actually executes) wait for the
    # compute stream to finish writing the partial sums.
    local_done = torch_npu.npu.Event()
    local_done.record(stream)
    caller = torch_npu.npu.current_stream(device)
    caller.wait_event(local_done)
    with torch_npu.npu.stream(caller):
        if dist.is_initialized():
            operation = dist.ReduceOp.MAX if infinite else dist.ReduceOp.SUM
            for group in groups.values():
                for mesh_dim in group.reduce_dims:
                    dist.all_reduce(
                        group.value,
                        op=operation,
                        group=group.mesh.get_group(mesh_dim),
                    )

        initial = float("-inf") if infinite else 0.0
        total = torch.full((), initial, dtype=torch.float32, device=device)
        for group in groups.values():
            if infinite:
                total.copy_(torch.maximum(total, group.value))
            else:
                total.add_(group.value)
        if pp_mesh is not None and dist.is_initialized():
            dist.all_reduce(
                total,
                op=dist.ReduceOp.MAX if infinite else dist.ReduceOp.SUM,
                group=pp_mesh.get_group(),
            )
        if infinite:
            total.copy_(torch.where(torch.isneginf(total), torch.zeros_like(total), total))
        else:
            total = total.clamp_min(0).pow(1.0 / norm_type)
        ready = torch_npu.npu.Event()
        ready.record(stream)
    return total, ready


def _submit_gradient(
    pipeline: _NormPipeline,
    source: torch.Tensor,
    destination: torch.Tensor,
) -> tuple[TransferHandle, ...]:
    flat_source = source.view(-1)
    flat_destination = destination.view(-1)
    chunk_numel = max(1, _CHUNK_BYTES // source.element_size())
    return tuple(
        pipeline.staging.submit_h2d(
            flat_source[offset : offset + chunk_numel],
            flat_destination[offset : offset + chunk_numel],
            stream=pipeline.staging.stream,
            track=False,
        )
        for offset in range(0, source.numel(), chunk_numel)
    )


def _cached_batches(
    parameter_gradients: list[tuple[torch.Tensor, DTensor]],
    pipeline: _NormPipeline,
    groups: dict[tuple[int, tuple[int, ...]], _NormGroup],
    norm_type: float,
    cache_entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Iterator[list[_CachedNormItem]]:
    batch: list[_CachedNormItem] = []
    batch_bytes = 0
    batch_dtype: torch.dtype | None = None
    chunk_bytes = _CHUNK_BYTES
    infinite = math.isinf(norm_type)

    for parameter, gradient in parameter_gradients:
        cpu_gradient = clip_state.local_tensor(gradient).detach()
        transfer_source = cpu_gradient if cpu_gradient.is_contiguous() else cpu_gradient.contiguous()
        npu_gradient = torch.empty(
            cpu_gradient.shape,
            dtype=cpu_gradient.dtype,
            device=pipeline.device,
        )
        cache_entries.append((parameter, cpu_gradient, npu_gradient))
        if cpu_gradient.numel() == 0:
            continue

        size_bytes = cpu_gradient.numel() * cpu_gradient.element_size()
        if batch and (batch_dtype != cpu_gradient.dtype or batch_bytes + size_bytes > chunk_bytes):
            yield batch
            batch = []
            batch_bytes = 0
        group = _get_norm_group(groups, gradient, pipeline.device, infinite=infinite)
        batch.append(
            _CachedNormItem(
                npu_gradient,
                group,
                _submit_gradient(pipeline, transfer_source, npu_gradient),
            )
        )
        batch_bytes += size_bytes
        batch_dtype = cpu_gradient.dtype

    if batch:
        yield batch


def _accumulate_cached_batch(
    batch: list[_CachedNormItem],
    norm_type: float,
    stream,
) -> None:
    for item in batch:
        for handle in item.handles:
            handle.wait_on(stream)

    with torch_npu.npu.stream(stream):
        norms = torch._foreach_norm(
            [item.npu_gradient for item in batch],
            norm_type,
        )
        by_group: dict[int, tuple[_NormGroup, list[torch.Tensor]]] = {}
        for item, norm in zip(batch, norms, strict=True):
            _, values = by_group.setdefault(id(item.group), (item.group, []))
            values.append(norm.float())
        for group, values in by_group.values():
            stacked = torch.stack(values)
            if math.isinf(norm_type):
                group.value.copy_(torch.maximum(group.value, stacked.amax()))
            else:
                group.value.add_(stacked.pow(norm_type).sum())


@torch.no_grad()
def _cached_local_norm(
    parameter_gradients: list[tuple[torch.Tensor, DTensor]],
    device: torch.device,
    norm_type: float,
    pp_mesh,
) -> tuple[
    torch.Tensor,
    Any,
    list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
]:
    pipeline = _pipeline(device)
    groups: dict[tuple[int, tuple[int, ...]], _NormGroup] = {}
    cache_entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    pending: list[_CachedNormItem] | None = None

    for batch in _cached_batches(
        parameter_gradients,
        pipeline,
        groups,
        norm_type,
        cache_entries,
    ):
        if pending is not None:
            _accumulate_cached_batch(pending, norm_type, pipeline.compute_stream)
        pending = batch
    if pending is not None:
        _accumulate_cached_batch(pending, norm_type, pipeline.compute_stream)

    total, ready = _finish_norm(
        groups,
        device,
        norm_type,
        pp_mesh,
        pipeline.compute_stream,
    )
    return total, ready, cache_entries


@torch.no_grad()
def _bounded_local_norm(
    gradients: list[DTensor],
    device: torch.device,
    norm_type: float,
    pp_mesh,
) -> tuple[torch.Tensor, Any]:
    infinite = math.isinf(norm_type)
    pipeline = _pipeline(device)
    transfer_stream = pipeline.staging.stream
    compute_stream = pipeline.compute_stream
    groups: dict[tuple[int, tuple[int, ...]], _NormGroup] = {}
    pending: tuple[_NormSlot, TransferHandle, _NormGroup, int] | None = None
    slot_indices: dict[tuple[torch.dtype, int], int] = {}

    def finish(item: tuple[_NormSlot, TransferHandle, _NormGroup, int]) -> None:
        slot, handle, group, valid_numel = item
        handle.wait_on(compute_stream)
        with torch_npu.npu.stream(compute_stream):
            # Only the first ``valid_numel`` elements were written by the H2D
            # copy; the rest of the chunk-sized slot buffer holds stale data
            # from a previous chunk and must not enter the norm.
            values = slot.buffer[:valid_numel].float()
            if infinite:
                group.value.copy_(torch.maximum(group.value, values.abs().amax()))
            else:
                group.value.add_(values.abs().pow(norm_type).sum())
            done = torch_npu.npu.Event()
            done.record(compute_stream)
        slot.done = done

    chunk_bytes = _CHUNK_BYTES
    for gradient in gradients:
        group = _get_norm_group(groups, gradient, device, infinite=infinite)
        local = clip_state.local_tensor(gradient).detach()
        if local.numel() == 0:
            continue
        flat = local.contiguous().view(-1) if not local.is_contiguous() else local.view(-1)
        chunk_numel = max(1, chunk_bytes // local.element_size())
        slots = pipeline.buffers(local.dtype, chunk_numel)
        slot_key = (local.dtype, chunk_numel)
        for offset in range(0, flat.numel(), chunk_numel):
            source = flat[offset : offset + chunk_numel]
            slot_index = slot_indices.get(slot_key, 0)
            slot_indices[slot_key] = slot_index + 1
            slot = slots[slot_index % 2]
            with torch_npu.npu.stream(transfer_stream):
                if slot.done is not None:
                    transfer_stream.wait_event(slot.done)
            handle = pipeline.staging.submit_h2d(
                source,
                slot.buffer[: source.numel()],
                stream=transfer_stream,
                track=False,
            )
            if pending is not None:
                finish(pending)
            pending = (slot, handle, group, source.numel())
    if pending is not None:
        finish(pending)

    return _finish_norm(groups, device, norm_type, pp_mesh, compute_stream)


def _normalize_norm_type(norm_type: float | str) -> float:
    if isinstance(norm_type, str):
        if norm_type.lower() != "inf":
            raise ValueError(f"unsupported norm_type {norm_type!r}")
        return float("inf")
    value = float(norm_type)
    if value <= 0 or math.isnan(value):
        raise ValueError(f"norm_type must be positive, got {value}")
    return value


@functools.wraps(_ORIGINAL_CLIP_GRAD_NORM)
@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh=None,
    ep_enabled: bool = False,
) -> torch.Tensor:
    parameter_list = [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    parameter_gradients = [(parameter, parameter.grad) for parameter in parameter_list if parameter.grad is not None]
    gradients = [gradient for _, gradient in parameter_gradients]
    cpu_gradients = [gradient for gradient in gradients if clip_state.local_tensor(gradient).device.type == "cpu"]
    if not cpu_gradients:
        return _ORIGINAL_CLIP_GRAD_NORM(
            parameter_list,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            ep_enabled,
        )
    if len(cpu_gradients) != len(gradients) or any(not isinstance(gradient, DTensor) for gradient in gradients):
        raise TypeError("CPU gradient clipping requires only CPU DTensor gradients")
    dtensor_gradients = cast("list[DTensor]", gradients)
    dtensor_parameter_gradients = cast(
        "list[tuple[torch.Tensor, DTensor]]",
        parameter_gradients,
    )

    norm_type = _normalize_norm_type(norm_type)
    device = _compute_device(parameter_list)
    if device is None:
        if any(
            isinstance(gradient, DTensor) and gradient.device_mesh.device_type == "npu" for gradient in cpu_gradients
        ):
            raise RuntimeError("CPU-canonical gradients use an NPU mesh, but no NPU compute device is available")
        return _ORIGINAL_CLIP_GRAD_NORM(
            parameter_list,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            ep_enabled,
        )
    if device.type != "npu":
        return _ORIGINAL_CLIP_GRAD_NORM(
            parameter_list,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
            ep_enabled,
        )

    channel = clip_state.get_active_channel()
    cache_entries: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if channel is not None and channel.has_consumer():
        total_norm, norm_ready, cache_entries = _cached_local_norm(
            dtensor_parameter_gradients,
            device,
            norm_type,
            pp_mesh,
        )
    else:
        total_norm, norm_ready = _bounded_local_norm(
            dtensor_gradients,
            device,
            norm_type,
            pp_mesh,
        )
    caller_stream = torch_npu.npu.current_stream(device)
    caller_stream.wait_event(norm_ready)
    if error_if_nonfinite and not bool(torch.isfinite(total_norm).item()):
        raise RuntimeError(f"The total norm of order {norm_type} for gradients is non-finite")

    # Compute the clip coefficient on the caller's stream: the optimizers
    # consume it on this same stream, so stream FIFO ordering makes the value
    # visible without cross-stream event plumbing. The recorded event still
    # protects any consumer that runs on a different stream.
    coefficient = (torch.as_tensor(float(max_norm), dtype=torch.float32, device=device) / (total_norm + 1e-6)).clamp(
        max=1.0
    )
    coefficient_ready = torch_npu.npu.Event()
    coefficient_ready.record(caller_stream)

    if cache_entries:
        if channel is None:
            raise RuntimeError("cache entries without a consumer channel")
        channel.publish(cache_entries, coefficient, coefficient_ready)
    else:
        value = float(coefficient.item())
        if value != 1.0:
            torch._foreach_mul_(
                [clip_state.local_tensor(gradient) for gradient in cpu_gradients],
                value,
            )
    return total_norm


def install() -> None:
    """Install the CPU-gradient clip replacement once per process.

    Called explicitly by the CPU-offload optimizer container
    (:class:`torchtitan_npu.override.common.optimizer.CpuOffloadOptimizersContainer`)
    when the user selects the ``cpu_offload`` override; importing this module
    alone has no side effects. Without the installation, TorchTitan's
    upstream ``clip_grad_norm_`` runs unchanged.
    """
    if getattr(distributed_utils, _PATCHED, False):
        return
    distributed_utils.clip_grad_norm_ = clip_grad_norm_
    setattr(distributed_utils, _PATCHED, True)
