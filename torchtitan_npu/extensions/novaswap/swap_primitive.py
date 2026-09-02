# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

import logging
from dataclasses import dataclass

import torch
import torch_npu

# TORCHTITAN-NPU MOD: use the plugin-local process pool instead of global state.
# Remove when: this backend is moved to a shared upstream swap package.
from .storage.pinned_cpu_memory_pool import PinnedCpuStorage

logger = logging.getLogger(__name__)


@dataclass
class SwapHandle:
    """Handle returned by asynchronous swap primitives."""

    tensor_npu: torch.Tensor | None = None
    tensor_cpu: torch.Tensor | None = None
    swap_event: torch_npu.npu.Event | None = None
    is_completed: bool = False
    use_storage_copy: bool = False
    owner: str | None = None
    transfer: str = "D2H"
    device_waited: bool = False


def _logical_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def validate_tensor_for_swap(tensor: torch.Tensor) -> None:
    """Validate the storage contract before scheduling an asynchronous copy."""
    expected_bytes = _logical_nbytes(tensor)
    actual_bytes = tensor.untyped_storage().size()
    if not (tensor.is_contiguous() and tensor.storage_offset() == 0 and actual_bytes == expected_bytes):
        raise ValueError(
            "Unified swap D2H only supports contiguous tensors with compact storage; "
            f"got shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}, "
            f"storage_offset={tensor.storage_offset()}, storage_bytes={actual_bytes}, "
            f"expected_bytes={expected_bytes}. This tensor is non-contiguous or a view."
        )


def _logical_storage(tensor: torch.Tensor):
    """Return the byte storage range occupied by a contiguous tensor view."""
    nbytes = _logical_nbytes(tensor)
    storage = tensor.untyped_storage()
    offset = tensor.storage_offset() * tensor.element_size()
    if offset == 0 and storage.size() == nbytes:
        return storage
    return storage[offset : offset + nbytes]


def _copy_logical_storage(source: torch.Tensor, destination: torch.Tensor) -> None:
    _logical_storage(destination).copy_(_logical_storage(source), non_blocking=True)


def d2h(
    tensor: torch.Tensor,
    stream: torch_npu.npu.Stream,
    owner: str | None = None,
) -> SwapHandle:
    """Asynchronously copy an NPU tensor to pinned CPU memory."""
    validate_tensor_for_swap(tensor)

    tensor_cpu = PinnedCpuStorage.allocate(_logical_nbytes(tensor), owner=owner)

    try:
        current_event = torch_npu.npu.Event()
        current_event.record()

        with torch.no_grad(), torch_npu.npu.stream(stream):
            stream.wait_event(current_event)
            _copy_logical_storage(tensor, tensor_cpu)
            swap_event = torch_npu.npu.Event()
            swap_event.record()
    except Exception:
        logger.warning(
            "D2H swap submission failed; synchronizing the stream and releasing "
            "the pinned CPU buffer before re-raising.",
            exc_info=True,
        )
        stream.synchronize()
        PinnedCpuStorage.free(tensor_cpu, owner=owner)
        raise

    return SwapHandle(
        tensor_npu=tensor,
        tensor_cpu=tensor_cpu,
        swap_event=swap_event,
        use_storage_copy=True,
        owner=owner,
        transfer="D2H",
    )


def h2d(handle: SwapHandle, stream: torch_npu.npu.Stream) -> SwapHandle:
    """Asynchronously copy pinned CPU storage back to the original NPU tensor."""
    tensor_npu = handle.tensor_npu
    tensor_cpu = handle.tensor_cpu
    if tensor_npu is None or tensor_cpu is None:
        raise RuntimeError("H2D requires both NPU and CPU tensors in SwapHandle")

    current_event = torch_npu.npu.Event()
    current_event.record()

    # Dtype and layout cannot distinguish raw buffers from logical uint8 tensors.
    use_storage_copy = handle.use_storage_copy
    if use_storage_copy:
        cpu_bytes = _logical_nbytes(tensor_cpu)
        npu_bytes = _logical_nbytes(tensor_npu)
        if cpu_bytes != npu_bytes:
            raise ValueError(f"Raw H2D storage size {cpu_bytes} does not match NPU tensor size {npu_bytes}.")
        # A pooled byte buffer may be a slice of a larger slab, so copy its
        # logical byte range instead of comparing the backing storage sizes.
        if tensor_npu.untyped_storage().size() != npu_bytes:
            tensor_npu.untyped_storage().resize_(npu_bytes)
    else:
        if tensor_npu.shape != tensor_cpu.shape or tensor_npu.dtype != tensor_cpu.dtype:
            raise ValueError("Logical H2D copy requires matching NPU and CPU tensor shape and dtype.")
        validate_tensor_for_swap(tensor_npu)

    try:
        with torch.no_grad(), torch_npu.npu.stream(stream):
            stream.wait_event(current_event)
            if handle.swap_event is not None:
                stream.wait_event(handle.swap_event)
            if use_storage_copy:
                _copy_logical_storage(tensor_cpu, tensor_npu)
            else:
                tensor_npu.copy_(tensor_cpu, non_blocking=True)
            swap_event = torch_npu.npu.Event()
            swap_event.record()
    except Exception:
        stream.synchronize()
        raise

    result = SwapHandle(
        tensor_npu=tensor_npu,
        tensor_cpu=tensor_cpu,
        swap_event=swap_event,
        use_storage_copy=use_storage_copy,
        owner=handle.owner,
        transfer="H2D",
    )
    handle.tensor_cpu = None  # Transfer CPU-buffer ownership to the H2D handle.
    return result
