# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

from dataclasses import dataclass

import torch
import torch_npu


def _swap_d2h(tensor, stream):
    forward_event = torch_npu.npu.Event()
    forward_event.record()
    # prepare cpu tensor
    tensor_cpu = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu")
    with torch.no_grad(), torch_npu.npu.stream(stream):
        stream.wait_event(forward_event)
        # begin to swap tensor from npu to cpu
        tensor_cpu.untyped_storage().copy_(tensor.untyped_storage(), non_blocking=True)
        swap_out_event = torch_npu.npu.Event()
        swap_out_event.record()
    return tensor_cpu, swap_out_event


def _swap_h2d(tensor, tensor_cpu, swap_out_event, stream):
    backward_event = torch_npu.npu.Event()
    backward_event.record()
    # prepare npu tensor
    tensor.untyped_storage().resize_(tensor_cpu.untyped_storage().size())
    with torch.no_grad(), torch_npu.npu.stream(stream):
        stream.wait_event(backward_event)
        stream.wait_event(swap_out_event)
        # begin to swap tensor from cpu to npu
        tensor.untyped_storage().copy_(tensor_cpu.untyped_storage(), non_blocking=True)
        swap_in_event = torch_npu.npu.Event()
        swap_in_event.record()
    return swap_in_event


@dataclass
class SwapMetadata:
    tensor_npu: torch.Tensor | None = None
    tensor_cpu: torch.Tensor | None = None
    swap_out_event: torch_npu.npu.Event | None = None
    swap_in_event: torch_npu.npu.Event | None = None


class TensorSwapping:
    # cache last forward swap-metadata
    swapping_cache = {}

    offload_stream = None
    prefetch_stream = None

    grad_store = {}

    @classmethod
    def swap_out(cls, name, tensor):
        if cls.swapping_cache.get(name) is not None:
            pre_metadata = cls.swapping_cache.pop(name)
            pre_swap_out_event = pre_metadata.swap_out_event
            pre_swap_in_event = pre_metadata.swap_in_event
            if pre_swap_out_event is not None and pre_swap_in_event is None:
                # free last swapping tensor, cross layers or micro-batch
                torch_npu.npu.current_stream().wait_event(pre_swap_out_event)
                pre_metadata.tensor_npu.untyped_storage().resize_(0)

        if cls.offload_stream is None:
            cls.offload_stream = torch_npu.npu.Stream(device=torch_npu.npu.current_device())

        tensor_cpu, swap_out_event = _swap_d2h(tensor, cls.offload_stream)
        metadata = SwapMetadata(tensor, tensor_cpu, swap_out_event)
        cls.swapping_cache[name] = metadata
        return metadata

    @classmethod
    def start_swapping_out(cls, tensor):
        if cls.offload_stream is None:
            cls.offload_stream = torch_npu.npu.Stream(device=torch_npu.npu.current_device())

        tensor_cpu, swap_out_event = _swap_d2h(tensor, cls.offload_stream)
        return SwapMetadata(tensor, tensor_cpu, swap_out_event)

    @classmethod
    def end_swapping_out(cls, metadata):
        torch_npu.npu.current_stream().wait_event(metadata.swap_out_event)
        metadata.tensor_npu.untyped_storage().resize_(0)

    @classmethod
    def start_swapping_in(cls, name, tensors):
        if cls.prefetch_stream is None:
            cls.prefetch_stream = torch_npu.npu.Stream(device=torch_npu.npu.current_device())

        metadata = tensors[name]
        swap_in_event = _swap_h2d(
            metadata.tensor_npu, metadata.tensor_cpu, metadata.swap_out_event, cls.prefetch_stream
        )
        tensors[name].swap_in_event = swap_in_event

    @classmethod
    def end_swapping_in(cls, name, tensors):
        metadata = tensors.pop(name)
        torch_npu.npu.current_stream().wait_event(metadata.swap_in_event)
        metadata.tensor_cpu.untyped_storage().resize_(0)
        metadata.swap_out_event = None
        metadata.swap_in_event = None

    @classmethod
    def get_swap_tensor(cls, name, tensors):
        metadata = tensors.pop(name)
        torch_npu.npu.current_stream().wait_event(metadata.swap_in_event)
        metadata.tensor_cpu.untyped_storage().resize_(0)
        metadata.swap_out_event = None
        metadata.swap_in_event = None
        return metadata.tensor_npu

    @classmethod
    def set_grad_tensor(cls, name, tensor):
        cls.grad_store[name] = tensor

    @classmethod
    def get_grad_tensor(cls, name):
        return cls.grad_store.pop(name)

    @classmethod
    def reset_cache(cls):
        cls.swapping_cache = {}

        cls.offload_stream = None
        cls.prefetch_stream = None

        cls.grad_store = {}
