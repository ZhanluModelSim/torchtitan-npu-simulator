# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

from itertools import count

import torch

# TORCHTITAN-NPU MOD: resolve the local explicit-session API, not a global singleton.
# Remove when: this backend is moved to a shared upstream swap package.
from . import swap_api


def base_check_fn(tensor) -> bool:
    """
    Basic check to determine if a tensor is eligible for offloading.
    - Skip Parameters and their views.
    - Skip empty storage tensors.
    - Skip internally overlapping tensors that cannot be restored with copy_.
    """
    expected_bytes = tensor.numel() * tensor.element_size()
    actual_bytes = tensor.untyped_storage().size()
    if not isinstance(tensor, torch.Tensor):
        return False
    if isinstance(tensor._base, torch.nn.parameter.Parameter) or isinstance(tensor, torch.nn.parameter.Parameter):
        return False
    if tensor.storage().size() <= 0:
        return False
    if torch._debug_has_internal_overlap(tensor) != 0:
        return False
    if tensor.nbytes <= 1 * 1024**2:
        return False
    if tensor._base is not None:
        return False
    if not tensor.is_contiguous():
        return False
    if tensor.storage_offset() != 0:
        return False
    return actual_bytes == expected_bytes


def make_pack_fn(tensor_name, do_d2h=False, custom_check_fn=None):
    """Build a pack_hook for saved_tensors_hooks.

    Registers each eligible saved tensor under a unique name and returns that
    name as the packed value. When do_d2h is True the pack itself issues the
    D2H offload; otherwise the offload timing is left to the application layer
    (e.g. the block-level cross-layer D2H loop).
    """
    saved_tensor_index = count()

    def pack_fn(tensor):
        if custom_check_fn:
            if not custom_check_fn(tensor):
                return tensor
        else:
            if not base_check_fn(tensor):
                return tensor

        index = next(saved_tensor_index)
        _tensor_name = f"{tensor_name}.saved_tensor{index}"

        swap_api.register_tensor(tensor, _tensor_name)
        if do_d2h:
            swap_api.execute(_tensor_name, "D2H")
        return _tensor_name

    return pack_fn


def unpack_fn(tensor_or_name, do_h2d=False):
    """Unpack_hook for saved_tensors_hooks.

    If the packed value is a name, restore the tensor: optionally issue the H2D
    (do_h2d=True for the self-contained mode; left to the application-level
    prefetch hook otherwise), then wait for the copy and pop it from the
    registry. If the packed value is already a tensor, return it unchanged.
    """
    if isinstance(tensor_or_name, str):
        if do_h2d:
            swap_api.execute(tensor_or_name, "H2D")
        swap_api.execute(tensor_or_name, "WAIT")
        tensor_or_name = swap_api.pop_tensor_by_name(tensor_or_name)
    return tensor_or_name
