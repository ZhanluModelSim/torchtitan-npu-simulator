# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
from torch.distributed.tensor import DTensor
from torchtitan.tools.utils import get_device_info

logger = logging.getLogger("torchtitan")


def unwrap_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Unwrap DTensor to local tensor."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def wrap_like_param(swap_tensor: torch.Tensor, param: torch.Tensor):
    """Wrap swap tensor to match param's distributed layout."""
    return swap_tensor


def is_swap_device(device: torch.device) -> bool:
    """Check if device is swap (CPU)."""
    return device.type == "cpu"


# State keys that need virtualization
ADAMW_STATE_KEYS = ["exp_avg", "exp_avg_sq"]
MUON_STATE_KEYS = ["momentum_buffer"]

ALL_VIRTUAL_KEYS = ADAMW_STATE_KEYS + MUON_STATE_KEYS


def has_virtual_state(state: dict) -> bool:
    """Check if state has any virtualized keys."""
    return any(key in state for key in ALL_VIRTUAL_KEYS)


class MuonVirtualAllocator:
    """Allocates optimizer states in swap memory (CPU) instead of GPU."""

    def __init__(
        self,
        pp_rank: int = 0,
        pp_size: int = 1,
    ):
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        device_info = get_device_info()
        self._device_module = device_info[1]
        self.device = torch.device(  # pyrefly: ignore[read-only]
            f"{device_info[0]}:{self._device_module.current_device()}"
        )
        self.actually_swap_size = 0
        self.theoretical_swap_size = 0

    def copy2swap(self, tensor: torch.Tensor):
        if isinstance(tensor, DTensor):
            tensor = tensor.to_local()
        tensor_bytes = tensor.numel() * tensor.element_size()
        self.actually_swap_size += tensor_bytes
        swap_tensor = tensor.to(device="cpu", non_blocking=True)
        return swap_tensor.detach()

    def copy2device(
        self,
        swap_tensor: torch.Tensor,
        ref_tensor: torch.Tensor = None,  # pyrefly: ignore[bad-function-definition]
        global_shape: torch.Size = None,  # pyrefly: ignore[bad-function-definition]
    ):
        device_tensor = swap_tensor.to(self.device, non_blocking=True)
        if ref_tensor is not None and isinstance(ref_tensor, DTensor):
            # For uneven sharding, shape and stride must both be provided
            # to avoid incorrect global shape inference from local_shape * mesh_size
            kwargs = dict(
                device_mesh=ref_tensor.device_mesh,
                placements=ref_tensor.placements,
                run_check=False,
            )
            if global_shape is not None:
                kwargs["shape"] = global_shape  # pyrefly: ignore[bad-typed-dict-key]
                kwargs[
                    "stride"
                ] = ref_tensor.stride()  # pyrefly: ignore[bad-typed-dict-key]
            device_tensor = DTensor.from_local(device_tensor, **kwargs)
        return device_tensor

    def compute_theoretical_swap_size(
        self, optimizers, muon_state_keys, adamw_state_keys
    ):
        for optim_idx, optim in enumerate(optimizers):
            keys = muon_state_keys if optim_idx == 0 else adamw_state_keys
            self._accumulate_param_bytes(optim, keys)

    def print_swap_summary(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        actual_mb = self.actually_swap_size / (1024 * 1024)
        theoretical_mb = self.theoretical_swap_size / (1024 * 1024)
        logger.info(
            f"[VirtualAllocator] Rank {rank} swap summary: "
            f"theoretical_offload={theoretical_mb:.2f}MB, "
            f"actual_offload={actual_mb:.2f}MB, "
            f"ratio={actual_mb / theoretical_mb * 100:.1f}%"
            if theoretical_mb > 0
            else f"[VirtualAllocator] Rank {rank} swap summary: "
            f"theoretical_offload=0MB, actual_offload={actual_mb:.2f}MB"
        )

    def _accumulate_state_bytes(self, state, keys):
        for key in keys:
            if key in state and state[key] is not None:
                t = state[key]
                if isinstance(t, DTensor):
                    t = t.to_local()
                self.theoretical_swap_size += t.numel() * t.element_size()

    def _accumulate_param_bytes(self, optim, keys):
        for group in optim.param_groups:
            for p in group["params"]:
                state = optim.state.get(p, {})
                if state:
                    self._accumulate_state_bytes(state, keys)
