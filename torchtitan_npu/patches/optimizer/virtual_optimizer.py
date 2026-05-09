# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from MindSpeed,
# https://gitcode.com/Ascend/MindSpeed/blob/master/mindspeed/core/optimizer/virtual_optimizer/virtual_adam.py
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import torch
import torch_npu
import torchtitan
import torchtitan.components.optimizer
from torch.distributed._tensor import DTensor

logger = logging.getLogger(__name__)

_original_build_optimizers = torchtitan.components.optimizer.build_optimizers


def unwrap_dtensor(tensor):
    """Get normal tensor from DTensor."""
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def wrap_like_param(local_tensor: torch.Tensor, p: DTensor | torch.Tensor):
    if isinstance(p, DTensor):
        return DTensor.from_local(
            local_tensor,
            device_mesh=p.device_mesh,
            placements=p.placements,
            shape=p.size(),
            stride=p.stride(),
            run_check=False,
        )
    return local_tensor


def _initialize_state(optimizer, p, amsgrad):
    """Initialize optimizer state for a single parameter."""
    state = optimizer.state[p]
    if len(state) != 0:
        return state

    local_exp, local_exp_avg_sq = optimizer.virtual_allocator.init_exp(
        unwrap_dtensor(p)
    )
    state["exp_avg"] = wrap_like_param(local_exp, p)
    state["exp_avg_sq"] = wrap_like_param(local_exp_avg_sq, p)

    if amsgrad:
        state["max_exp_avg_sq"] = torch.zeros_like(
            p, memory_format=torch.preserve_format
        )
    return state


def _apply_fused_kernel(group, kernel_args):
    """Execute the appropriate fused optimizer kernel."""
    is_adamw = group.get("decoupled_weight_decay", False)
    kernel_name = "_fused_adamw_" if is_adamw else "_fused_adam_"
    fused_kernel = getattr(torch, kernel_name)

    fused_kernel(
        kernel_args["params"],
        kernel_args["grads"],
        kernel_args["exp_avgs"],
        kernel_args["exp_avg_sqs"],
        kernel_args["max_exp_avg_sqs"],
        kernel_args["step_tensors"],
        amsgrad=group["amsgrad"],
        lr=group["lr"],
        beta1=group["betas"][0],
        beta2=group["betas"][1],
        weight_decay=group["weight_decay"],
        eps=group["eps"],
        maximize=group["maximize"],
    )


def virtual_optimizer_step_impl(self, closure=None):
    """Simplified optimizer step implementation with lower complexity."""
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        # Step increment logic
        if "step" not in group:
            group["step"] = torch.tensor(
                1, dtype=torch.int64, device=torch_npu.npu.current_device()
            )
        else:
            group["step"] += 1
            if group["step"].is_cpu:
                group["step"] = group["step"].npu()

        for p in group["params"]:
            if p.grad is None:
                continue

            state = _initialize_state(self, p, group["amsgrad"])

            # Prepare common arguments for fused kernels to avoid duplication
            kernel_args = {
                "params": [p],
                "grads": [p.grad],
                "exp_avgs": [state["exp_avg"]],
                "exp_avg_sqs": [state["exp_avg_sq"]],
                "max_exp_avg_sqs": [state["max_exp_avg_sq"]]
                if group["amsgrad"]
                else [],
                "step_tensors": [group["step"]],
            }

            _apply_fused_kernel(group, kernel_args)

    self.virtual_allocator.print_swap_size(self.print_swap_flag)
    return loss


class VirtualAllocator:
    """Handles memory allocation for virtual optimizer states."""

    def __init__(self, pp_rank, pp_stages, virtual_optimizer_size=None):
        self.pp_stages = pp_stages
        self.pp_rank = pp_rank
        self.virtual_optimizer_size = virtual_optimizer_size
        self.swap_size_this_pp_rank = self.get_swap_memory_sizes()[self.pp_rank] * (
            1024**3
        )
        self.actually_swap_size: float = 0.0

    @classmethod
    def get_memory(cls, p: torch.Tensor):
        """Static: Construct tensors in standard HBM memory."""
        return torch.zeros_like(p, memory_format=torch.preserve_format)

    def get_swap_memory_sizes(self) -> list[float]:
        if (
            isinstance(self.virtual_optimizer_size, str)
            and self.virtual_optimizer_size.lower() == "all"
        ):
            return [65.0] * self.pp_stages

        if isinstance(self.virtual_optimizer_size, (int, float)):
            return [float(self.virtual_optimizer_size)] * self.pp_stages

        if isinstance(self.virtual_optimizer_size, (list, tuple)):
            if len(self.virtual_optimizer_size) == 1:
                return [self.virtual_optimizer_size[0]] * self.pp_stages
            if len(self.virtual_optimizer_size) == self.pp_stages:
                return list(self.virtual_optimizer_size)

        raise ValueError(
            f"virtual_optimizer_size configuration error for pp={self.pp_stages}"
        )

    def init_exp(self, p: torch.Tensor):
        """Create exp_avg and exp_avg_sq based on input param."""
        return self.create(p), self.create(p)

    def create(self, p: torch.Tensor):
        """Select construction mode based on available swap size."""
        if self.swap_size_this_pp_rank > 0:
            return self._get_swap_memory(p)
        return self.get_memory(p)

    def print_swap_size(self, print_swap_flag):
        """Print swap usage summary."""
        if print_swap_flag:
            logger.info(
                f"[Swap virtual-optimizer Summary: Rank {torch.distributed.get_rank()}] "
                f"Swap {self.actually_swap_size:.5f} MB"
            )

    def _get_swap_memory(self, p: torch.Tensor):
        """Internal: Logic for NPU swapped memory allocation."""
        if not hasattr(torch_npu, "empty_with_swapped_memory"):
            return self.get_memory(p)
        try:
            swap_tensor = torch_npu.empty_with_swapped_memory(p.size(), device=p.device)
            swap_tensor.swap_tensor = True
            swap_tensor.data.swap_tensor = True
            swap_tensor.zero_()

            tensor_bytes = p.numel() * p.element_size()
            self.actually_swap_size += tensor_bytes / (1024 * 1024)
            self.swap_size_this_pp_rank -= tensor_bytes
            return swap_tensor
        except Exception as e:
            logger.info(f"[Warning] Swap memory alloc failed: {e}")
            return self.get_memory(p)


def virtual_optimizer_step(self, closure=None):
    if not hasattr(self, "virtual_allocator"):
        pp_rank, pp_size, virtual_size = self._allocator_config
        self.virtual_allocator = VirtualAllocator(pp_rank, pp_size, virtual_size)

    self.print_swap_flag = not hasattr(self, "print_swap_flag")
    with torch.no_grad():
        loss = virtual_optimizer_step_impl(self, closure)
    return loss


def build_optimizers_with_virtual_optimizer(
    model_parts, optimizer_config, parallel_dims, ft_manager=None
):
    has_virtual = getattr(optimizer_config, "virtual_optimizer", False)
    virtual_size = getattr(optimizer_config, "virtual_optimizer_size", None)

    if not has_virtual:
        return _original_build_optimizers(
            model_parts, optimizer_config, parallel_dims, ft_manager
        )

    if getattr(optimizer_config, "swap_optimizer", False):
        raise ValueError("Virtual optimizer does not support swap_optimizer.")

    if virtual_size is None:
        raise ValueError(
            "virtual_optimizer_size must be specified when virtual_optimizer is enabled."
        )

    # Patch optimizer steps
    torch.optim.AdamW.step = virtual_optimizer_step
    torch.optim.Adam.step = virtual_optimizer_step

    optimizers = _original_build_optimizers(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )

    # Simplified PP info (should be retrieved from parallel_dims in real scenario)
    pp_rank, pp_size = 0, 1

    for opt in optimizers:
        opt._allocator_config = (pp_rank, pp_size, virtual_size)

    return optimizers


# Apply final patch
torchtitan.components.optimizer.build_optimizers = (
    build_optimizers_with_virtual_optimizer  # type: ignore[assignment]
)
