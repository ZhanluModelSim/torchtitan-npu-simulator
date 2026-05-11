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
from functools import wraps
from typing import Any, cast

import torch
import torch_npu
import torchtitan
import torchtitan.components.optimizer
from torch.distributed._tensor import DTensor

logger = logging.getLogger(__name__)

_original_build_optimizers = getattr(
    torchtitan.components.optimizer, "build_optimizers", None
)

OPTIMIZER_STATE_KEYS = ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]


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


def is_swap_tensor(tensor: torch.Tensor) -> bool:
    return hasattr(tensor, "swap_tensor") and tensor.swap_tensor


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
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        for p in group["params"]:
            if p.grad is None:
                continue
            state = _initialize_state(self, p, group["amsgrad"])

            if "step" not in state:
                state["step"] = torch.tensor(1, dtype=torch.int64, device=p.device)
            else:
                state["step"] += 1

            kernel_args = {
                "params": [p],
                "grads": [p.grad],
                "exp_avgs": [state["exp_avg"]],
                "exp_avg_sqs": [state["exp_avg_sq"]],
                "max_exp_avg_sqs": [state["max_exp_avg_sq"]]
                if group["amsgrad"]
                else [],
                "step_tensors": [state["step"]],
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


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(x) for x in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.detach()
    return obj


def _process_state_tensor(state, k):
    local_t = unwrap_dtensor(state[k])
    cpu_t = local_t.cpu().clone()
    if hasattr(cpu_t, "swap_tensor"):
        object.__delattr__(cpu_t, "swap_tensor")
    state[k] = wrap_like_param(cpu_t, state[k])


def _save_original_states(self):
    """Save original states before conversion and process state tensors."""
    original_states = {}
    if not hasattr(self, "virtual_allocator"):
        return original_states
    for p, state in self.state.items():
        original_states[p] = {k: v for k, v in state.items()}
        for k in OPTIMIZER_STATE_KEYS:
            if k in state and isinstance(state[k], torch.Tensor):
                _process_state_tensor(state, k)
    return original_states


def _restore_original_states(self, original_states):
    """Restore original states after state_dict save."""
    if not original_states:
        return
    for p, o_state in original_states.items():
        for k, v in o_state.items():
            if v is not None:
                self.state[p][k] = v


def patched_state_dict(self) -> dict[str, Any]:
    original_states = _save_original_states(self)
    sd = self._original_state_dict()  # type: ignore[assignment]
    sd = cast(dict[str, Any], sanitize(sd))
    _restore_original_states(self, original_states)
    return sd


def _restore_single_tensor(state, k, p, allocator):
    if k in state and state[k] is not None:
        local_t = unwrap_dtensor(state[k])
        if not is_swap_tensor(local_t):
            swap_t = allocator.create(local_t)
            swap_t.copy_(local_t)
            state[k] = wrap_like_param(swap_t, state[k])


def _process_step_device(state, p):
    if "step" in state and isinstance(state["step"], torch.Tensor):
        if state["step"].device.type == "cpu":
            state["step"] = state["step"].to(p.device)


def _update_param_states(optimizer, p):
    """Update all optimizer state tensors for a parameter."""
    state = optimizer.state.get(p)
    if not state:
        return
    for k in OPTIMIZER_STATE_KEYS:
        _restore_single_tensor(state, k, p, optimizer.virtual_allocator)
    _process_step_device(state, p)


def virtual_optimizer_replace(optimizer):
    if not hasattr(optimizer, "virtual_allocator"):
        return
    for group in optimizer.param_groups:
        for p in group["params"]:
            _update_param_states(optimizer, p)


def virtual_optimizer_step(self, closure=None):
    if not hasattr(self, "virtual_allocator"):
        pp_rank, pp_size, virtual_size = self._allocator_config
        self.virtual_allocator = VirtualAllocator(pp_rank, pp_size, virtual_size)

    self.print_swap_flag = not hasattr(self, "print_swap_flag")
    with torch.no_grad():
        loss = virtual_optimizer_step_impl(self, closure)
    return loss


def swap_tensor_copy_wrapper(func):
    def wrapped(*args, **kwargs):
        dst, src = args[0], args[1]
        non_blocking = kwargs.get("non_blocking", False)

        if dst is src:
            return dst

        dst_swap = is_swap_tensor(dst)
        src_swap = is_swap_tensor(src)
        dst_dev = dst.device
        src_dev = src.device

        if dst_swap or src_swap:
            if dst.shape != src.shape:
                raise RuntimeError(
                    f"Shape mismatch in swap tensor copy: {dst.shape} vs {src.shape}"
                )
            if dst_dev.type == "cpu" and src_dev.type == "cpu":
                func(dst, src, non_blocking=non_blocking)

            elif dst_dev.type == "cpu" and (src_dev.type == "npu" or src_swap):
                src_cpu = src.to(dst_dev, non_blocking=non_blocking)
                func(dst, src_cpu, non_blocking=non_blocking)

            elif (dst_dev.type == "npu" or dst_swap) and src_dev.type == "cpu":
                src_npu = src.to(dst_dev, non_blocking=non_blocking)
                dst.fill_(1).mul_(src_npu)

            elif (dst_dev.type == "npu" or dst_swap) and (
                src_dev.type == "npu" or src_swap
            ):
                if dst_dev == src_dev:
                    dst.fill_(1).mul_(src)
                else:
                    src_npu = src.to(dst_dev, non_blocking=non_blocking)
                    dst.fill_(1).mul_(src_npu)
        else:
            func(*args, **kwargs)

        return dst

    return wrapped


def swap_tensor_func_wrapper(org_func, func_type):
    def wrapped(*args, **kwargs):
        if is_swap_tensor(args[0]):
            if func_type == "detach":
                detach = org_func(*args, **kwargs)
                detach.swap_tensor = True
                detach.data.swap_tensor = True
                return detach
            src = torch.empty_like(args[0])
            src.copy_(args[0])
            if func_type == "cpu":
                return src.cpu()
            elif func_type == "clone":
                return src
            else:
                raise ValueError(f"func_type {func_type} not supported")
        else:
            return org_func(*args, **kwargs)

    return wrapped


def _make_patched_load(orig_load):
    @wraps(orig_load)  # type: ignore[assignment]
    def patched_load(self, state_dict):
        self._original_load_state_dict(state_dict)  # type: ignore[assignment]
        virtual_optimizer_replace(self)

    return patched_load


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

    torch.Tensor.copy_ = swap_tensor_copy_wrapper(torch.Tensor.copy_)
    torch.Tensor.cpu = swap_tensor_func_wrapper(torch.Tensor.cpu, "cpu")
    torch.Tensor.clone = swap_tensor_func_wrapper(torch.Tensor.clone, "clone")
    torch.Tensor.detach = swap_tensor_func_wrapper(torch.Tensor.detach, "detach")

    for cls in [torch.optim.AdamW, torch.optim.Adam]:
        if not hasattr(cls, "_original_state_dict"):
            cls._original_state_dict = cls.state_dict  # type: ignore[assignment]
            cls.state_dict = patched_state_dict  # type: ignore[assignment]
            cls.step = virtual_optimizer_step  # type: ignore[assignment]

            cls._original_load_state_dict = cls.load_state_dict  # type: ignore[assignment]
            cls.load_state_dict = _make_patched_load(cls.load_state_dict)  # type: ignore[assignment]

    optimizers = _original_build_optimizers(
        model_parts, optimizer_config, parallel_dims, ft_manager
    )

    # Simplified PP info (should be retrieved from parallel_dims in real scenario)
    pp_rank, pp_size = 0, 1

    for opt in optimizers:
        opt._allocator_config = (pp_rank, pp_size, virtual_size)

    return optimizers


if _original_build_optimizers is not None:
    torchtitan.components.optimizer.build_optimizers = (
        build_optimizers_with_virtual_optimizer  # type: ignore[assignment]
    )
else:
    logger.warning(
        "Skipping virtual optimizer patch: build_optimizers is not available "
        "in current torchtitan.components.optimizer API."
    )
