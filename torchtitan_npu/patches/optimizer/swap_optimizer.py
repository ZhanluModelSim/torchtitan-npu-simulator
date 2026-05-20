# Adapted from
# https://gitcode.com/Ascend/MindSpeed/blob/master/mindspeed/core/optimizer/swap_optimizer/swap_optimizer.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
from dataclasses import dataclass, fields, replace
from typing import ClassVar, TypeVar

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.optim.optimizer import _use_grad_for_differentiable
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.utils import get_device_info


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Optimizer)


def _check_build_kwargs(config, kwargs):
    config_fields = {f.name for f in fields(config)}
    overlap = config_fields & kwargs.keys()
    if overlap:
        raise ValueError(
            f"build() kwargs {overlap} overlap with config fields. "
            "Put these values in the Config, not in build() kwargs."
        )


def _base_optimizer_config(config) -> OptimizersContainer.Config:
    return OptimizersContainer.Config(
        name=config.name,
        lr=config.lr,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        implementation=config.implementation,
    )


def _swap_config_from(config) -> "SwapOptimizersContainer.Config":
    return SwapOptimizersContainer.Config(
        name=config.name,
        lr=config.lr,
        beta1=config.beta1,
        beta2=config.beta2,
        eps=config.eps,
        weight_decay=config.weight_decay,
        implementation=config.implementation,
        swap_optimizer=config.swap_optimizer,
        swap_optimizer_times=config.swap_optimizer_times,
    )


def get_torch_device():
    return get_device_info()[1]


def unwrap_dtensor(tensor):
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


class SwapOptimizersContainer(OptimizersContainer):
    """A container for optimizers which can be swapped between host and device to save memory during training.

    It will offload the optimizer states to the host (CPU) during the forward and backward passes.
    During the optimizer.step(), it will load, update, and offload these states in slices.
    This pipelined approach significantly reduces GPU memory pressure during the optimizer step,
    making it highly beneficial for memory-intensive scenarios.
    """

    swap_to_device_stream = None
    swap_to_host_stream = None

    param_to_cpu_states_map = {}
    param_to_device_states_map = {}

    swap_to_host_events_map = {}
    swap_to_device_events_map = {}
    param_update_events_map = {}

    state_keys = ["exp_avg", "exp_avg_sq", "max_exp_avg_sq"]

    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        _owner: ClassVar[type | None] = None
        swap_optimizer: bool = False
        swap_optimizer_times: int = 16

        def build(self, **kwargs):
            _check_build_kwargs(self, kwargs)
            if not self.swap_optimizer:
                return OptimizersContainer(
                    config=_base_optimizer_config(self),
                    **kwargs,
                )
            if self._owner is None:
                raise NotImplementedError(
                    f"{type(self).__name__} has no owner class. "
                    "Define Config inside a Configurable subclass."
                )
            if not kwargs:
                return self._owner(config=replace(self))
            return self._owner(config=replace(self), **kwargs)

    def __init__(
        self,
        config: Config,
        *,
        model_parts: list[nn.Module],
    ) -> None:
        optimizer_cls = self._resolve_optimizer_cls(config.name)
        optimizer_kwargs = self._build_optimizer_kwargs(config)
        all_params = []
        self.optimizers = []
        self.model_parts = model_parts
        for model in self.model_parts:
            params = [p for p in model.parameters() if p.requires_grad]
            self.optimizers.append(optimizer_cls(params, **optimizer_kwargs))
            all_params.extend(params)
        self._validate_length(len(self.model_parts))
        self._post_init(all_params, optimizer_kwargs)

        # patch optimizer step functions for swap optimizer
        torch.optim.AdamW.step = swap_optimizer_step
        torch.optim.Adam.step = swap_optimizer_step
        logger.info("[SwapOptimizer] Patched AdamW.step and Adam.step")

        # create streams for swapping
        if SwapOptimizersContainer.swap_to_device_stream is None:
            # pyrefly: ignore [missing-attribute]
            SwapOptimizersContainer.swap_to_device_stream = get_torch_device().Stream()
            # pyrefly: ignore [missing-attribute]
            SwapOptimizersContainer.swap_to_host_stream = get_torch_device().Stream()

        # initialize states and cpu counterparts for each device param
        for idx, optim in enumerate(self.optimizers):
            optim.param_to_group_map = {}
            for group in optim.param_groups:
                for p in group["params"]:
                    optim.param_to_group_map[p] = group
                    SwapOptimizersContainer.param_state_initialization(p, optim)
            swap_num = sum(
                [
                    sum([unwrap_dtensor(p).numel() for p in group["params"]])
                    for group in optim.param_groups
                ]
            )
            optim.swap_numel = swap_num // config.swap_optimizer_times
            logger.info(
                f"Swap param numel for optimizer_{idx}: {optim.swap_numel} / {swap_num}\n"
            )

    @classmethod
    def param_state_initialization(cls, param, optim):
        cls.swap_to_host_events_map[param] = None

        device_state = optim.state[param]
        cls.param_to_device_states_map[param] = device_state
        cpu_state = {}
        cls.param_to_cpu_states_map[param] = cpu_state

        amsgrad = optim.param_to_group_map[param]["amsgrad"]

        for key in cls.state_keys:
            if key in device_state:
                continue
            if key == "max_exp_avg_sq" and not amsgrad:
                device_state[key] = None
                cpu_state[key] = None
            else:
                device_state[key] = torch.zeros_like(
                    param, memory_format=torch.contiguous_format
                )
                unwrap_dtensor(device_state[key]).untyped_storage().resize_(
                    0
                )  # offload device states
                cpu_state[key] = torch.zeros_like(
                    unwrap_dtensor(param), pin_memory=True, device="cpu"
                )

    @classmethod
    def swap_states_to_device(cls, param):
        if param not in cls.param_to_cpu_states_map:
            return

        cpu_state = cls.param_to_cpu_states_map[param]
        device_state = cls.param_to_device_states_map[param]
        for key in cls.state_keys:
            if key not in cpu_state or cpu_state[key] is None:
                continue
            local_state = unwrap_dtensor(device_state[key])
            if local_state.untyped_storage().size() == 0:
                local_state.untyped_storage().resize_(
                    cpu_state[key].untyped_storage().size()
                )
                local_state.copy_(cpu_state[key], non_blocking=True)

        cls.swap_to_device_events_map[param] = (
            # pyrefly: ignore [missing-attribute]
            get_torch_device()
            .current_stream()
            .record_event()
        )

    @classmethod
    def swap_states_to_host(cls, param):
        if param not in cls.param_to_device_states_map:
            return

        device_state = cls.param_to_device_states_map[param]
        cpu_state = cls.param_to_cpu_states_map[param]
        for key in cls.state_keys:
            if key not in device_state or device_state[key] is None:
                continue
            local_state = unwrap_dtensor(device_state[key])
            if local_state.untyped_storage().size() != 0:
                cpu_state[key].copy_(local_state, non_blocking=True)
                local_state.untyped_storage().resize_(0)

        cls.swap_to_host_events_map[param] = (
            # pyrefly: ignore [missing-attribute]
            get_torch_device()
            .current_stream()
            .record_event()
        )

    @classmethod
    def wait_swap_to_device_event(cls, param):
        event = cls.swap_to_device_events_map.get(param, None)
        if event is not None:
            # pyrefly: ignore [missing-attribute]
            get_torch_device().current_stream().wait_event(event)
            cls.swap_to_device_events_map[param] = None

    @classmethod
    def wait_param_update_event(cls, param):
        event = cls.param_update_events_map.get(param, None)
        if event is not None:
            # pyrefly: ignore [missing-attribute]
            get_torch_device().current_stream().wait_event(event)
            cls.param_update_events_map[param] = None


def param_update(param, state, param_group):
    beta1, beta2 = param_group["betas"]
    step_func = (
        torch._fused_adamw_
        if param_group["decoupled_weight_decay"]
        else torch._fused_adam_
    )
    step_func(
        [param],
        [param.grad],
        [state["exp_avg"]],
        [state["exp_avg_sq"]],
        [state["max_exp_avg_sq"]] if param_group["amsgrad"] else [],
        [param_group["step"]],
        amsgrad=param_group["amsgrad"],
        lr=param_group["lr"],
        beta1=beta1,
        beta2=beta2,
        weight_decay=param_group["weight_decay"],
        eps=param_group["eps"],
        maximize=param_group["maximize"],
    )


def pipeline_load_param(swap_numel, params_list, start_index, current_swap_count):
    torch_device = get_torch_device()
    # pyrefly: ignore [missing-attribute]
    torch_device.current_stream().wait_stream(
        SwapOptimizersContainer.swap_to_host_stream
    )

    # pyrefly: ignore [missing-attribute]
    with torch_device.stream(SwapOptimizersContainer.swap_to_device_stream):
        # pyrefly: ignore [missing-attribute]
        torch_device.current_stream().wait_stream(
            SwapOptimizersContainer.swap_to_host_stream
        )

        idx = start_index
        while idx < len(params_list):
            param_local = unwrap_dtensor(params_list[idx])
            if params_list[idx].grad is None:
                idx += 1
                continue  # skip no grad param

            numel = param_local.numel()
            if current_swap_count > 0 and current_swap_count + numel > swap_numel:
                break  # stop load params when the buffer is full

            SwapOptimizersContainer.swap_states_to_device(params_list[idx])
            current_swap_count += numel
            idx += 1

    return current_swap_count


@_use_grad_for_differentiable
def swap_optimizer_step(self, closure=None):
    if torch.jit.is_scripting():
        raise NotImplementedError(
            "SwapOptimizer does not support torch.jit.script by now."
        )

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        if "step" in group:
            group["step"] += 1
            if group["step"].is_cpu:
                group["step"] = group["step"].cuda()
        else:
            group["step"] = torch.tensor(
                1,
                dtype=torch.int64,
                # pyrefly: ignore [missing-attribute]
                device=get_torch_device().current_device(),
            )

    swap_count = 0
    params_list = [p for group in self.param_groups for p in group["params"]]
    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError(
                "SwapOptimizer step function does not support sparse gradients for now."
            )

        state = self.state[param]
        group = self.param_to_group_map[param]
        amsgrad = group["amsgrad"]

        # state initialization
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
            state["exp_avg_sq"] = torch.zeros_like(
                param, memory_format=torch.preserve_format
            )
        if "max_exp_avg_sq" not in state:
            state["max_exp_avg_sq"] = (
                torch.zeros_like(param, memory_format=torch.preserve_format)
                if amsgrad
                else None
            )

        # pipelined swap update (load -> update -> offload)
        # load
        if swap_count == 0:
            swap_count = pipeline_load_param(
                self.swap_numel, params_list, i, swap_count
            )

        # update
        SwapOptimizersContainer.wait_swap_to_device_event(param)
        param_update(param, state, group)
        SwapOptimizersContainer.param_update_events_map[param] = (
            # pyrefly: ignore [missing-attribute]
            get_torch_device()
            .current_stream()
            .record_event()
        )
        # offload
        # pyrefly: ignore [missing-attribute]
        with get_torch_device().stream(SwapOptimizersContainer.swap_to_host_stream):
            SwapOptimizersContainer.wait_param_update_event(param)
            swap_count -= unwrap_dtensor(param).numel()
            SwapOptimizersContainer.swap_states_to_host(param)

    return loss


_OWNER_ATTR = "_owner"
setattr(SwapOptimizersContainer.Config, _OWNER_ATTR, SwapOptimizersContainer)


def _build_npu_optimizer_config(self, **kwargs):
    return _swap_config_from(self).build(**kwargs)


def _patch_npu_optimizer_config_build():
    from torchtitan_npu.config.configs import OptimizerConfig

    OptimizerConfig.build = _build_npu_optimizer_config


def patch_optimizer_step():
    """Patch optimizer step functions for swap optimizer support."""
    torch.optim.AdamW.step = swap_optimizer_step
    torch.optim.Adam.step = swap_optimizer_step
    logger.info("[SwapOptimizer] Patched AdamW.step and Adam.step")


_patch_npu_optimizer_config_build()
