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
from typing import Any, ClassVar, TypeVar

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.optim import Optimizer
from torch.optim.optimizer import _use_grad_for_differentiable
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.tools.utils import get_device_info

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    DistributedMuon,
    MuonHybridOptimizersContainer,
)

try:
    from torch.distributed.checkpoint.state_dict import _get_fqns
except ImportError:
    _get_fqns = None


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


def get_torch_device():
    return get_device_info()[1]


def unwrap_dtensor(tensor):
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def wrap_like_param(local_tensor: torch.Tensor, tensor):
    if isinstance(tensor, DTensor):
        return DTensor.from_local(
            local_tensor,
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
            shape=tensor.size(),
            stride=tensor.stride(),
            run_check=False,
        )
    return local_tensor


def wrap_like_param_without_device_move(local_tensor: torch.Tensor, tensor):
    if isinstance(tensor, DTensor):
        spec = DTensorSpec(
            tensor.device_mesh,
            tensor.placements,
            tensor_meta=TensorMeta(
                shape=tensor.size(),
                stride=tensor.stride(),
                dtype=local_tensor.dtype,
            ),
        )
        return DTensor(
            local_tensor.view_as(local_tensor),
            spec,
            requires_grad=local_tensor.requires_grad,
        )
    return local_tensor


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
    _MISSING = object()
    MISSING = _MISSING

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
                    f"{type(self).__name__} has no owner class. Define Config inside a Configurable subclass."
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
            SwapOptimizersContainer.swap_to_device_stream = get_torch_device().Stream()
            SwapOptimizersContainer.swap_to_host_stream = get_torch_device().Stream()

        # initialize states and cpu counterparts for each device param
        for idx, optim in enumerate(self.optimizers):
            optim.param_to_group_map = {}
            for group in optim.param_groups:
                for p in group["params"]:
                    optim.param_to_group_map[p] = group
                    SwapOptimizersContainer.param_state_initialization(p, optim)
            swap_num = sum([sum([unwrap_dtensor(p).numel() for p in group["params"]]) for group in optim.param_groups])
            optim.swap_numel = swap_num // config.swap_optimizer_times
            logger.info(f"Swap param numel for optimizer_{idx}: {optim.swap_numel} / {swap_num}\n")

    @staticmethod
    def _restore_states(original_states):
        for state, original_state in original_states:
            state.clear()
            state.update(original_state)

    @staticmethod
    def _empty_device_cache():
        try:
            get_torch_device().empty_cache()
        except Exception as exc:
            logger.debug("Failed to empty device cache after optimizer load: %s", exc)

    @staticmethod
    def _move_step_to_device(step):
        if torch.is_tensor(step) and step.device.type == "cpu":
            return step.to(get_torch_device().current_device())
        return step

    @staticmethod
    def _save_optimizer_states():
        original_states = []
        for cpu_state in SwapOptimizersContainer.param_to_cpu_states_map.values():
            original_states.append((cpu_state, dict(cpu_state)))
        for state in SwapOptimizersContainer.param_to_device_states_map.values():
            original_states.append((state, dict(state)))
        return original_states

    @staticmethod
    def _save_param_groups(optimizers):
        return [(group, dict(group)) for optim in optimizers for group in optim.param_groups]

    @staticmethod
    def _restore_param_groups(original_param_groups):
        for group, original_group in original_param_groups:
            group.clear()
            group.update(original_group)

    @staticmethod
    def _param_group_value_for_state_dict(value):
        if torch.is_tensor(value):
            return unwrap_dtensor(value).detach().cpu().clone()
        return value

    @staticmethod
    def _param_names_by_param(model_part):
        param_names_by_param: dict[Any, list[str]] = {}
        try:
            named_parameters = model_part.named_parameters(remove_duplicate=False)
        except TypeError:
            named_parameters = model_part.named_parameters()
        for name, param in named_parameters:
            param_names = param_names_by_param.get(param)
            if param_names is None:
                param_names = []
                param_names_by_param[param] = param_names
            param_names.append(name)
        return param_names_by_param

    @staticmethod
    def _fqns_by_param(model_part):
        fqns_by_param = {}
        param_names_by_param = SwapOptimizersContainer._param_names_by_param(model_part)
        for name, param in model_part.named_parameters():
            ordered_fqns = []
            if _get_fqns is None:
                ordered_fqns = param_names_by_param.get(param, [name])
                fqns_by_param[param] = tuple(dict.fromkeys(ordered_fqns))
                continue
            for param_name in param_names_by_param.get(param, [name]):
                fqns = set(_get_fqns(model_part, param_name))
                if not fqns:
                    raise AssertionError(f"Expected at least 1 FQN for parameter '{param_name}', got 0")
                if len(fqns) > 1 and param_name not in fqns:
                    raise NotImplementedError(
                        "Swap optimizer checkpoint does not support saving "
                        f"flattened parameter '{param_name}' that maps to multiple "
                        f"FQNs: {sorted(fqns)}"
                    )
                ordered_fqns.extend(fqn for fqn in [param_name, *sorted(fqns)] if fqn in fqns)
            ordered_fqns = list(dict.fromkeys(ordered_fqns))
            fqns_by_param[param] = tuple(ordered_fqns)
        return fqns_by_param

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
                local_param = unwrap_dtensor(param)
                device_state[key] = torch.zeros_like(param, memory_format=torch.contiguous_format)
                unwrap_dtensor(device_state[key]).untyped_storage().resize_(0)
                cpu_state[key] = torch.zeros_like(local_param, pin_memory=True, device="cpu")
                device_state[key] = cls._clone_loaded_state_for_device_placeholder(
                    param,
                    cpu_state[key],
                )

    @classmethod
    def swap_states_to_device(cls, param):
        if param not in cls.param_to_cpu_states_map:
            return

        cpu_state = cls.param_to_cpu_states_map[param]
        device_state = cls.param_to_device_states_map[param]
        local_param = unwrap_dtensor(param)
        for key in cls.state_keys:
            if key not in cpu_state or cpu_state[key] is None:
                continue
            local_state = unwrap_dtensor(device_state[key])
            if local_state.untyped_storage().size() == 0:
                if local_state.device != local_param.device:
                    local_state = torch.empty_strided(
                        cpu_state[key].size(),
                        cpu_state[key].stride(),
                        dtype=cpu_state[key].dtype,
                        layout=cpu_state[key].layout,
                        device=local_param.device,
                    )
                    device_state[key] = wrap_like_param(local_state, param)
                else:
                    local_state.untyped_storage().resize_(cpu_state[key].untyped_storage().size())
                local_state.copy_(cpu_state[key], non_blocking=True)

        cls.swap_to_device_events_map[param] = get_torch_device().current_stream().record_event()

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

        cls.swap_to_host_events_map[param] = get_torch_device().current_stream().record_event()

    @classmethod
    def wait_swap_to_device_event(cls, param):
        event = cls.swap_to_device_events_map.get(param, None)
        if event is not None:
            get_torch_device().current_stream().wait_event(event)
            cls.swap_to_device_events_map[param] = None

    @classmethod
    def wait_param_update_event(cls, param):
        event = cls.param_update_events_map.get(param, None)
        if event is not None:
            get_torch_device().current_stream().wait_event(event)
            cls.param_update_events_map[param] = None

    @classmethod
    def _tensor_for_state_dict(cls, tensor, like_param=None):
        local_tensor = unwrap_dtensor(tensor)
        if local_tensor.numel() != 0 and local_tensor.untyped_storage().size() == 0:
            raise RuntimeError("Cannot checkpoint a swapped optimizer state without CPU cache.")
        cpu_tensor = local_tensor.detach() if local_tensor.device.type == "cpu" else local_tensor.detach().cpu()
        if isinstance(like_param, DTensor):
            return wrap_like_param_without_device_move(cpu_tensor, like_param)
        return cpu_tensor

    @classmethod
    def _state_value_for_state_dict(cls, param, state, key):
        cpu_state = cls.param_to_cpu_states_map.get(param)
        if cpu_state is not None:
            cpu_value = cpu_state.get(key)
            if cpu_value is not None:
                return cls._tensor_for_state_dict(cpu_value, param)

        value = state.get(key)
        if value is None:
            return None
        return cls._tensor_for_state_dict(value, param)

    @classmethod
    def _state_step_for_state_dict(cls, optim, param, state):
        if "step" in state:
            step = state["step"]
            if torch.is_tensor(step):
                return step.detach().cpu().clone()
            return step

        group = getattr(optim, "param_to_group_map", {}).get(param)
        if group is not None and "step" in group:
            step = group["step"]
            if torch.is_tensor(step):
                return step.detach().cpu().clone()
            return step
        if any(state.get(key) is not None for key in cls.state_keys):
            return torch.tensor(0, dtype=torch.int64, device="cpu")
        return None

    @classmethod
    def _add_param_group_to_state_dict(cls, state_dict, group, fqn):
        for key, value in group.items():
            if key == "params":
                continue
            state_dict[f"param_groups.{fqn}.{key}"] = cls._param_group_value_for_state_dict(value)

    @classmethod
    def _add_param_state_to_state_dict(cls, state_dict, optim, param, fqn):
        state = optim.state[param]
        for key in cls.state_keys:
            value = cls._state_value_for_state_dict(param, state, key)
            if value is not None:
                state_dict[f"state.{fqn}.{key}"] = value

        step = cls._state_step_for_state_dict(optim, param, state)
        if step is not None:
            state_dict[f"state.{fqn}.step"] = step

    @classmethod
    def _optimizer_state_dict(cls, model_part, optim):
        fqns_by_param = cls._fqns_by_param(model_part)
        state_dict = {}
        for group in optim.param_groups:
            for param in group["params"]:
                fqn = fqns_by_param[param][0]
                cls._add_param_state_to_state_dict(state_dict, optim, param, fqn)
                cls._add_param_group_to_state_dict(state_dict, group, fqn)
        return state_dict

    @classmethod
    def _wait_pending_swap_to_host(cls):
        if cls.swap_to_host_stream is None:
            return
        cls.swap_to_host_stream.synchronize()

    @classmethod
    def _clone_to_cpu_cache(cls, tensor):
        cpu_tensor = unwrap_dtensor(tensor).detach().cpu()
        try:
            cached_tensor = torch.empty_like(
                cpu_tensor,
                pin_memory=True,
                device="cpu",
            )
            cached_tensor.copy_(cpu_tensor, non_blocking=True)
            return cached_tensor
        except RuntimeError:
            return cpu_tensor.clone()

    @classmethod
    def _clone_loaded_state_for_device_placeholder(cls, param, tensor):
        local_tensor = unwrap_dtensor(tensor)
        local_param = unwrap_dtensor(param)
        placeholder = torch.empty_strided(
            local_tensor.size(),
            local_tensor.stride(),
            dtype=local_tensor.dtype,
            layout=local_tensor.layout,
            device=local_tensor.device,
        )
        placeholder.untyped_storage().resize_(0)
        if placeholder.device != local_param.device:
            return placeholder
        return wrap_like_param(placeholder, param)

    @classmethod
    def _clone_loaded_value(cls, value):
        if torch.is_tensor(value):
            return unwrap_dtensor(value).detach().clone()
        return value

    @classmethod
    def _state_dict_value_for_fqns(cls, state_dict, prefix, fqns, key):
        for fqn in fqns:
            flat_key = f"{prefix}.{fqn}.{key}"
            if flat_key in state_dict:
                return state_dict[flat_key]
        return cls._MISSING

    @classmethod
    def _load_param_group(cls, group, fqns, state_dict):
        for key in group:
            if key == "params":
                continue
            value = cls._state_dict_value_for_fqns(state_dict, "param_groups", fqns, key)
            if value is not cls._MISSING:
                group[key] = cls._clone_loaded_value(value)

    @classmethod
    def _load_param_state(cls, optim, param, fqns, state_dict):
        group = optim.param_to_group_map[param]
        state = optim.state[param]
        state.clear()
        cpu_state = cls.param_to_cpu_states_map.setdefault(param, {})
        cls.param_to_device_states_map[param] = state

        for key in cls.state_keys:
            value = cls._state_dict_value_for_fqns(state_dict, "state", fqns, key)
            if value is not cls._MISSING:
                cpu_state[key] = cls._clone_to_cpu_cache(value)
                state[key] = cls._clone_loaded_state_for_device_placeholder(
                    param,
                    cpu_state[key],
                )
                unwrap_dtensor(state[key]).untyped_storage().resize_(0)
            elif key == "max_exp_avg_sq" and not group["amsgrad"]:
                state[key] = None
                cpu_state[key] = None
            else:
                cpu_state.pop(key, None)

        step = cls._state_dict_value_for_fqns(state_dict, "state", fqns, "step")
        if step is cls._MISSING:
            step = cls._state_dict_value_for_fqns(state_dict, "param_groups", fqns, "step")
        if step is not cls._MISSING:
            group["step"] = cls._clone_loaded_value(step)

    @classmethod
    def _load_optimizer_state_dict(cls, model_part, optim, state_dict):
        fqns_by_param = cls._fqns_by_param(model_part)
        optim.param_to_group_map = {}

        for group in optim.param_groups:
            loaded_group = False
            for param in group["params"]:
                optim.param_to_group_map[param] = group
                fqns = fqns_by_param[param]
                if not loaded_group:
                    cls._load_param_group(group, fqns, state_dict)
                    loaded_group = True
                cls._load_param_state(optim, param, fqns, state_dict)

    def state_dict(self) -> dict[str, Any]:
        self._wait_pending_swap_to_host()
        state_dict = {}
        for model_part, optim in zip(self.model_parts, self.optimizers, strict=False):
            state_dict.update(self._optimizer_state_dict(model_part, optim))
        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        original_states = self._save_optimizer_states()
        original_param_groups = self._save_param_groups(self.optimizers)
        try:
            for model_part, optim in zip(self.model_parts, self.optimizers, strict=False):
                self._load_optimizer_state_dict(model_part, optim, state_dict)
        except Exception:
            self._restore_states(original_states)
            self._restore_param_groups(original_param_groups)
            raise
        self._empty_device_cache()

    # Public aliases for classmethods used by subclasses
    fqns_by_param = _fqns_by_param
    clone_to_cpu_cache = _clone_to_cpu_cache
    state_step_for_state_dict = _state_step_for_state_dict
    add_param_state_to_state_dict = _add_param_state_to_state_dict
    add_param_group_to_state_dict = _add_param_group_to_state_dict
    optimizer_state_dict = _optimizer_state_dict
    wait_pending_swap_to_host = _wait_pending_swap_to_host
    load_param_group = _load_param_group
    load_param_state = _load_param_state
    state_dict_value_for_fqns = _state_dict_value_for_fqns
    clone_loaded_value = _clone_loaded_value
    load_optimizer_state_dict = _load_optimizer_state_dict
    empty_device_cache = _empty_device_cache


def param_update(param, state, param_group):
    beta1, beta2 = param_group["betas"]
    step_func = torch._fused_adamw_ if param_group["decoupled_weight_decay"] else torch._fused_adam_
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
    torch_device.current_stream().wait_stream(SwapOptimizersContainer.swap_to_host_stream)

    with torch_device.stream(SwapOptimizersContainer.swap_to_device_stream):
        torch_device.current_stream().wait_stream(SwapOptimizersContainer.swap_to_host_stream)

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
        raise NotImplementedError("SwapOptimizer does not support torch.jit.script by now.")

    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()

    for group in self.param_groups:
        if "step" in group:
            group["step"] += 1
            group["step"] = SwapOptimizersContainer._move_step_to_device(group["step"])
        else:
            group["step"] = torch.tensor(
                1,
                dtype=torch.int64,
                device=get_torch_device().current_device(),
            )

    swap_count = 0
    params_list = [p for group in self.param_groups for p in group["params"]]
    for i, param in enumerate(params_list):
        if param.grad is None:
            continue
        if param.grad.is_sparse:
            raise RuntimeError("SwapOptimizer step function does not support sparse gradients for now.")

        state = self.state[param]
        group = self.param_to_group_map[param]
        amsgrad = group["amsgrad"]

        # state initialization
        if len(state) == 0:
            state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
            state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        if "max_exp_avg_sq" not in state:
            state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format) if amsgrad else None

        # pipelined swap update (load -> update -> offload)
        # load
        if swap_count == 0:
            swap_count = pipeline_load_param(self.swap_numel, params_list, i, swap_count)

        # update
        SwapOptimizersContainer.wait_swap_to_device_event(param)
        param_update(param, state, group)
        SwapOptimizersContainer.param_update_events_map[param] = get_torch_device().current_stream().record_event()
        # offload
        with get_torch_device().stream(SwapOptimizersContainer.swap_to_host_stream):
            SwapOptimizersContainer.wait_param_update_event(param)
            swap_count -= unwrap_dtensor(param).numel()
            SwapOptimizersContainer.swap_states_to_host(param)

    return loss


_OWNER_ATTR = "_owner"
setattr(SwapOptimizersContainer.Config, _OWNER_ATTR, SwapOptimizersContainer)


def patch_optimizer_step():
    """Patch optimizer step functions for swap optimizer support."""
    torch.optim.AdamW.step = swap_optimizer_step
    torch.optim.Adam.step = swap_optimizer_step
    logger.info("[SwapOptimizer] Patched AdamW.step and Adam.step")


# =============================================================================
# Swap Muon optimizer: offloads Muon momentum_buffer to CPU and swaps on demand.
# =============================================================================


class SwapMuonState:
    """Per-parameter swap state for Muon momentum_buffer."""

    def __init__(self, param, device_module):
        self.param = param
        self.device_module = device_module
        self._cpu_momentum = None
        self._on_device = True
        self._swap_event = None
        self._optim_state: dict | None = None
        self._buf_shape = None
        self._buf_dtype = None

    @property
    def cpu_momentum(self):
        return self._cpu_momentum

    @cpu_momentum.setter
    def cpu_momentum(self, value):
        self._cpu_momentum = value

    @property
    def on_device(self):
        return self._on_device

    @on_device.setter
    def on_device(self, value):
        self._on_device = value

    @property
    def buf_shape(self):
        return self._buf_shape

    @buf_shape.setter
    def buf_shape(self, value):
        self._buf_shape = value

    @property
    def buf_dtype(self):
        return self._buf_dtype

    @buf_dtype.setter
    def buf_dtype(self, value):
        self._buf_dtype = value

    @property
    def optim_state(self):
        return self._optim_state

    @optim_state.setter
    def optim_state(self, value):
        self._optim_state = value

    def init_from_momentum_buffer(self, momentum_buffer):
        local_buf = unwrap_dtensor(momentum_buffer)
        self._cpu_momentum = torch.zeros_like(local_buf, pin_memory=True, device="cpu")
        self._cpu_momentum.copy_(local_buf, non_blocking=False)
        self._buf_shape = local_buf.shape
        self._buf_dtype = local_buf.dtype
        self._set_momentum_buffer(None)
        self._on_device = False

    def swap_to_device(self, stream=None):
        if self._cpu_momentum is None or self._on_device:
            return
        state = self._get_momentum_buffer()
        cpu = self._cpu_momentum
        if state is None:
            if self._buf_shape is None or self._buf_dtype is None:
                raise RuntimeError("SwapMuonState buffer metadata is not initialized.")
            local_param = unwrap_dtensor(self.param)
            local_state = torch.empty(
                self._buf_shape,
                dtype=self._buf_dtype,
                device=local_param.device,
            )
            self._set_momentum_buffer(
                wrap_like_param(local_state, self.param) if isinstance(self.param, DTensor) else local_state
            )
        local_state = unwrap_dtensor(self._get_momentum_buffer())
        local_state.copy_(cpu, non_blocking=True)
        self._on_device = True
        if stream is not None:
            self._swap_event = stream.record_event()
        else:
            self._swap_event = self.device_module.current_stream().record_event()

    def swap_to_host(self, stream=None):
        if self._cpu_momentum is None or not self._on_device:
            return
        state = self._get_momentum_buffer()
        if state is not None:
            local_state = unwrap_dtensor(state)
            if local_state.untyped_storage().size() != 0:
                self._cpu_momentum.copy_(local_state, non_blocking=True)
            self._set_momentum_buffer(None)
        self._on_device = False
        if stream is not None:
            self._swap_event = stream.record_event()
        else:
            self._swap_event = self.device_module.current_stream().record_event()

    def wait_swap(self):
        if self._swap_event is not None:
            self.device_module.current_stream().wait_event(self._swap_event)
            self._swap_event = None

    def _get_momentum_buffer(self):
        if self._optim_state is not None:
            return self._optim_state.get("momentum_buffer")
        return None

    def _set_momentum_buffer(self, value):
        if self._optim_state is not None:
            self._optim_state["momentum_buffer"] = value


class SwapMuonHybridOptimizersContainer(MuonHybridOptimizersContainer):
    """Container for Muon + AdamW hybrid optimizers with swap support."""

    @dataclass(kw_only=True, slots=True)
    class Config(MuonHybridOptimizersContainer.Config):
        _owner: ClassVar[type | None] = None
        swap_optimizer_times: int = 16
        swap_merge_buckets: int = 1

        def build(self, **kwargs):
            model_parts = kwargs["model_parts"]
            parallel_dims = kwargs.get("parallel_dims")
            if parallel_dims is None:
                from torchtitan_npu.patches.torchtitan._trainer_config_stash import get_active_parallel_dims

                parallel_dims = get_active_parallel_dims()
            ft_manager = kwargs.get("ft_manager")
            base = MuonHybridOptimizersContainer.Config(
                **{
                    f.name: getattr(self, f.name)
                    for f in fields(MuonHybridOptimizersContainer.Config)
                    if hasattr(self, f.name)
                },
            ).build(model_parts=model_parts, parallel_dims=parallel_dims, ft_manager=ft_manager)
            owner = self._owner
            if owner is None:
                raise RuntimeError("SwapMuonHybridOptimizersContainer.Config owner is not initialized.")
            return owner(
                model_parts,
                base.optimizers,
                muon_adjust_lr_fn=base.muon_adjust_lr_fn,
                swap_optimizer_times=self.swap_optimizer_times,
                swap_merge_buckets=self.swap_merge_buckets,
            )

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizers: list[Optimizer],
        muon_adjust_lr_fn: str | None = None,
        swap_optimizer_times: int = 16,
        swap_merge_buckets: int = 1,
    ) -> None:
        super().__init__(model_parts, optimizers, muon_adjust_lr_fn)
        if swap_optimizer_times <= 0:
            raise ValueError(f"swap_optimizer_times must be positive, got {swap_optimizer_times}")
        if swap_merge_buckets <= 0:
            raise ValueError(f"swap_merge_buckets must be positive, got {swap_merge_buckets}")
        self._swap_optimizer_times = swap_optimizer_times
        self._device_module = get_torch_device()
        if SwapOptimizersContainer.swap_to_device_stream is None:
            SwapOptimizersContainer.swap_to_device_stream = self._device_module.Stream()
            SwapOptimizersContainer.swap_to_host_stream = self._device_module.Stream()
        self._swap_to_device_stream = SwapOptimizersContainer.swap_to_device_stream
        self._swap_to_host_stream = SwapOptimizersContainer.swap_to_host_stream
        self._muon_swap_states: dict[int, SwapMuonState] = {}
        muon_optim = self.optimizers[0]
        if not isinstance(muon_optim, DistributedMuon):
            raise TypeError(f"First optimizer must be DistributedMuon, got {type(muon_optim)}")
        if not muon_optim.fsdp_enabled:
            raise RuntimeError("Swap optimizer requires FSDP to be enabled; DDP is not supported")
        muon_optim._swap_enabled = True
        muon_optim._swap_container = self
        muon_optim._swap_merge_buckets = swap_merge_buckets
        muon_optim._device_module = self._device_module
        muon_optim._swap_to_device_stream = self._swap_to_device_stream
        muon_optim._swap_to_host_stream = self._swap_to_host_stream
        self._enable_adamw_swap(swap_optimizer_times)
        self._pre_init_swap_states()
        logger.info(
            f"[SwapMuon] Built SwapMuonHybridOptimizersContainer "
            f"swap_optimizer_times={swap_optimizer_times} | "
            f"swap_merge_buckets={swap_merge_buckets} | "
            f"muon_swap_states={len(self._muon_swap_states)}"
        )

    @property
    def muon_swap_states(self):
        return self._muon_swap_states

    def step(self, *args, **kwargs) -> None:
        self.optimizers[0].step(*args, **kwargs)
        if len(self.optimizers) > 1:
            self.optimizers[1].step(*args, **kwargs)

    def get_swap_state(self, param_id: int):
        return self._muon_swap_states.get(param_id)

    def state_dict(self) -> dict[str, Any]:
        self._wait_pending_swap_to_host()
        merged = {}
        muon_optim = self.optimizers[0]
        for model in self.model_parts:
            merged.update(self._muon_state_dict_for_model(model, muon_optim))
        adamw_optim = self.optimizers[1] if len(self.optimizers) > 1 else None
        if adamw_optim is not None:
            for model in self.model_parts:
                merged.update(self._adamw_state_dict_for_model(model, adamw_optim))
        return merged

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        muon_optim = self.optimizers[0]
        if not self._muon_swap_states:
            self._ensure_swap_states_initialized()
        for model in self.model_parts:
            self._load_muon_state_dict_for_model(model, muon_optim, state_dict)
        adamw_optim = self.optimizers[1] if len(self.optimizers) > 1 else None
        if adamw_optim is not None:
            adamw_optim.param_to_group_map = {}
            for model in self.model_parts:
                self._load_adamw_state_dict_for_model(model, adamw_optim, state_dict)
        SwapOptimizersContainer.empty_device_cache()

    def _muon_state_dict_for_model(self, model, muon_optim):
        merged = {}
        fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
        for group in muon_optim.param_groups:
            self._serialize_group_state(group, fqns_by_param, muon_optim, merged)
        return merged

    def _serialize_group_state(self, group, fqns_by_param, muon_optim, merged):
        for param in group["params"]:
            if param not in fqns_by_param:
                continue
            fqn = fqns_by_param[param][0]
            self._serialize_param_state(param, fqn, group, muon_optim, merged)

    def _load_muon_state_dict_for_model(self, model, muon_optim, state_dict):
        fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
        for group in muon_optim.param_groups:
            self._load_muon_group_state(group, fqns_by_param, muon_optim, state_dict)

    def _load_muon_group_state(self, group, fqns_by_param, muon_optim, state_dict):
        loaded_group = False
        for param in group["params"]:
            if param not in fqns_by_param:
                continue
            fqns = fqns_by_param[param]
            if not loaded_group:
                SwapOptimizersContainer.load_param_group(group, fqns, state_dict)
                loaded_group = True
            self._load_param_state(param, fqns, group, muon_optim, state_dict)

    def _enable_adamw_swap(self, swap_optimizer_times: int):
        adamw_optim = self.optimizers[1] if len(self.optimizers) > 1 else None
        if adamw_optim is None:
            return
        adamw_optim.param_to_group_map = {}
        for group in adamw_optim.param_groups:
            for p in group["params"]:
                adamw_optim.param_to_group_map[p] = group
                SwapOptimizersContainer.param_state_initialization(p, adamw_optim)
        swap_num = sum(unwrap_dtensor(p).numel() for group in adamw_optim.param_groups for p in group["params"])
        adamw_optim.swap_numel = swap_num // swap_optimizer_times
        adamw_optim.step = swap_optimizer_step.__get__(adamw_optim, type(adamw_optim))
        adamw_param_count = len(adamw_optim.param_groups[0]["params"])
        logger.info(f"[SwapMuon] AdamW swap enabled: {adamw_param_count} params | swap_numel={adamw_optim.swap_numel}")

    def _pre_init_swap_states(self):
        muon_optim = self.optimizers[0]
        pre_init_count = 0
        for group in muon_optim.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = muon_optim.state.setdefault(p, {})
                local_p = unwrap_dtensor(p)
                zero_buf = torch.zeros_like(local_p)
                if isinstance(p, DTensor):
                    zero_buf = wrap_like_param(zero_buf, p)
                state["momentum_buffer"] = zero_buf
                swap_state = SwapMuonState(p, self._device_module)
                swap_state.optim_state = state
                swap_state.init_from_momentum_buffer(zero_buf)
                self._muon_swap_states[id(p)] = swap_state
                pre_init_count += 1
        logger.info(
            f"[SwapMuon] Pre-init swap states: {pre_init_count} momentum_buffers created (zeros) and offloaded to CPU"
        )

    def _serialize_param_state(self, param, fqn, group, muon_optim, merged):
        merged[f"state.{fqn}.momentum_buffer"] = self._serialize_momentum_buffer(param, muon_optim)
        state = muon_optim.state.get(param, {})
        step = SwapOptimizersContainer.state_step_for_state_dict(muon_optim, param, state)
        if step is not None:
            merged[f"state.{fqn}.step"] = step
        SwapOptimizersContainer.add_param_group_to_state_dict(merged, group, fqn)

    def _serialize_momentum_buffer(self, param, muon_optim):
        swap_state = self._muon_swap_states.get(id(param))
        if swap_state is not None and swap_state.cpu_momentum is not None:
            cpu_tensor = SwapOptimizersContainer.clone_to_cpu_cache(swap_state.cpu_momentum)
            if isinstance(param, DTensor):
                cpu_tensor = wrap_like_param_without_device_move(cpu_tensor, param)
            return cpu_tensor
        local_p = unwrap_dtensor(param)
        placeholder = torch.zeros_like(local_p, device="cpu")
        if isinstance(param, DTensor):
            placeholder = wrap_like_param_without_device_move(placeholder, param)
        return placeholder

    def _load_param_state(self, param, fqns, group, muon_optim, state_dict):
        swap_state = self._muon_swap_states.get(id(param))
        if swap_state is not None:
            value = SwapOptimizersContainer.state_dict_value_for_fqns(state_dict, "state", fqns, "momentum_buffer")
            if value is not SwapOptimizersContainer.MISSING:
                self._load_momentum_from_state_dict(swap_state, value, muon_optim, param)
        self._load_step_from_state_dict(state_dict, fqns, group)

    def _load_momentum_from_state_dict(self, swap_state, value, muon_optim, param):
        swap_state.cpu_momentum = SwapOptimizersContainer.clone_to_cpu_cache(value)
        if swap_state.cpu_momentum is not None:
            swap_state.buf_shape = swap_state.cpu_momentum.shape
            swap_state.buf_dtype = swap_state.cpu_momentum.dtype
        state = muon_optim.state.get(param, {})
        buf = state.get("momentum_buffer")
        if buf is not None:
            buf_local = buf.to_local() if isinstance(buf, DTensor) else buf
            if buf_local.untyped_storage().size() != 0:
                buf_local.zero_()
        state["momentum_buffer"] = None
        swap_state.on_device = False

    def _load_step_from_state_dict(self, state_dict, fqns, group):
        step = SwapOptimizersContainer.state_dict_value_for_fqns(state_dict, "state", fqns, "step")
        if step is SwapOptimizersContainer.MISSING:
            step = SwapOptimizersContainer.state_dict_value_for_fqns(state_dict, "param_groups", fqns, "step")
        if step is not SwapOptimizersContainer.MISSING:
            group["step"] = SwapOptimizersContainer.clone_loaded_value(step)

    def _wait_pending_swap_to_host(self):
        SwapOptimizersContainer.wait_pending_swap_to_host()

    def _ensure_swap_states_initialized(self):
        muon_optim = self.optimizers[0]
        for group in muon_optim.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                pid = id(p)
                if pid in self._muon_swap_states:
                    continue
                state = muon_optim.state.setdefault(p, {})
                if "momentum_buffer" not in state or state["momentum_buffer"] is None:
                    local_p = unwrap_dtensor(p)
                    cpu_buf = torch.zeros_like(local_p, pin_memory=True, device="cpu")
                    state["momentum_buffer"] = None
                    swap_state = SwapMuonState(p, self._device_module)
                    swap_state.optim_state = state
                    swap_state.cpu_momentum = cpu_buf
                    swap_state.buf_shape = local_p.shape
                    swap_state.buf_dtype = local_p.dtype
                    swap_state.on_device = False
                else:
                    swap_state = SwapMuonState(p, self._device_module)
                    swap_state.optim_state = state
                self._muon_swap_states[pid] = swap_state

    def _adamw_state_dict_for_model(self, model, optim):
        fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
        state_dict = {}
        for group in optim.param_groups:
            for param in group["params"]:
                if param not in fqns_by_param:
                    continue
                fqn = fqns_by_param[param][0]
                SwapOptimizersContainer.add_param_state_to_state_dict(state_dict, optim, param, fqn)
                SwapOptimizersContainer.add_param_group_to_state_dict(state_dict, group, fqn)
        return state_dict

    def _load_adamw_state_dict_for_model(self, model, optim, state_dict):
        fqns_by_param = SwapOptimizersContainer.fqns_by_param(model)
        for group in optim.param_groups:
            loaded_group = False
            for param in group["params"]:
                if param not in fqns_by_param:
                    continue
                optim.param_to_group_map[param] = group
                fqns = fqns_by_param[param]
                if not loaded_group:
                    SwapOptimizersContainer.load_param_group(group, fqns, state_dict)
                    loaded_group = True
                SwapOptimizersContainer.load_param_state(optim, param, fqns, state_dict)
