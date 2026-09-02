# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""Explicit CPU offload policy for DistributedMuon and AdamW optimizer state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
from torch.distributed._tensor import DTensor
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import derive, override
from torchtitan.distributed.flex_shard.distributed_muon import DistributedMuon

from torchtitan_npu.extensions.novaswap import swap_api
from torchtitan_npu.extensions.novaswap.swap_primitive import validate_tensor_for_swap


class MuonSwapOptimizersContainer(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def __init__(self, config: Config, *, model_parts: list[Any]) -> None:
        # Preserve the standard constructor, parameter grouping, FlexShard setup, and order.
        super().__init__(config=config, model_parts=model_parts)
        parameter_to_part = {
            id(parameter): part_index
            for part_index, model_part in enumerate(model_parts)
            for parameter in model_part.parameters()
        }
        optimizer_instance_by_part: dict[int, int] = {}
        for optimizer in self.optimizers:
            if isinstance(optimizer, DistributedMuon):
                part_indices = {
                    parameter_to_part[id(parameter)]
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                if len(part_indices) != 1:
                    raise RuntimeError(
                        "DistributedMuon optimizer must own parameters from exactly "
                        f"one model part, got part indices {sorted(part_indices)}"
                    )
                part_index = next(iter(part_indices))
                optimizer_instance = optimizer_instance_by_part.get(part_index, 0)
                optimizer_instance_by_part[part_index] = optimizer_instance + 1
                _install_muon_swap_adapter(optimizer, part_index, optimizer_instance)
            elif isinstance(optimizer, torch.optim.AdamW):
                _install_adamw_swap_adapter(optimizer)

    def state_dict(self) -> dict[str, Any]:
        raise RuntimeError("Muon state swap v1 does not support optimizer checkpoint save")

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        raise RuntimeError("Muon state swap v1 does not support optimizer checkpoint load")


class MuonSwapCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        pass

    def __init__(self, config: Config, **kwargs: Any) -> None:
        # v1 has no CPU payload lease. Model-only checkpointing is deliberately
        # rejected until it has its own compatibility test instead of inferred safe.
        if config.enable or config.initial_load_path:
            raise ValueError(
                "Muon state swap v1 does not support checkpoint save or load; "
                "disable checkpointing and remove checkpoint.initial_load_path"
            )
        super().__init__(config=config, **kwargs)


def _set_runtime_attribute(target: object, name: str, value: object) -> None:
    setattr(target, name, value)


def _install_muon_swap_adapter(optimizer: DistributedMuon, model_part: int, optimizer_instance: int) -> None:
    names: dict[torch.Tensor, str] = {}
    contracts: dict[torch.Tensor, tuple[Any, ...]] = {}
    _set_runtime_attribute(optimizer, "_torchtitan_npu_swap_names", names)
    _set_runtime_attribute(optimizer, "_torchtitan_npu_swap_contracts", contracts)

    original_preflight = optimizer._preflight_step
    original_momentum = optimizer._momentum
    original_prepare_local = optimizer._prepare_local

    def name_for(compute_layout) -> str:
        name = names.get(compute_layout.param)
        if name is None:
            name = f"optimizer_state.{model_part}.{optimizer_instance}.{compute_layout.fqn}.momentum_buffer"
            names[compute_layout.param] = name
        return name

    def preflight(self) -> None:
        for layout in self._parameter_compute_layouts:
            momentum = self.state.get(layout.param, {}).get("momentum_buffer")
            if momentum is None or momentum.to_local().numel() == 0:
                continue
            name = name_for(layout)
            if swap_api.get_handle_phase(name) != "D2H":
                raise RuntimeError(
                    f"Muon swap handle is not ready for H2D for {layout.fqn!r}: {swap_api.get_handle_phase(name)!r}"
                )
        original_preflight()
        for layout in self._parameter_compute_layouts:
            momentum = self.state.get(layout.param, {}).get("momentum_buffer")
            if momentum is None:
                continue
            local = momentum.to_local()
            if local.numel():
                name = name_for(layout)
                contract = (
                    tuple(local.shape),
                    tuple(local.stride()),
                    local.dtype,
                    local.device,
                    local.untyped_storage()._cdata,
                )
                if contract != contracts[layout.param]:
                    raise RuntimeError(f"Muon local momentum identity changed for {layout.fqn!r}")

    def momentum(self, compute_layout, grad):
        state = self.state[compute_layout.param]
        created = "momentum_buffer" not in state
        result = original_momentum(compute_layout, grad)
        local = result.to_local()
        if local.numel() == 0:
            return result
        name = name_for(compute_layout)
        if created:
            validate_tensor_for_swap(local)
            swap_api.register_tensor(local, name)
            contracts[compute_layout.param] = (
                tuple(local.shape),
                tuple(local.stride()),
                local.dtype,
                local.device,
                local.untyped_storage()._cdata,
            )
            swap_api.execute(name, "D2H")
        return result

    def prepare_local(self, compute_layout, out) -> None:
        name = name_for(compute_layout)
        local = self.state[compute_layout.param]["momentum_buffer"].to_local()
        if local.numel() == 0:
            original_prepare_local(compute_layout, out)
            return
        swap_api.execute(name, "H2D")
        swap_api.execute(name, "WAIT_DEVICE")
        original_prepare_local(compute_layout, out)
        swap_api.execute(name, "D2H")

    optimizer._preflight_step = MethodType(preflight, optimizer)
    optimizer._momentum = MethodType(momentum, optimizer)
    optimizer._prepare_local = MethodType(prepare_local, optimizer)

    def reject_state_dict(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Muon state swap v1 does not support optimizer state_dict access")

    _set_runtime_attribute(optimizer, "state_dict", reject_state_dict)
    _set_runtime_attribute(optimizer, "load_state_dict", reject_state_dict)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _install_adamw_swap_adapter(optimizer: torch.optim.AdamW) -> None:
    """Offload AdamW moments around a whole optimizer step.

    AdamW creates its states lazily inside its first ``step()``.  The post hook
    therefore registers and offloads the first pair of moments; later pre hooks
    restore all moments before AdamW reads them, and post hooks offload them
    again after AdamW writes them.
    """
    contracts: dict[tuple[torch.Tensor, str], tuple[Any, ...]] = {}
    _set_runtime_attribute(optimizer, "_torchtitan_npu_swap_contracts", contracts)

    def name_for(parameter: torch.Tensor, state_key: str) -> str:
        return f"adamw.{id(parameter)}.{state_key}"

    def swap_states():
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                for state_key in ("exp_avg", "exp_avg_sq"):
                    moment = state.get(state_key)
                    if moment is not None:
                        yield parameter, state_key, _local_tensor(moment)

    def pre_hook(_optimizer, _args, _kwargs) -> None:
        ready_names = []
        for parameter, state_key, moment in swap_states():
            if moment.numel() == 0:
                continue
            name = name_for(parameter, state_key)
            if swap_api.get_handle_phase(name) != "D2H":
                raise RuntimeError(
                    f"AdamW swap handle is not ready for H2D for {name!r}: {swap_api.get_handle_phase(name)!r}"
                )
            swap_api.execute(name, "H2D")
            ready_names.append(name)
        for name in ready_names:
            swap_api.execute(name, "WAIT_DEVICE")

    def post_hook(_optimizer, _args, _kwargs) -> None:
        for parameter, state_key, moment in swap_states():
            if moment.numel() == 0:
                continue
            name = name_for(parameter, state_key)
            contract = (
                tuple(moment.shape),
                tuple(moment.stride()),
                moment.dtype,
                moment.device,
                moment.untyped_storage()._cdata,
            )
            known_contract = contracts.get((parameter, state_key))
            if known_contract is None:
                validate_tensor_for_swap(moment)
                swap_api.register_tensor(moment, name)
                contracts[(parameter, state_key)] = contract
            elif known_contract != contract:
                raise RuntimeError(f"AdamW local moment identity changed for {name!r}")
            swap_api.execute(name, "D2H")

    optimizer.register_step_pre_hook(pre_hook)
    optimizer.register_step_post_hook(post_hook)


@override(
    target=OptimizersContainer.Config,
    description="Offload DistributedMuon and AdamW state by globally unique names",
)
def muon_state_swap(cfg: OptimizersContainer.Config) -> MuonSwapOptimizersContainer.Config:
    return derive(cfg, MuonSwapOptimizersContainer.Config)


@override(
    target=CheckpointManager.Config,
    description="Reject checkpoint paths while Muon swap v1 owns offloaded optimizer state",
    exact=True,
)
def muon_state_swap_checkpoint(
    cfg: CheckpointManager.Config,
) -> MuonSwapCheckpointManager.Config:
    return derive(cfg, MuonSwapCheckpointManager.Config)
