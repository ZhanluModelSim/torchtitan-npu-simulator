# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

"""Global swap API keyed by globally unique tensor names."""

from collections.abc import Sequence

import torch

# TORCHTITAN-NPU MOD: resolve the copied backend from the plugin package.
# Remove when: this backend is moved to a shared upstream swap package.
from .swap_engine import SwapEngine
from .swap_ir import ActionManager, SwapAction
from .swap_registry import SwapRegistry

metainfo = {}


def register_tensor(tensor: torch.Tensor | Sequence[torch.Tensor], tensor_name: str) -> None:
    SwapRegistry.register_tensor(tensor, tensor_name)


def pop_tensor_by_name(tensor_name: str):
    SwapEngine.remove(tensor_name)
    entry = SwapRegistry.pop_entry_by_name(tensor_name)
    return None if entry is None else entry.payload


def remove_tensor(tensor_name: str):
    SwapEngine.remove(tensor_name)
    entry = SwapRegistry.remove_entry_by_name(tensor_name)
    return None if entry is None else entry.payload


def clear_tensors() -> None:
    SwapEngine.reset()
    SwapRegistry.clear_tensors()


def get_all_tensor_names():
    return SwapRegistry.get_all_tensor_names()


def register_hook(hook_name: str) -> None:
    SwapRegistry.register_hook(hook_name)
    SwapEngine.on_operator_hook(hook_name)


def remove_hook(hook_name: str):
    return SwapRegistry.remove_hook(hook_name)


def clear_hooks() -> None:
    SwapRegistry.clear_hooks()


def get_hook_registry():
    return SwapRegistry.get_hook_registry()


def execute(tensor_name: str, action_type: str) -> None:
    SwapEngine.execute(SwapAction(tensor_name, "", action_type))


def submit(tensor_name: str, action_type: str, hook_name: str) -> None:
    ActionManager.add_action(SwapAction(tensor_name, hook_name, action_type))


def clear_actions() -> None:
    ActionManager.clear()


def get_handle_phase(tensor_name: str) -> str | None:
    return SwapEngine.handle_phase(tensor_name)


def shutdown() -> None:
    SwapEngine.shutdown()
