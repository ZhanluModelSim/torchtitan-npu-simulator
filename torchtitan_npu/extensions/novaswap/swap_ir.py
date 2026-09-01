# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

from dataclasses import dataclass


@dataclass
class SwapAction:
    """Define one memory swap action, the core Swap IR unit."""

    # 1. Target object: globally unique tensor identifier.
    tensor_name: str

    # 2. Trigger point: where this action fires in the graph lifecycle.
    hook_name: str

    # 3. Swap direction or operation type.
    action_type: str

    @property
    def tensor(self) -> str:
        return self.tensor_name

    @property
    def type(self) -> str:
        return self.action_type


class ActionManager:
    """Global swap action manager indexed by hook name."""

    # Key: hook_name. Value: actions executed when that hook fires.
    actions_by_hook: dict[str, list[SwapAction]] = {}

    @classmethod
    def add_action(cls, action: SwapAction):
        """Add one swap instruction under its hook name."""
        cls.actions_by_hook.setdefault(action.hook_name, []).append(action)

    @classmethod
    def get_actions_by_name(cls, hook_name: str) -> list[SwapAction]:
        """Return actions for the backend swap engine when a hook fires."""
        return cls.actions_by_hook.get(hook_name, [])

    @classmethod
    def clear(cls):
        """Clear all actions after each iteration."""
        cls.actions_by_hook.clear()
