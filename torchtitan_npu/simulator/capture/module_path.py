# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Track the module owning each captured forward or backward operator."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten


class ModulePathTracker:
    """Return stable model FQNs for forward, recompute, and backward ops.

    Forward hooks preserve the existing exact path behavior. Lightweight
    output-gradient hooks mark the most specific module whose backward graph
    is about to run. Unlike full module backward hooks, tensor hooks do not
    insert view operators into the captured graph.
    """

    def __init__(self, root: nn.Module) -> None:
        self.root = root
        self.stack: list[str] = []
        self._backward_path = ""
        self._root_name = type(root).__name__
        self._has_engine_callback = False
        self._handles: list[object] = []

    def __enter__(self) -> "ModulePathTracker":
        names = {id(module): name or module.__class__.__name__ for name, module in self.root.named_modules()}

        def pre_hook(module: nn.Module, _args: object) -> None:
            self.stack.append(names.get(id(module), module.__class__.__name__))

        def post_hook(module: nn.Module, _args: object, output: object) -> None:
            if self.stack:
                self.stack.pop()
            name = names.get(id(module), module.__class__.__name__)
            tensors, _spec = tree_flatten(output)
            seen: set[int] = set()
            for tensor in tensors:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.requires_grad
                    or id(tensor) in seen
                ):
                    continue
                seen.add(id(tensor))
                self._handles.append(
                    tensor.register_hook(
                        lambda grad, module_name=name: self._mark_backward(
                            module_name,
                            grad,
                        )
                    )
                )

        for _, module in self.root.named_modules():
            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()  # type: ignore[attr-defined]
        self._handles.clear()
        self.stack.clear()
        self._backward_path = ""
        self._has_engine_callback = False

    def current_path(self) -> str:
        if self.stack:
            return self.stack[-1]
        return self._backward_path

    def _mark_backward(
        self,
        module_path: str,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        current = self._backward_path
        incoming_is_ancestor = current != module_path and (
            module_path == self._root_name
            or current.startswith(f"{module_path}.")
        )
        if not incoming_is_ancestor:
            self._backward_path = module_path

        if not self._has_engine_callback:
            self._has_engine_callback = True

            def clear_backward_path() -> None:
                self._backward_path = ""
                self._has_engine_callback = False

            torch.autograd.Variable._execution_engine.queue_callback(
                clear_backward_path
            )
        return grad
