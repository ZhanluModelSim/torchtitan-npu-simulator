# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tracks the current module call stack via forward hooks, so captured ops
can be tagged with the dotted module path (e.g. "layers.5.attention.wq")
that produced them. This tagging is what lets the HTML exporter (Task 15)
fold visually-identical repeated layers (e.g. 61 TransformerBlocks) instead
of rendering every op of every layer."""

from __future__ import annotations

import torch.nn as nn


class ModulePathTracker:
    """Context manager that maintains a stack of "currently executing
    module" names, updated via forward pre/post hooks on every submodule of
    `root`."""

    def __init__(self, root: nn.Module) -> None:
        self.root = root
        self.stack: list[str] = []
        self._handles: list[object] = []

    def __enter__(self) -> "ModulePathTracker":
        names = {id(module): name or module.__class__.__name__ for name, module in self.root.named_modules()}

        def pre_hook(module: nn.Module, _args: object) -> None:
            self.stack.append(names.get(id(module), module.__class__.__name__))

        def post_hook(module: nn.Module, _args: object, _output: object) -> None:
            if self.stack:
                self.stack.pop()

        for _, module in self.root.named_modules():
            self._handles.append(module.register_forward_pre_hook(pre_hook))
            self._handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, *_exc: object) -> None:
        for handle in self._handles:
            handle.remove()  # type: ignore[attr-defined]
        self._handles.clear()

    def current_path(self) -> str:
        return self.stack[-1] if self.stack else ""
