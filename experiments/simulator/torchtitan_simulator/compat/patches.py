# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Scoped replacements with explicit ownership and restoration."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from inspect import getattr_static
from typing import Any

_MISSING = object()


@dataclass
class _Binding:
    owner: Any
    name: str
    previous: Any
    replacement: Any


class PatchScope(AbstractContextManager):
    """Own only installed bindings; never overwrite another owner's changes."""

    def __init__(self):
        self._bindings: list[_Binding] = []
        self._cleanups = []

    def defer(self, cleanup) -> None:
        self._cleanups.append(cleanup)

    def replace(self, owner: Any, name: str, replacement: Any) -> None:
        previous = getattr_static(owner, name, _MISSING) if name in vars(owner) else _MISSING
        setattr(owner, name, replacement)
        self._bindings.append(_Binding(owner, name, previous, replacement))

    def __exit__(self, exc_type, exc_value, traceback):
        conflicts = []
        for binding in reversed(self._bindings):
            if getattr_static(binding.owner, binding.name, _MISSING) is binding.replacement:
                if binding.previous is _MISSING:
                    delattr(binding.owner, binding.name)
                else:
                    setattr(binding.owner, binding.name, binding.previous)
            else:
                conflicts.append(binding.name)
        self._bindings.clear()
        for cleanup in reversed(self._cleanups):
            cleanup()
        self._cleanups.clear()
        if conflicts:
            raise RuntimeError(f"Simulator patch ownership changed during capture: {conflicts}")
        return False
