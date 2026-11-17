# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Avoid unused empty Meta chunks inside audited FSDP initialization methods.

The upstream helper and its public list contract remain untouched. Only private
function namespaces of the two initialization consumers use the local sequence;
no parameter, chunk, or function replacement survives the parallelize scope.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from types import CellType, FunctionType
from typing import overload

import torch

from torchtitan_simulator.compat.patches import PatchScope


class _InitializationChunks(Sequence[torch.Tensor]):
    """Retain native nonempty chunks; materialize each requested empty once."""

    __slots__ = ("__weakref__", "_chunks", "_count", "_empty")

    def __init__(self, tensor: torch.Tensor, count: int, dim: int) -> None:
        self._chunks = torch.chunk(tensor, count, dim=dim)
        self._count = count
        self._empty: dict[int, torch.Tensor] = {}

    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> torch.Tensor: ...

    @overload
    def __getitem__(self, index: slice) -> list[torch.Tensor]: ...

    def __getitem__(self, index: int | slice) -> torch.Tensor | list[torch.Tensor]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(self._count))]
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError("list index out of range")
        if index < len(self._chunks):
            return self._chunks[index]
        if index not in self._empty:
            self._empty[index] = self._chunks[0].new_empty(0)
        return self._empty[index]


def _initialization_chunks(
    tensor: torch.Tensor,
    num_chunks: int,
    dim: int,
    *,
    native: Callable[[torch.Tensor, int, int], list[torch.Tensor]],
) -> Sequence[torch.Tensor]:
    if not tensor.is_meta:
        return native(tensor, num_chunks, dim)
    return _InitializationChunks(tensor, num_chunks, dim)


def _rebind_initializer(method: FunctionType) -> FunctionType:
    """Preserve decorator closures while rebinding the audited helper reference.

    FSDP's main initializer is decorated with no_grad. Cloning only __wrapped__
    would discard that behavior; cloning its closure preserves the decorator as
    well as any additional well-formed wrapper installed before this scope.
    """
    namespace = method.__globals__
    if "_chunk_with_empty" in method.__code__.co_names:
        namespace = dict(namespace)
        namespace["_chunk_with_empty"] = partial(_initialization_chunks, native=namespace["_chunk_with_empty"])
    wrapped = getattr(method, "__wrapped__", None)
    rebound_wrapped = _rebind_initializer(wrapped) if isinstance(wrapped, FunctionType) else None
    closure = method.__closure__
    if closure is not None and rebound_wrapped is not None:
        closure = tuple(CellType(rebound_wrapped) if cell.cell_contents is wrapped else cell for cell in closure)
    rebound = FunctionType(method.__code__, namespace, method.__name__, method.__defaults__, closure)
    rebound.__kwdefaults__ = method.__kwdefaults__
    rebound.__annotations__ = method.__annotations__
    rebound.__dict__.update(method.__dict__)
    rebound.__module__ = method.__module__
    rebound.__qualname__ = method.__qualname__
    rebound.__doc__ = method.__doc__
    if rebound_wrapped is not None:
        rebound.__dict__["__wrapped__"] = rebound_wrapped
    return rebound


@contextmanager
def meta_initialization() -> Iterator[None]:
    """Limit Meta chunk optimization to native FSDP initialization consumers."""
    import torch.distributed.fsdp._fully_shard._fsdp_param as fsdp_param

    with PatchScope() as patches:
        for name in ("_init_sharded_param", "_init_sharded_post_forward_param_metadata"):
            method = getattr(fsdp_param.FSDPParam, name)
            if not isinstance(method, FunctionType):
                continue
            patches.replace(fsdp_param.FSDPParam, name, _rebind_initializer(method))
        yield
