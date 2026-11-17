# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SwapTensorEntry:
    """Logical swap payload and the physical tensors backing it."""

    payload: Any
    storage_tensors: tuple[Any, ...]


def _detach_payload(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_detach_payload(item) for item in value)
    return value.detach()


def _flatten_storage_tensors(value: Any) -> Iterable[Any]:
    """Expand lists and quantized-tensor providers into real tensor leaves."""
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_storage_tensors(item)
        return

    provider = getattr(value, "get_storage", None)
    if provider is not None:
        for item in provider().values():
            if item is None:
                continue
            yield from _flatten_storage_tensors(item)
        return

    yield value.detach()


class SwapRegistry:
    """Global name-to-group registry and strong-reference owner."""

    # key: logical tensor/group name, value: SwapTensorEntry.
    tensor_registry: dict[str, SwapTensorEntry] = {}

    # hook_name
    hook_registry = set()

    # Tensor APIs

    @classmethod
    def _register_payload(cls, payload: Any, tensor_name: str) -> None:
        detached_payload = _detach_payload(payload)
        storage_tensors = tuple(_flatten_storage_tensors(detached_payload))
        if not storage_tensors:
            raise ValueError(f"Unified swap group '{tensor_name}' contains no tensors")
        if tensor_name in cls.tensor_registry:
            raise ValueError(f"Swap tensor name {tensor_name!r} is already registered")
        cls.tensor_registry[tensor_name] = SwapTensorEntry(
            payload=detached_payload,
            storage_tensors=storage_tensors,
        )

    @classmethod
    def register_tensor(cls, tensor: Any, tensor_name: str):
        """Register one tensor or a tensor list under one logical name."""
        if isinstance(tensor, (list, tuple)):
            cls._register_payload(tuple(tensor), tensor_name)
            return
        cls._register_payload(tensor, tensor_name)

    @classmethod
    def get_entry_by_name(cls, tensor_name: str):
        return cls.tensor_registry.get(tensor_name, None)

    @classmethod
    def pop_entry_by_name(cls, tensor_name: str):
        return cls.tensor_registry.pop(tensor_name, None)

    @classmethod
    def remove_entry_by_name(cls, tensor_name: str):
        return cls.pop_entry_by_name(tensor_name)

    @classmethod
    def clear_tensors(cls):
        cls.tensor_registry.clear()

    @classmethod
    def get_all_tensor_names(cls):
        return tuple(cls.tensor_registry.keys())

    # Hook APIs
    @classmethod
    def register_hook(cls, hook_name: str):
        cls.hook_registry.add(hook_name)

    @classmethod
    def remove_hook(cls, hook_name: str):
        return cls.hook_registry.discard(hook_name)

    @classmethod
    def clear_hooks(cls):
        cls.hook_registry.clear()

    @classmethod
    def get_hook_registry(cls):
        return cls.hook_registry
