# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Source: https://gitcode.com/ascend-nova/novaswap
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class SwapGroupMember:
    """A tensor's authoritative byte range in a packed group payload."""

    tensor: torch.Tensor
    byte_offset: int
    num_bytes: int


@dataclass(frozen=True, slots=True)
class SwapGroupRegistration:
    """The logical ownership and packed layout of a swap group."""

    name: str
    members: tuple[SwapGroupMember, ...]
    total_bytes: int


@dataclass(frozen=True)
class SwapTensorEntry:
    """Logical swap payload and the physical tensors backing it."""

    payload: Any
    storage_tensors: tuple[Any, ...]
    group_registration: SwapGroupRegistration | None = None


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
    def _register_payload(
        cls,
        payload: Any,
        tensor_name: str,
        group_registration: SwapGroupRegistration | None = None,
    ) -> None:
        detached_payload = _detach_payload(payload)
        storage_tensors = tuple(_flatten_storage_tensors(detached_payload))
        if not storage_tensors:
            raise ValueError(f"Unified swap group '{tensor_name}' contains no tensors")
        if tensor_name in cls.tensor_registry:
            raise ValueError(f"Swap tensor name {tensor_name!r} is already registered")
        cls.tensor_registry[tensor_name] = SwapTensorEntry(
            payload=detached_payload,
            storage_tensors=storage_tensors,
            group_registration=group_registration,
        )

    @classmethod
    def register_tensor(cls, tensor: Any, tensor_name: str):
        """Register one tensor or a tensor list under one logical name."""
        if isinstance(tensor, (list, tuple)):
            cls._register_payload(tuple(tensor), tensor_name)
            return
        cls._register_payload(tensor, tensor_name)

    @classmethod
    def register_group(
        cls,
        tensors: tuple[torch.Tensor, ...],
        tensor_name: str,
    ) -> SwapGroupRegistration:
        """Register compact tensors as one packed transport payload."""
        if not tensors:
            raise ValueError("Swap group must contain at least one tensor")
        members = []
        total_bytes = 0
        seen_storages = set()
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise ValueError("Swap group members must be tensors")
            num_bytes = tensor.numel() * tensor.element_size()
            storage = tensor.untyped_storage()
            if not (tensor.is_contiguous() and tensor.storage_offset() == 0 and storage.size() == num_bytes):
                raise ValueError("Swap group members must be contiguous tensors with compact storage")
            storage_key = (str(tensor.device), storage.data_ptr(), storage.size())
            if storage_key in seen_storages:
                raise ValueError("Swap group members must not share storage")
            seen_storages.add(storage_key)
            alignment = tensor.element_size()
            total_bytes = (total_bytes + alignment - 1) // alignment * alignment
            members.append(SwapGroupMember(tensor, total_bytes, num_bytes))
            total_bytes += num_bytes
        registration = SwapGroupRegistration(tensor_name, tuple(members), total_bytes)
        cls._register_payload(tensors, tensor_name, registration)
        return registration

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
