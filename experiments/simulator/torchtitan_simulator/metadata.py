# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lift host-built native metadata, preserving aliases and host split vectors."""

import copy
from dataclasses import fields, is_dataclass

import torch


def iter_metadata_tensors(value):
    """Visit tensor leaves of native metadata without retaining duplicate owners."""
    seen = set()

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            yield item
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                yield from visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                yield from visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                yield from visit(child)

    yield from visit(value)


def to_meta_metadata(value):
    memo = {}

    def lift(item, *, field_name=""):
        if field_name in {"send_splits_tensor", "recv_splits_tensor"}:
            return item
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, torch.Tensor):
            base = getattr(item, "_base", None)
            if item.device.type == "meta":
                result = item
            elif isinstance(base, torch.Tensor) and item.layout == torch.strided:
                lifted_base = lift(base)
                offset = lifted_base.storage_offset() + item.storage_offset() - base.storage_offset()
                result = lifted_base.as_strided(item.shape, item.stride(), offset)
            else:
                result = item.to("meta")
            memo[id(item)] = result
            return result
        if is_dataclass(item) and not isinstance(item, type):
            result = copy.copy(item)
            memo[id(item)] = result
            for field in fields(item):
                object.__setattr__(result, field.name, lift(getattr(item, field.name), field_name=field.name))
            return result
        if isinstance(item, dict):
            result = {key: lift(child) for key, child in item.items()}
        elif isinstance(item, tuple):
            values = tuple(lift(child) for child in item)
            result = type(item)(*values) if hasattr(item, "_fields") else values
        elif isinstance(item, list):
            result = [lift(child) for child in item]
        else:
            return item
        memo[id(item)] = result
        return result

    return lift(value)
