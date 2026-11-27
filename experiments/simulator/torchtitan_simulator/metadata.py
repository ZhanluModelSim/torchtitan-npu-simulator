# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lift host-built native metadata, preserving aliases and host split vectors."""

import copy
from dataclasses import fields, is_dataclass

import torch

# torch 2.14's BlockMask is not a dataclass instance, so the generic lift
# below cannot discover its tensor fields; visit them explicitly.
_BLOCKMASK_TENSOR_FIELDS = (
    "seq_lengths",
    "kv_num_blocks",
    "kv_indices",
    "full_kv_num_blocks",
    "full_kv_indices",
    "q_num_blocks",
    "q_indices",
    "full_q_num_blocks",
    "full_q_indices",
    "dq_write_order",
    "dq_write_order_full",
    "dq_kv_order",
    "dq_kv_order_spt",
)


_ATTENTION_HOST_FIELDS = {
    "cu_seq_q",
    "cu_seq_k",
    "cu_seqlens",
    "cu_seqlens_q",
    "cu_seqlens_ori_kv",
    "cu_seqlens_cmp_k",
    "seqused",
}


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


def to_meta_metadata(value, *, host_splits=()):
    # Seed before the walk: an alias can occur before the plan or under an
    # unrelated key. Only the planner knows which lists carry route identity.
    memo = {id(split): split for split in host_splits}

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
                source = getattr(item, field.name)
                lifted = lift(source, field_name=field.name)
                if (
                    field.name in _ATTENTION_HOST_FIELDS
                    and isinstance(source, torch.Tensor)
                    and source.device.type == "cpu"
                    and source.ndim == 1
                    and source.dtype in (torch.int32, torch.int64)
                ):
                    lifted._sim_host_attention_values = tuple(source.tolist())
                object.__setattr__(result, field.name, lifted)
            return result
        if isinstance(item, torch.nn.attention.flex_attention.BlockMask):
            result = copy.copy(item)
            memo[id(item)] = result
            for name in _BLOCKMASK_TENSOR_FIELDS:
                value = getattr(item, name, None)
                if isinstance(value, torch.Tensor):
                    object.__setattr__(result, name, lift(value, field_name=field_name))
            return result
        if isinstance(item, dict):
            result = {key: lift(child) for key, child in item.items()}
        elif isinstance(item, tuple):
            field_names = getattr(item, "_fields", None)
            if field_names is None:
                result = tuple(lift(child) for child in item)
            else:
                values = {name: lift(child, field_name=name) for name, child in zip(field_names, item, strict=True)}
                if "cu_seq_q_host" in values and values["cu_seq_q_host"] is None:
                    offsets = item.cu_seq_q
                    if offsets.device.type == "cpu":
                        values["cu_seq_q_host"] = tuple(offsets.tolist())
                result = type(item)(**values)
        elif isinstance(item, list):
            result = [lift(child) for child in item]
        else:
            return item
        memo[id(item)] = result
        return result

    return lift(value)
