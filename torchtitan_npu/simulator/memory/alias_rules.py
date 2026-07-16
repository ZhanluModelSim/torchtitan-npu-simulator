# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conservative allocation classification rules for meta tensor captures."""

from __future__ import annotations

from torchtitan_npu.simulator.memory.records import RawMemoryEvent

_ALIAS_TOKENS = (
    "view",
    "reshape",
    "transpose",
    "permute",
    "slice",
    "select",
    "split",
    "narrow",
    "as_strided",
    "squeeze",
    "unsqueeze",
    "detach",
    "t.default",
)

_ALLOC_TOKENS = (
    "clone",
    "contiguous",
    "empty",
    "zeros",
    "ones",
    "randn",
    "rand",
    "cat",
    "stack",
)


def is_alias_event(event: RawMemoryEvent) -> bool:
    if not event.inputs or not event.outputs:
        return False
    raw = event.raw_op_type.lower()
    if any(token in raw for token in _ALLOC_TOKENS):
        return False
    if any(token in raw for token in _ALIAS_TOKENS):
        return True
    input_ids = {ref.tensor_id for ref in event.inputs}
    return any(ref.tensor_id in input_ids for ref in event.outputs)


def is_mutation_event(event: RawMemoryEvent) -> bool:
    if not event.inputs or not event.outputs:
        return False
    input_ids = {ref.tensor_id for ref in event.inputs}
    if any(ref.tensor_id in input_ids for ref in event.outputs):
        return True
    raw = event.raw_op_type.lower()
    op_name = raw.split("::")[-1].split(".")[0]
    return op_name.endswith("_") or "copy_" in raw or "foreach" in raw
