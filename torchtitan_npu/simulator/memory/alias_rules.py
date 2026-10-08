# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Conservative allocation classification rules for meta tensor captures."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan_npu.simulator.memory.records import RawMemoryEvent

# These operators alias their first tensor input. Match namespace and exact
# operator name: token_permute, slice_backward and view_copy allocate storage.
# Conditional views such as reshape are captured as view or clone + _unsafe_view.
_ALIAS_OPERATORS = frozenset(
    {
        "aten._reshape_alias",
        "aten._unsafe_view",
        "aten.alias",
        "aten.as_strided",
        "aten.detach",
        "aten.expand",
        "aten.narrow",
        "aten.permute",
        "aten.select",
        "aten.slice",
        "aten.split",
        "aten.split_with_sizes",
        "aten.squeeze",
        "aten.t",
        "aten.transpose",
        "aten.unsqueeze",
        "aten.view",
        "aten.view_as_complex",
        "aten.view_as_real",
    }
)


def is_alias_event(event: RawMemoryEvent) -> bool:
    if not event.inputs or not event.outputs:
        return False
    operator = ".".join(event.raw_op_type.replace("::", ".").split(".")[:2])
    if operator in _ALIAS_OPERATORS:
        return True
    input_ids = {ref.tensor_id for ref in event.inputs}
    # Identity-preserving outputs do not prove that other outputs are aliases.
    # The estimator handles each unchanged input/output ID as a mutation.
    return all(ref.tensor_id in input_ids for ref in event.outputs)


def is_mutation_event(event: RawMemoryEvent) -> bool:
    if not event.inputs or not event.outputs:
        return False
    input_ids = {ref.tensor_id for ref in event.inputs}
    if any(ref.tensor_id in input_ids for ref in event.outputs):
        return True
    raw = event.raw_op_type.lower()
    op_name = raw.split("::")[-1].split(".")[0]
    return op_name.endswith("_") or "copy_" in raw or "foreach" in raw
