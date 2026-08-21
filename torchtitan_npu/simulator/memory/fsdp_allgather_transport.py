# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Experimental FSDP all-gather transport precision model.

This does not change the tensor dtype used by model computation. It models an
external quantize/all-gather/dequantize implementation by changing only the
all-gather transport volume and its short-lived staging buffer.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torchtitan_npu.simulator.capture.tensor_utils import (
    normalize_supported_dtype,
    tensor_volume_bytes,
)
from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.memory.plugins import MemoryModelContext, MemoryModelPlugin
from torchtitan_npu.simulator.memory.records import TensorLifetime


def _field(event: Any, name: str, default: Any = "") -> Any:
    return getattr(event, name, default)


def _is_fsdp_allgather(event: Any) -> bool:
    return (
        _field(event, "comm_primitive") == "allgather"
        and "fsdp" in str(_field(event, "comm_dim")).lower()
    )


def apply_fsdp_allgather_transport_dtype(
    comm_events: Iterable[Any],
    nodes: dict[int, OpNode],
    dtype: str,
) -> int:
    """Override transport metadata after collective group resolution."""

    dtype = normalize_supported_dtype(dtype)
    overridden = 0
    for event in comm_events:
        if not _is_fsdp_allgather(event):
            continue
        captured_dtype = event.dtype
        captured_volume_bytes = event.volume_bytes
        event.dtype = dtype
        event.volume_bytes = tensor_volume_bytes(event.tensor_shape, dtype)
        node = nodes.get(event.op_id)
        if node is not None:
            node.comm_bytes = event.volume_bytes
            node.annotations.update(
                {
                    "transport_dtype": dtype,
                    "transport_bytes": event.volume_bytes,
                    "captured_transport_dtype": captured_dtype,
                    "captured_transport_bytes": captured_volume_bytes,
                }
            )
        overridden += 1
    return overridden


class FSDPAllGatherTransportPlugin(MemoryModelPlugin):
    """Model only FSDP all-gather staging storage in a transport dtype."""

    def __init__(self, dtype: str) -> None:
        self.dtype = normalize_supported_dtype(dtype)

    def apply(self, context: MemoryModelContext) -> list[TensorLifetime]:
        staged_tensor_ids = {
            ref.tensor_id
            for event in context.events
            if _is_fsdp_allgather(context.comm_by_op.get(event.op_id))
            for ref in event.outputs
        }
        overridden = 0
        for tensor_id in staged_tensor_ids:
            lifetime = context.lifetimes_by_tensor_id.get(tensor_id)
            if lifetime is None or lifetime.reason != "fsdp_allgather_staging":
                continue
            lifetime.modeled_num_bytes = tensor_volume_bytes(lifetime.shape, self.dtype)
            lifetime.modeled_dtype = self.dtype
            lifetime.reason = f"fsdp_allgather_staging_transport_{self.dtype}"
            overridden += 1

        context.notes.append(
            "FSDP all-gather transport override used "
            f"{self.dtype} for {overridden} staging buffers; full parameters remain at captured dtype."
        )
        return []
