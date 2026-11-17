# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One per-layer transfer payload, deduplicated by allocation identity."""

import json

import pytest
from torchtitan_simulator.memory.export import export_memory_summary
from torchtitan_simulator.memory.records import (
    CheckpointTensorRecord,
    MemoryPlan,
    RawMemoryEvent,
    TensorLifetime,
)


@pytest.mark.parametrize("enabled", [False, True], ids=["resident", "offloaded"])
def test_layer_payload_export_deduplicates_views_and_checkpoint_roles_without_doubling_directions(enabled, tmp_path):
    events = [
        RawMemoryEvent(i, i, i, "aten.sin.default", "sin", "forward", name, (), (), "original_forward")
        for i, name in enumerate(("layers.10.norm", "layers.2.attention", "layers.3.norm", "mtp_layers.0.norm"))
    ]
    lifetimes = [
        TensorLifetime("a", "activation", 64, 0, 10, 0, storage_key="shared:0"),
        TensorLifetime("view", "activation", 64, 0, 10, 0, storage_key="shared:0"),
        TensorLifetime("b", "activation", 32, 0, 10, 0, storage_key="other:0"),
    ]

    def record(layer, tensor, size, role):
        return CheckpointTensorRecord(
            checkpoint_id=f"part0:{layer}",
            marker_kind="layer",
            seq_idx=0,
            tensor_id=tensor,
            role=role,
            shape=(size // 4,),
            dtype="float32",
            num_bytes=size,
            modeled_num_bytes=0,
            requires_grad=True,
            is_saved_activation=True,
            residency_policy="offloaded",
        )

    plan = MemoryPlan(
        offload_ac_saved_tensors=enabled,
        raw_events=events,
        tensor_lifetimes=lifetimes,
        checkpoint_tensors=[
            record("layers.2", "a", 16, "input"),
            record("layers.2", "view", 16, "recompute_saved"),
            record("layers.2", "b", 32, "input"),
            record("mtp_layers.0", "b", 32, "input"),
        ],
        activation_offload_tensors=[record("layers.2", "a", 16, "activation_saved")],
    )
    export_memory_summary(plan, str(tmp_path))
    summary = json.loads((tmp_path / "memory/memory_summary.json").read_text())
    # A shared 64-byte allocation plus a separate 32-byte allocation: one
    # payload of 96, not the 16-byte view, duplicated save slots, or 2x96.
    assert summary["activation_offload_by_layer"] == [
        {"layer": "layers.2", "transfer_bytes": 96 if enabled else 0},
        {"layer": "layers.3", "transfer_bytes": 0},
        {"layer": "layers.10", "transfer_bytes": 0},
        {"layer": "mtp_layers.0", "transfer_bytes": 32 if enabled else 0},
    ]
