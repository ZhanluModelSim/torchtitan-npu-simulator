# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Interleaved loss saves must occupy device memory without an offload charge."""

import pytest
from torchtitan_simulator.memory.estimator import estimate_static_memory
from torchtitan_simulator.memory.records import (
    AutogradSavedTensorEvent,
    CheckpointBoundaryEvent,
    RawMemoryEvent,
    TensorRef,
)


def _ref(tensor, size):
    return TensorRef(
        tensor, str(tensor), (size // 4,), "float32", "meta", size, storage_key=f"{tensor}:0", storage_bytes=size
    )


def _event(seq, inputs, outputs, *, block, phase="forward", path="", kind=None):
    return RawMemoryEvent(
        seq,
        seq,
        seq,
        "aten.test.default",
        "test",
        phase,
        path,
        tuple(inputs),
        tuple(outputs),
        kind or ("backward" if phase == "backward" else "original_forward"),
        pp_stage=0,
        pp_mb_idx=0,
        comp_type=block,
    )


@pytest.mark.parametrize("late_use", ["none", "decoder", "recompute"])
@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "checkpoint-input"])
def test_loss_storage_is_resident_unless_shared_with_later_decoder_backward(late_use, checkpointed):
    decoder, loss = _ref(1, 64), _ref(2, 128)
    events = [
        _event(0, [], [decoder], block="F", path="layers.0"),
        _event(1, [], [loss], block="F"),
        _event(2, [loss], [], block="F", phase="backward"),
        _event(4, [decoder, *([loss] if late_use != "none" else [])], [], block="B", phase="backward", path="layers.0"),
    ]
    if late_use == "recompute":
        events.append(_event(3, [loss], [], block="B", phase="backward", path="layers.1", kind="recompute"))
        # Only the recompute reads loss storage; there is no second save slot
        # and no ordinary B/backward consumer for this allocation.
        events[3] = _event(4, [decoder], [], block="B", phase="backward", path="layers.0")
        events.sort(key=lambda item: item.seq_idx)
    saves = [
        AutogradSavedTensorEvent(
            0,
            1,
            "1:0",
            64,
            64,
            (16,),
            "float32",
            0,
            unpack_seq=4,
            comp_type="F",
            pp_stage=0,
            pp_mb_idx=0,
            phase="forward",
            execution_kind="original_forward",
        ),
        AutogradSavedTensorEvent(
            1,
            2,
            "2:0",
            128,
            128,
            (32,),
            "float32",
            1,
            unpack_seq=2,
            comp_type="F",
            pp_stage=0,
            pp_mb_idx=0,
            phase="forward",
            execution_kind="original_forward",
        ),
    ]
    if late_use == "decoder":
        saves.append(
            AutogradSavedTensorEvent(
                2,
                2,
                "2:0",
                128,
                128,
                (32,),
                "float32",
                1,
                unpack_seq=4,
                comp_type="F",
                pp_stage=0,
                pp_mb_idx=0,
                phase="forward",
                execution_kind="original_forward",
            )
        )
    boundaries = [CheckpointBoundaryEvent("part0:layers.1", 1, (loss,), ())] if checkpointed else []
    resident = estimate_static_memory(events, autograd_saved_tensor_events=saves, checkpoint_boundary_events=boundaries)
    projected = estimate_static_memory(
        events,
        autograd_saved_tensor_events=saves,
        checkpoint_boundary_events=boundaries,
        offload_ac_saved_tensors=True,
    )
    loss_lifetime = next(item for item in projected.tensor_lifetimes if item.storage_key == "2:0")
    assert loss_lifetime.resident_num_bytes == (128 if late_use == "none" else 0)
    if late_use == "none":
        assert loss_lifetime.residency_policy == "resident"
        assert (loss_lifetime.birth_seq, loss_lifetime.death_seq) == (1, 2)
        assert projected.peak_active_bytes == 128
        assert resident.peak_active_bytes == 64 + 128
        assert all(record.tensor_id != loss_lifetime.tensor_id for record in projected.activation_offload_tensors)
        assert all(
            record.residency_policy == "resident" and record.modeled_num_bytes == 128
            for record in projected.checkpoint_tensors
            if record.tensor_id == loss_lifetime.tensor_id
        )
        assert all(
            item["transfer_bytes"] == 0
            for item in projected.to_summary_dict()["activation_offload_by_layer"]
            if item["layer"] == "layers.1"
        )
    assert all(item.resident_num_bytes == 0 for item in projected.tensor_lifetimes if item.storage_key == "1:0")


@pytest.mark.parametrize("mode", ["none", "full", "selective"])
@pytest.mark.parametrize(
    "family,vocab", [("deepseek_v4_smoketest", 129280), ("kimi_k3_smoketest", 2048)], ids=["dsv4", "kimi"]
)
def test_native_chunked_loss_storages_are_kept_in_device_timeline_and_excluded_from_transfer(mode, family, vocab):
    from torchtitan_simulator.config import SimulationConfig
    from torchtitan_simulator.runner import simulate

    workload = simulate(
        SimulationConfig(
            config=family,
            seq_len=128,
            activation_checkpoint_mode=mode,
            memory_offload_ac_saved_tensors=True,
        )
    )
    plan = workload.iteration.schedule.annotations["memory_plan"]
    loss_storages = [item for item in plan.tensor_lifetimes if item.producer_raw_op == "aten._log_softmax.default"]
    # Sequence=128 is split into eight 16-token chunks by the native loss.
    assert len(loss_storages) == 8
    for item in loss_storages:
        assert item.num_bytes == item.resident_num_bytes == 16 * vocab * 4
        assert item.residency_policy == "resident"
        assert any(
            event.action == "alloc" and event.tensor_id == item.tensor_id and event.num_bytes == 16 * vocab * 4
            for event in plan.timeline_events
        )
        assert any(
            event.action == "free" and event.tensor_id == item.tensor_id and event.num_bytes == 16 * vocab * 4
            for event in plan.timeline_events
        )
        assert all(record.tensor_id != item.tensor_id for record in plan.activation_offload_tensors)
    assert plan.peak_active_bytes >= 16 * vocab * 4
    events = {event.seq_idx: event for event in plan.raw_events}
    head_saves = [
        item
        for item in plan.tensor_lifetimes
        if item.reason == "autograd_saved_tensor"
        and events[item.birth_seq].module_path in {"norm", "hc_head", "output_attn_res.norm"}
    ]
    assert head_saves
    assert all(item.resident_num_bytes == item.num_bytes and item.residency_policy == "resident" for item in head_saves)
    assert not any("<unattributed>" in record.checkpoint_id for record in plan.activation_offload_tensors)
    assert any(
        item["transfer_bytes"] > 0
        for item in plan.to_summary_dict()["activation_offload_by_layer"]
        if item["layer"].startswith("layers.")
    )


@pytest.mark.parametrize("owner", ["hc_head", "norm", "output_attn_res", ""])
def test_non_layer_and_unknown_saves_stay_resident_instead_of_only_hiding_the_transfer_row(owner):
    decoder, head = _ref(1, 64), _ref(2, 128)
    events = [
        _event(0, [], [decoder], block="F", path="layers.0"),
        _event(1, [], [head], block="F", path=owner),
        _event(3, [head], [], block="B", phase="backward", path="layers.0" if owner else ""),
        _event(4, [decoder], [], block="B", phase="backward", path="layers.0"),
    ]
    saves = [
        AutogradSavedTensorEvent(
            0,
            1,
            "1:0",
            64,
            64,
            (16,),
            "float32",
            0,
            unpack_seq=4,
            phase="forward",
            execution_kind="original_forward",
            module_path="layers.0",
        ),
        AutogradSavedTensorEvent(
            1,
            2,
            "2:0",
            128,
            128,
            (32,),
            "float32",
            1,
            unpack_seq=3,
            phase="forward",
            execution_kind="original_forward",
            module_path=owner,
        ),
    ]
    plan = estimate_static_memory(events, autograd_saved_tensor_events=saves, offload_ac_saved_tensors=True)
    lifetime = next(item for item in plan.tensor_lifetimes if item.storage_key == "2:0")
    assert lifetime.resident_num_bytes == 128 and lifetime.residency_policy == "resident"
    assert plan.peak_active_bytes == 128
    assert plan.to_summary_dict()["activation_offload_by_layer"] == [{"layer": "layers.0", "transfer_bytes": 64}]


@pytest.mark.parametrize("other_save", [False, True], ids=["use-def-fallback", "exact-slots"])
def test_unbound_non_layer_recompute_storage_without_its_own_save_slot_is_not_offloaded(other_save):
    value, other = _ref(2, 128), _ref(3, 16)
    events = [
        _event(0, [], [value, *([other] if other_save else [])], block="F", path="hc_head._checkpoint_wrapped_module"),
        _event(
            1, [value], [], block="B", phase="backward", kind="recompute", path="hc_head._checkpoint_wrapped_module"
        ),
    ]
    saves = (
        [
            AutogradSavedTensorEvent(
                0,
                3,
                "3:0",
                16,
                16,
                (4,),
                "float32",
                0,
                unpack_seq=1,
                phase="forward",
                execution_kind="original_forward",
                module_path="hc_head",
            )
        ]
        if other_save
        else []
    )
    plan = estimate_static_memory(events, autograd_saved_tensor_events=saves, offload_ac_saved_tensors=True)
    lifetime = next(item for item in plan.tensor_lifetimes if item.storage_key == "2:0")
    assert lifetime.resident_num_bytes == 128 and lifetime.residency_policy == "resident"
    assert plan.peak_active_bytes == 128 + (16 if other_save else 0)
    assert plan.to_summary_dict()["activation_offload_by_layer"] == []
