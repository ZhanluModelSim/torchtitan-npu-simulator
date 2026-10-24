# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Closed-form lifetime oracles for optimizer state residency."""

from dataclasses import replace

import pytest
from torchtitan_simulator.memory.estimator import estimate_static_memory
from torchtitan_simulator.memory.records import RawMemoryEvent, TensorRef


def _ref(tensor_id, num_bytes, *, key, device="meta"):
    return TensorRef(
        tensor_id, "value", (num_bytes // 4,), "float32", device, num_bytes, storage_key=key, storage_bytes=num_bytes
    )


def _event(seq, phase, *, inputs=(), outputs=()):
    return RawMemoryEvent(seq, seq + 1, seq, "aten.add.Tensor", "elementwise", phase, "", inputs, outputs, phase)


def _capture(*, preinitialized=False):
    activation = _ref(1, 100, key="activation:0")
    gradient = _ref(2, 20, key="gradient:0")
    state = _ref(3, 80, key="state:0")
    view = replace(state, tensor_id=4, shape=(4,), num_bytes=16, alias_of=3)
    host_state = _ref(5, 4000, key="host:0", device="cpu")
    events = [
        _event(0, "forward", outputs=(activation,)),
        _event(2, "backward", inputs=(activation,), outputs=(gradient,)),
        _event(3, "optimizer", outputs=() if preinitialized else (state, view)),
        _event(4, "optimizer", inputs=(gradient, state, view)),
    ]
    return events, (state, view, host_state)


def test_cold_state_projection_deduplicates_storage_and_can_change_peak_phase():
    events, states = _capture()
    plan = estimate_static_memory(events, optimizer_state_tensors=states, counted_device_type="meta")
    assert (plan.forward_peak_active_bytes, plan.backward_peak_active_bytes, plan.optimizer_peak_active_bytes) == (
        100,
        120,
        100,
    )
    assert plan.peak_active_bytes == 120
    assert plan.optimizer_state_bytes == 80
    assert plan.optimizer_state_storage_count == 1
    assert (
        plan.steady_state_forward_peak_active_bytes,
        plan.steady_state_backward_peak_active_bytes,
        plan.steady_state_optimizer_peak_active_bytes,
    ) == (180, 200, 100)
    assert plan.steady_state_peak_active_bytes == 200
    assert sum(item.resident_num_bytes for item in plan.tensor_lifetimes if item.kind == "optimizer_state") == 80
    summary = plan.to_summary_dict()
    assert summary == {
        "steady_state_active_bytes_peak": 200,
        "steady_state_forward_active_bytes_peak": 180,
        "steady_state_backward_active_bytes_peak": 200,
        "steady_state_optimizer_active_bytes_peak": 100,
        "persistent_param_bytes": 0,
        "optimizer_state_bytes": 80,
    }


def test_preinitialized_state_is_already_resident_and_never_added_twice():
    events, states = _capture(preinitialized=True)
    plan = estimate_static_memory(
        events, initial_tensors=states, optimizer_state_tensors=states, counted_device_type="meta"
    )
    assert plan.optimizer_state_bytes == 80
    assert plan.peak_active_bytes == plan.steady_state_peak_active_bytes == 200
    assert plan.forward_peak_active_bytes == plan.steady_state_forward_peak_active_bytes == 180
    assert plan.backward_peak_active_bytes == plan.steady_state_backward_peak_active_bytes == 200
    assert plan.optimizer_peak_active_bytes == plan.steady_state_optimizer_peak_active_bytes == 100


def test_unknown_owner_is_distinct_from_optimizer_with_no_state():
    events, _ = _capture()
    unknown = estimate_static_memory(events)
    stateless = estimate_static_memory(events, optimizer_state_tensors=[])
    assert unknown.optimizer_state_bytes is None
    assert unknown.steady_state_peak_active_bytes is None
    assert unknown.to_summary_dict()["optimizer_state_bytes"] is None
    assert stateless.optimizer_state_bytes == 0
    assert stateless.steady_state_peak_active_bytes == stateless.peak_active_bytes == 120


def test_unwitnessed_optimizer_state_is_rejected_instead_of_fabricated():
    events, _ = _capture()
    with pytest.raises(ValueError, match="absent from the captured memory inventory"):
        estimate_static_memory(events, optimizer_state_tensors=[_ref(9, 8, key="missing:0")])


def test_initial_state_without_phase_events_still_has_a_total_residency_peak():
    state = _ref(1, 80, key="initial_state:0")
    plan = estimate_static_memory([], initial_tensors=[state], optimizer_state_tensors=[state])
    assert plan.optimizer_state_bytes == 80
    assert plan.peak_active_bytes == plan.steady_state_peak_active_bytes == 80
    assert plan.steady_state_forward_peak_active_bytes == 0
    assert plan.steady_state_backward_peak_active_bytes == 0
    assert plan.steady_state_optimizer_peak_active_bytes == 0
