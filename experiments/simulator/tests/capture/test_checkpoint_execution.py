# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import (
    CheckpointPolicy,
    DefaultDeviceType,
    create_selective_checkpoint_contexts,
)
from torchtitan_simulator.capture.checkpoint_execution import (
    RECOMPUTE,
    _compose_context_fn,
    current_execution_kind,
    install_checkpoint_execution_tracking,
)
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.module_path import ModulePathTracker
from torchtitan_simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_simulator.capture.step_boundary import StepBoundaryTracker
from torchtitan_simulator.memory.estimator import estimate_static_memory


def _capture_checkpointed_step(model: nn.Module) -> list:
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        boundary = StepBoundaryTracker()
        capture = OpDispatchCapture(phase_provider=lambda: boundary.current_phase)
        inputs = torch.randn(2, 4, requires_grad=True)
        with boundary, capture:
            model(inputs).sum().backward()
        return list(capture.build_nodes().values())
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)


def test_full_checkpoint_marks_only_replayed_ops_as_recompute():
    model = checkpoint_wrapper(
        nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4)),
        preserve_rng_state=False,
    )
    assert install_checkpoint_execution_tracking([model]) == 1
    assert install_checkpoint_execution_tracking([model]) == 0

    nodes = _capture_checkpointed_step(model)
    recompute_nodes = [node for node in nodes if node.annotations["is_recompute"]]

    assert recompute_nodes
    assert all(node.annotations["phase"] == "backward" for node in recompute_nodes)
    assert all(node.annotations["execution_kind"] == "recompute" for node in recompute_nodes)
    assert any(
        node.annotations["execution_kind"] == "backward" and not node.annotations["is_recompute"] for node in nodes
    )
    assert any(node.annotations["execution_kind"] == "original_forward" for node in nodes)


def test_checkpoint_wrapper_records_original_forward_boundary():
    model = nn.Sequential(
        checkpoint_wrapper(
            nn.Sequential(nn.Linear(4, 4), nn.GELU()),
            preserve_rng_state=False,
        )
    )
    assert install_checkpoint_execution_tracking([model]) == 1
    capture = OpDispatchCapture()
    inputs = torch.randn(2, 4, requires_grad=True)
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        with capture:
            model(inputs).sum().backward()
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)

    boundaries = capture.checkpoint_boundary_events()
    assert len(boundaries) == 1
    assert boundaries[0].checkpoint_id == "part0:0"
    assert boundaries[0].inputs[0].shape == (2, 4)
    assert boundaries[0].inputs[0].requires_grad is True
    assert boundaries[0].outputs[0].shape == (2, 4)


def test_selective_checkpoint_preserves_policy_and_excludes_saved_op_from_recompute():
    policy_calls: list[tuple[bool, str]] = []

    def policy(ctx, op, *args, **kwargs):
        policy_calls.append((ctx.is_recompute, str(op)))
        if op == torch.ops.aten.gelu.default:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    model = checkpoint_wrapper(
        nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 4)),
        context_fn=lambda: create_selective_checkpoint_contexts(policy),
        preserve_rng_state=False,
    )
    assert install_checkpoint_execution_tracking([model]) == 1

    nodes = _capture_checkpointed_step(model)
    recompute_ops = {
        node.annotations["raw_op_type"] for node in nodes if node.annotations["execution_kind"] == "recompute"
    }

    # New checkpoint policies are queried only in forward; replay uses the
    # cached decisions. The captured replay, rather than this deprecated
    # context flag, proves the selected operation was saved.
    assert any(op == "aten.gelu.default" for _, op in policy_calls)
    assert "aten.addmm.default" in recompute_ops
    assert "aten.gelu.default" not in recompute_ops


def test_unwrapped_model_has_no_recompute_ops():
    nodes = _capture_checkpointed_step(nn.Sequential(nn.Linear(4, 4), nn.GELU()))

    assert not any(node.annotations["is_recompute"] for node in nodes)
    assert {node.annotations["execution_kind"] for node in nodes} == {
        "original_forward",
        "backward",
    }


def test_full_checkpoint_recompute_context_can_be_reentered():
    _, recompute_context = _compose_context_fn(None)()

    # DualPipeV may split one checkpointed backward into input- and
    # weight-gradient passes. Full AC uses a reusable nullcontext; our marker
    # must preserve that property.
    with recompute_context:
        assert current_execution_kind("backward") == RECOMPUTE
    with recompute_context:
        assert current_execution_kind("backward") == RECOMPUTE


def test_full_checkpoint_boundary_input_stays_live_until_backward_unpack():
    model = nn.Sequential(checkpoint_wrapper(nn.Sequential(nn.Linear(4, 4), nn.GELU())))
    assert install_checkpoint_execution_tracking([model]) == 1
    boundary = StepBoundaryTracker()
    tracker = ModulePathTracker(model)
    capture = OpDispatchCapture(
        module_path_tracker=tracker,
        phase_provider=lambda: boundary.current_phase,
    )
    inputs = torch.randn(2, 4, requires_grad=True)
    previous_device_type = DefaultDeviceType.get_device_type()
    DefaultDeviceType.set_device_type("cpu")
    try:
        with boundary, tracker, capture, AutogradSavedTensorCapture():
            model(inputs).sum().backward()
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)
    capture.finalize_autograd_saved_tensors()

    plan = estimate_static_memory(
        capture.memory_events(),
        model_parts=[model],
        checkpoint_boundary_events=capture.checkpoint_boundary_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
    )
    saved = next(item for item in plan.tensor_lifetimes if item.kind == "checkpoint_saved_activation")
    unpack_seq = max(
        event.unpack_seq
        for event in capture.autograd_saved_tensor_events()
        if event.phase == "forward" and event.execution_kind == "original_forward" and event.unpack_seq >= 0
    )

    input_key = next(
        ref.storage_key for raw in capture.memory_events() for ref in raw.inputs if ref.tensor_id == id(inputs)
    )
    slots = [event for event in capture.autograd_saved_tensor_events() if event.storage_key == input_key]
    assert slots and all(event.release_tracking for event in slots)
    capture_end = max(event.seq_idx for event in capture.memory_events()) + 1
    release_seq = max(event.release_seq if event.release_seq >= 0 else capture_end for event in slots)
    assert saved.death_seq == max(unpack_seq, release_seq, *saved.consumer_seqs)
    assert saved.death_seq > max(event.seq_idx for event in capture.memory_events() if event.phase == "forward")

    offloaded_plan = estimate_static_memory(
        capture.memory_events(),
        model_parts=[model],
        checkpoint_boundary_events=capture.checkpoint_boundary_events(),
        autograd_saved_tensor_events=capture.autograd_saved_tensor_events(),
        offload_ac_saved_tensors=True,
    )
    offloaded = next(item for item in offloaded_plan.tensor_lifetimes if item.kind == "checkpoint_saved_activation")
    assert offloaded.death_seq == max(unpack_seq, release_seq, *offloaded.consumer_seqs)
    assert offloaded.residency_policy == "offloaded"
    assert offloaded.resident_num_bytes == 0
