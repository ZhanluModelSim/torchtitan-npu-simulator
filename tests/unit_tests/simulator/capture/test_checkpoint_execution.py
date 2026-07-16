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

from torchtitan_npu.simulator.capture.checkpoint_execution import (
    install_checkpoint_execution_tracking,
)
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.step_boundary import StepBoundaryTracker


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
        node.annotations["execution_kind"] == "backward" and not node.annotations["is_recompute"]
        for node in nodes
    )
    assert any(node.annotations["execution_kind"] == "original_forward" for node in nodes)


def test_selective_checkpoint_preserves_policy_and_excludes_saved_op_from_recompute():
    policy_calls: list[tuple[bool, str]] = []

    def policy(ctx, op, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
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
        node.annotations["raw_op_type"]
        for node in nodes
        if node.annotations["execution_kind"] == "recompute"
    }

    assert any(is_recompute for is_recompute, _ in policy_calls)
    assert "aten.addmm.default" in recompute_ops
    assert "aten.gelu.default" not in recompute_ops


def test_unwrapped_model_has_no_recompute_ops():
    nodes = _capture_checkpointed_step(nn.Sequential(nn.Linear(4, 4), nn.GELU()))

    assert not any(node.annotations["is_recompute"] for node in nodes)
    assert {node.annotations["execution_kind"] for node in nodes} == {
        "original_forward",
        "backward",
    }
