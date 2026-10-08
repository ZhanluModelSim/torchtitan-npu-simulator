# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext

import pytest
import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.saved_tensors import AutogradSavedTensorCapture
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory


class _Retained(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        saved = x.sin()
        ctx.save_for_backward(saved)
        return saved.cos()

    @staticmethod
    def backward(ctx, grad):
        (saved,) = ctx.saved_tensors
        return grad * saved + saved


def _capture(device, hooks):
    phase = ["forward"]
    capture = OpDispatchCapture(phase_provider=lambda: phase[0])
    x = torch.ones(4, device=device, requires_grad=True)
    with capture, AutogradSavedTensorCapture() if hooks else nullcontext():
        output = _Retained.apply(x)
        loss = output.sum()
        phase[0] = "backward"
        loss.backward()
    capture.finalize_autograd_saved_tensors()
    plan = estimate_static_memory(
        capture.memory_events(), autograd_saved_tensor_events=capture.autograd_saved_tensor_events()
    )
    return capture, plan, output, x.grad


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_saved_hook_preserves_forward_dependency_and_storage_accounting(device):
    _reference, reference_plan, expected, expected_grad = _capture(device, False)
    capture, plan, actual, actual_grad = _capture(device, True)
    producer = next(event for event in capture._events if event.raw_op_type == "aten.sin.default")
    consumers = [event for event in capture._events if event.raw_op_type in {"aten.mul.Tensor", "aten.add.Tensor"}]
    assert all(producer.op_id in event.predecessors for event in consumers)
    assert len([item for item in plan.tensor_lifetimes if item.kind == "external_input"]) == 1
    assert plan.peak_active_bytes == reference_plan.peak_active_bytes
    assert sum(item.num_bytes for item in plan.tensor_lifetimes) == sum(
        item.num_bytes for item in reference_plan.tensor_lifetimes
    )
    if device == "cpu":
        assert torch.equal(actual, expected)
        assert torch.equal(actual_grad, expected_grad)


def test_saved_alias_keeps_view_value_and_observed_mutation_dependencies():
    capture = OpDispatchCapture()
    with capture:
        base = torch.ones(8, device="meta")
        view = base[2:6]
        capture.record_autograd_saved_tensor_pack(view)
        with capture.suspend_recording():
            detached = view.detach()
        base.add_(1)
        detached.sum()
    events = capture._events
    view_event = next(event for event in events if event.raw_op_type == "aten.slice.Tensor")
    mutation = next(event for event in events if event.raw_op_type == "aten.add_.Tensor")
    consumer = next(event for event in events if event.raw_op_type == "aten.sum.default")
    assert view_event.op_id in consumer.predecessors
    assert mutation.op_id in consumer.predecessors


def test_capture_exit_releases_storage_witnesses():
    capture = OpDispatchCapture()
    with capture:
        tensor = torch.ones(4, device="meta")
        capture.record_autograd_saved_tensor_pack(tensor)
        assert capture._storage_identities
        assert capture._saved_value_ids
    assert not capture._storage_identities
    assert not capture._saved_value_ids


def test_dtensor_saved_hooks_track_local_storage(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        x = torch.ones(2, 4, device="meta", requires_grad=True)
        phase = ["forward"]
        capture = OpDispatchCapture(phase_provider=lambda: phase[0])
        with capture, AutogradSavedTensorCapture():
            distributed = DTensor.from_local(x, mesh, [Shard(0)], run_check=False, shape=(4, 4), stride=(4, 1))
            output = distributed.sin().cos().sum()
            phase[0] = "backward"
            output.backward()
        assert x.grad.shape == x.shape
        assert capture.autograd_saved_tensor_events()
        assert all(saved.storage_bytes == 32 for saved in capture.autograd_saved_tensor_events())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_storage_aliases_with_different_layouts_share_one_lifetime(device):
    base = torch.ones(8, device=device)
    left = base[:4].detach()
    right = base[4:].detach()
    capture = OpDispatchCapture()
    with capture:
        left.sum()
        right.sum()
    plan = estimate_static_memory(capture.memory_events())
    external = [item for item in plan.tensor_lifetimes if item.kind == "external_input"]
    assert len(external) == 1
    assert external[0].num_bytes == 32
    assert external[0].death_seq == capture.memory_events()[-1].seq_idx


def test_synthetic_output_alias_does_not_allocate_saved_storage_twice():
    capture = OpDispatchCapture()
    with capture:
        original = torch.ones(4, device="meta")
        capture.record_autograd_saved_tensor_pack(original)
        with capture.suspend_recording():
            returned = original.detach()
        capture.record_synthetic_op("custom.identity", [original], [returned])
        returned.sum()
    plan = estimate_static_memory(capture.memory_events())
    allocations = [item for item in plan.tensor_lifetimes if item.producer_raw_op == "custom.identity"]
    assert all(item.num_bytes == 0 for item in allocations)
    root = next(item for item in plan.tensor_lifetimes if item.tensor_id == f"tensor:{capture.tensor_id(original)}")
    assert root.death_seq == capture.memory_events()[-1].seq_idx


def test_producer_lookup_observes_writes_through_detached_storage_alias():
    capture = OpDispatchCapture()
    with capture:
        base = torch.ones(4, device="meta")
        capture.record_autograd_saved_tensor_pack(base)
        with capture.suspend_recording():
            detached = base.detach()
        detached.add_(1)
        writer = capture._events[-1].op_id
        assert capture.producer_op(base) == writer
