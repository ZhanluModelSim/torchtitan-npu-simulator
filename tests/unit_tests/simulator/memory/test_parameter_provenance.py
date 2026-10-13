# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives
from torchtitan_npu.simulator.capture.fsdp_residency import _record_residency
from torchtitan_npu.simulator.memory.estimator import estimate_static_memory


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_same_metadata_does_not_make_external_tensor_a_parameter(device):
    model = nn.Module()
    model.weight = nn.Parameter(torch.ones(4, device=device))
    ordinary = torch.ones_like(model.weight)
    alias = model.weight.detach()
    materialized = model.weight.clone().detach()
    capture = OpDispatchCapture()
    with capture:
        (ordinary + alias + materialized).sum()
    plan = estimate_static_memory(capture.memory_events(), model_parts=[model])
    external = [item for item in plan.tensor_lifetimes if item.kind == "external_input"]
    assert {item.tensor_id for item in external} == {
        f"external:{capture.tensor_id(ordinary)}",
        f"external:{capture.tensor_id(materialized)}",
    }
    assert sum(item.num_bytes for item in external) == 32
    assert plan.persistent_param_bytes == 16


def test_dtensor_local_and_shared_parameter_storage_have_provenance(tmp_path):
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=2)
    try:
        mesh = DeviceMesh("cpu", [0, 1])
        model = nn.Module()
        local = torch.ones(2, 4, device="meta")
        distributed = DTensor.from_local(local, mesh, [Shard(0)], run_check=False, shape=(4, 4), stride=(4, 1))
        model.weight = nn.Parameter(distributed)
        model.shared_weight = model.weight
        ordinary = torch.ones(2, 4, device="meta")
        capture = OpDispatchCapture()
        with capture:
            alias = model.weight.to_local()
            (alias + ordinary).sum()
        plan = estimate_static_memory(capture.memory_events(), model_parts=[model])
        assert plan.persistent_param_bytes == 32
        external = [item for item in plan.tensor_lifetimes if item.kind == "external_input"]
        assert [item.tensor_id for item in external] == [f"external:{capture.tensor_id(ordinary)}"]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("checkpoint_mode", ["none", "full", "selective"])
@pytest.mark.parametrize("weight_shape", [(8, 4), (2, 8, 4)])
def test_fsdp_storage_owns_transposed_parameter_views(checkpoint_mode, weight_shape):
    from contextlib import nullcontext
    from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

    weight = torch.empty(weight_shape, device="meta")
    transposed = weight.transpose(-1, -2)
    ordinary = torch.empty_like(transposed, requires_grad=True)
    group = SimpleNamespace(_module_fqn="layer", _training_state="forward")
    capture = OpDispatchCapture()

    def forward(x):
        return (x * transposed).sum()

    with capture_fake_collectives() as recorder, capture:
        _record_residency(group, "alloc", (weight.numel() * 4, (weight,)), 2, "f")
        if checkpoint_mode == "none":
            result = forward(ordinary)
        else:
            context_fn = (
                lambda: create_selective_checkpoint_contexts([torch.ops.aten.mul.Tensor])
            ) if checkpoint_mode == "selective" else lambda: (nullcontext(), nullcontext())
            result = checkpoint(forward, ordinary, use_reentrant=False, context_fn=context_fn)
        _record_residency(group, "free", (weight.numel() * 4, (weight,)), 2, "f")
        group._training_state = "backward"
        _record_residency(group, "alloc", (weight.numel() * 4, (weight,)), 2, "b")
        result.backward()
        _record_residency(group, "free", (weight.numel() * 4, (weight,)), 2, "b")

    plan = estimate_static_memory(capture.memory_events(), fsdp_residency_events=recorder.fsdp_residency_events)
    external = {item.tensor_id for item in plan.tensor_lifetimes if item.kind == "external_input"}
    assert f"external:{capture.tensor_id(transposed)}" not in external
    assert f"external:{capture.tensor_id(ordinary)}" in external
    intervals = [item for item in plan.tensor_lifetimes if item.kind == "fsdp_full_param"]
    assert [(item.birth_seq, item.death_seq) for item in intervals] == [
        (recorder.fsdp_residency_events[0].seq_idx, recorder.fsdp_residency_events[1].seq_idx),
        (recorder.fsdp_residency_events[2].seq_idx, recorder.fsdp_residency_events[3].seq_idx),
    ]
