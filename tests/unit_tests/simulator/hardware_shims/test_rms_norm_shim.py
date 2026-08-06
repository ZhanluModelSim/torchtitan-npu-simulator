# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
from torchtitan.models.common.rmsnorm import RMSNorm

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.rms_norm_shim import (
    SimRMSNorm,
    run_meta_rms_norm,
)


@pytest.fixture(scope="module")
def fake_process_group():
    owned = not dist.is_initialized()
    if owned:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = "29715"
        dist.init_process_group("fake", rank=0, world_size=8)
    elif dist.get_world_size() != 8:
        pytest.skip("RMSNorm DTensor test requires an 8-rank fake process group")
    yield
    if owned:
        dist.destroy_process_group()


def test_sim_rms_norm_records_forward_and_backward_and_preserves_gradients():
    shim = SimRMSNorm(RMSNorm(RMSNorm.Config(normalized_shape=16, eps=1e-6)))
    x = torch.empty((2, 3, 16), device="meta", requires_grad=True)
    phase = {"value": "forward"}
    capture = OpDispatchCapture(phase_provider=lambda: phase["value"])

    with capture:
        output = shim(x)
        phase["value"] = "backward"
        output.sum().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert shim.weight.grad is not None
    assert shim.weight.grad.shape == shim.weight.shape

    nodes = capture.build_nodes().values()
    forward_names = {node.annotations["raw_op_type"] for node in nodes if node.annotations["phase"] == "forward"}
    backward_names = {node.annotations["raw_op_type"] for node in nodes if node.annotations["phase"] == "backward"}
    assert "npu.npu_rms_norm.default" in forward_names
    assert "npu.npu_rms_norm_backward.default" in backward_names


def test_sim_rms_norm_preserves_dtensor_placement(fake_process_group):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    mesh = DeviceMesh.from_group(
        dist.group.WORLD,
        "meta",
        mesh=list(range(8)),
    )
    local_x = torch.empty(
        (2, 2, 16),
        device="meta",
        requires_grad=True,
    )
    local_weight = torch.empty(16, device="meta", requires_grad=True)
    x = DTensor.from_local(
        local_x,
        mesh,
        [Shard(1)],
        shape=torch.Size((2, 16, 16)),
        stride=(256, 16, 1),
        run_check=False,
    )
    weight = DTensor.from_local(
        local_weight,
        mesh,
        [Replicate()],
        run_check=False,
    )

    output = run_meta_rms_norm(x, weight, 1e-5)
    output.to_local().sum().backward()

    assert isinstance(output, DTensor)
    assert output.placements == x.placements
    assert output.shape == x.shape
    assert local_x.grad is not None
    assert local_weight.grad is not None
