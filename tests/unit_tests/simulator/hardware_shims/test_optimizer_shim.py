# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_npu.simulator.hardware_shims.optimizer_shim import _meta_safe_fused_adamw


@pytest.fixture(scope="module")
def fake_process_group():
    owned = not dist.is_initialized()
    if owned:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = "29714"
        dist.init_process_group("fake", rank=0, world_size=8)
    elif dist.get_world_size() != 8:
        pytest.skip("optimizer HSDP test requires an 8-rank fake process group")
    yield
    if owned:
        dist.destroy_process_group()


def test_meta_safe_fused_adamw_does_not_read_values_or_dispatch_updates():
    param = torch.empty(4, device="meta")
    grad = torch.empty_like(param)
    exp_avg = torch.empty_like(param)
    exp_avg_sq = torch.empty_like(param)
    state_step = torch.empty((), device="meta")

    capture = OpDispatchCapture()
    with capture:
        _meta_safe_fused_adamw(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [],
            [state_step],
            amsgrad=False,
            beta1=0.9,
            beta2=0.999,
            lr=1e-3,
            weight_decay=0.01,
            eps=1e-8,
            maximize=False,
        )

    event = next(
        event for event in capture.memory_events()
        if event.raw_op_type == "npu.npu_apply_adam_w.default"
    )
    assert len(event.inputs) == 5
    assert len(event.outputs) == 4
    assert {ref.tensor_id for ref in event.outputs} <= {
        ref.tensor_id for ref in event.inputs
    }


def test_hsdp_optimizer_node_uses_global_shapes_but_memory_uses_local_shapes(
    fake_process_group,
):
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard

    mesh = DeviceMesh(
        "meta",
        torch.arange(8).reshape(2, 4),
        mesh_dim_names=("dp_replicate", "fsdp"),
    )

    def hsdp_tensor() -> DTensor:
        local = torch.empty(1, 8, device="meta")
        return DTensor.from_local(
            local,
            mesh,
            [Replicate(), Shard(0)],
            shape=torch.Size((3, 8)),
            stride=(8, 1),
            run_check=False,
        )

    param, grad, exp_avg, exp_avg_sq = (hsdp_tensor() for _ in range(4))
    state_step = torch.empty((), device="meta")
    capture = OpDispatchCapture(phase_provider=lambda: "optimizer")

    with capture:
        _meta_safe_fused_adamw(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [],
            [state_step],
            amsgrad=False,
            beta1=0.9,
            beta2=0.999,
            lr=1e-3,
            weight_decay=0.01,
            eps=1e-8,
            maximize=False,
        )

    node = next(
        node for node in capture.build_nodes().values()
        if node.annotations["raw_op_type"] == "npu.npu_apply_adam_w.default"
    )
    event = next(
        event for event in capture.memory_events()
        if event.raw_op_type == "npu.npu_apply_adam_w.default"
    )

    assert [meta.shape for meta in node.inputs[:4]] == [(3, 8)] * 4
    assert [meta.shape for meta in node.outputs[:3]] == [(3, 8)] * 3
    assert node.annotations["tensor_shape_scope"] == "global"
    assert [ref.shape for ref in event.inputs[:4]] == [(1, 8)] * 4
    assert [ref.shape for ref in event.outputs[:3]] == [(1, 8)] * 3
