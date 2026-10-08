# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import nn

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture
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
