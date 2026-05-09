# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

import torchtitan.components.optimizer as tt_optimizer
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from torchtitan_npu.patches.optimizer.virtual_optimizer import (
    unwrap_dtensor,
    virtual_optimizer_step_impl,
    VirtualAllocator,
    wrap_like_param,
)


@pytest.fixture(scope="module", autouse=True)
def setup_dist_env():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    if dist.is_initialized():
        dist.destroy_process_group()


def test_unwrap_dtensor_logic():
    plain_tensor = torch.randn(4, 4)
    assert unwrap_dtensor(plain_tensor) is plain_tensor

    mesh = DeviceMesh("cpu", [0])
    dtensor = DTensor.from_local(plain_tensor, mesh, [Replicate()])
    result = unwrap_dtensor(dtensor)
    assert not isinstance(result, DTensor)
    assert torch.equal(result, plain_tensor)


def test_wrap_like_param():
    mesh = DeviceMesh("cpu", [0])
    local_tensor = torch.randn(4, 4)
    p = DTensor.from_local(torch.randn(4, 4), mesh, [Shard(0)])

    wrapped = wrap_like_param(local_tensor, p)

    assert isinstance(wrapped, DTensor)
    assert wrapped.device_mesh == p.device_mesh
    assert wrapped.placements == p.placements
    assert wrapped.size() == p.size()


def test_virtual_allocator_memory_logic():
    alloc = VirtualAllocator(pp_rank=0, pp_stages=2, virtual_optimizer_size=10.0)
    assert alloc.get_swap_memory_sizes() == [10.0, 10.0]

    alloc_list = VirtualAllocator(
        pp_rank=1, pp_stages=2, virtual_optimizer_size=[5.0, 15.0]
    )
    assert alloc_list.get_swap_memory_sizes() == [5.0, 15.0]

    p = torch.randn(2, 2)
    with patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu", spec=[]):
        mem = alloc.create(p)
        assert mem.shape == p.shape
        assert torch.all(mem == 0)


def test_build_optimizers_virtual_validation():
    config = types.SimpleNamespace(
        virtual_optimizer=True, swap_optimizer=True, virtual_optimizer_size=10.0
    )
    with pytest.raises(
        ValueError, match="Virtual optimizer does not support swap_optimizer"
    ):
        tt_optimizer.build_optimizers([], config, None, None)

    config_no_size = types.SimpleNamespace(
        virtual_optimizer=True, swap_optimizer=False, virtual_optimizer_size=None
    )
    with pytest.raises(ValueError, match="virtual_optimizer_size must be specified"):
        tt_optimizer.build_optimizers([], config_no_size, None, None)


@patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu")
@patch("torchtitan_npu.patches.optimizer.virtual_optimizer._original_build_optimizers")
def test_build_optimizers_success_path(mock_orig, mock_npu):
    mock_npu.npu.current_device.return_value = 0
    mock_opt = MagicMock(spec=torch.optim.AdamW)
    mock_opt.param_groups = [{"params": []}]
    mock_orig.return_value = [mock_opt]

    config = types.SimpleNamespace(
        virtual_optimizer=True,
        swap_optimizer=False,
        virtual_optimizer_size=20.0,
        name="AdamW",
    )

    results = tt_optimizer.build_optimizers(["part1"], config, None, None)

    assert len(results) == 1
    # Fix: Use getattr to access protected member for testing purposes
    config_val = results[0]._allocator_config
    assert config_val == (0, 1, 20.0)
    assert torch.optim.AdamW.step.__name__ == "virtual_optimizer_step"


@patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu")
def test_virtual_optimizer_step_impl_logic(mock_npu):
    mock_npu.npu.current_device.return_value = torch.device("cpu")

    def mock_npu_method(self, *args, **kwargs):
        return self

    torch.Tensor.npu = mock_npu_method

    mesh = DeviceMesh("cpu", [0])
    local_p = torch.randn(2, 2, requires_grad=True)
    local_p.grad = torch.randn(2, 2)

    p_dtensor = DTensor.from_local(local_p.detach(), mesh, [Replicate()])
    p_dtensor.grad = DTensor.from_local(local_p.grad, mesh, [Replicate()])

    class MockOptimizer:
        def __init__(self):
            self.param_groups = [
                {
                    "params": [p_dtensor],
                    "lr": 0.01,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "amsgrad": False,
                    "maximize": False,
                    "decoupled_weight_decay": True,
                }
            ]
            self.state = {p_dtensor: {}}
            self.virtual_allocator = MagicMock()
            self.virtual_allocator.init_exp.return_value = (
                torch.zeros(2, 2),
                torch.zeros(2, 2),
            )
            self.print_swap_flag = False

    opt = MockOptimizer()

    with patch("torch._fused_adamw_") as mock_fused:
        virtual_optimizer_step_impl(opt)
        assert "exp_avg" in opt.state[p_dtensor]
        assert isinstance(opt.state[p_dtensor]["exp_avg"], DTensor)
        assert mock_fused.called


@patch("torchtitan_npu.patches.optimizer.virtual_optimizer.torch_npu")
def test_virtual_optimizer_smoke_test(mock_npu):
    """Smoke test: verify complete optimizer step flow with multiple parameters."""
    mock_npu.npu.current_device.return_value = torch.device("cpu")

    def mock_npu_method(self, *args, **kwargs):
        return self

    torch.Tensor.npu = mock_npu_method

    mesh = DeviceMesh("cpu", [0])

    param1 = torch.randn(4, 4, requires_grad=True)
    param1.grad = torch.randn(4, 4)
    p1_dtensor = DTensor.from_local(param1.detach(), mesh, [Replicate()])
    p1_dtensor.grad = DTensor.from_local(param1.grad, mesh, [Replicate()])

    param2 = torch.randn(2, 2, requires_grad=True)
    param2.grad = torch.randn(2, 2)
    p2_dtensor = DTensor.from_local(param2.detach(), mesh, [Replicate()])
    p2_dtensor.grad = DTensor.from_local(param2.grad, mesh, [Replicate()])

    class MockOptimizer:
        def __init__(self):
            self.param_groups = [
                {
                    "params": [p1_dtensor, p2_dtensor],
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "amsgrad": False,
                    "maximize": False,
                    "decoupled_weight_decay": True,
                }
            ]
            self.state = {p1_dtensor: {}, p2_dtensor: {}}
            self.virtual_allocator = MagicMock()
            self.virtual_allocator.init_exp.return_value = (
                torch.zeros(2, 2),
                torch.zeros(2, 2),
            )
            self.print_swap_flag = False

    opt = MockOptimizer()

    with patch("torch._fused_adamw_") as mock_fused:
        virtual_optimizer_step_impl(opt)
        assert "exp_avg" in opt.state[p1_dtensor]
        assert isinstance(opt.state[p1_dtensor]["exp_avg"], DTensor)
        assert "exp_avg" in opt.state[p2_dtensor]
        assert isinstance(opt.state[p2_dtensor]["exp_avg"], DTensor)
        first_call_count = mock_fused.call_count

        p1_dtensor.grad = DTensor.from_local(torch.randn(4, 4), mesh, [Replicate()])
        p2_dtensor.grad = DTensor.from_local(torch.randn(2, 2), mesh, [Replicate()])
        virtual_optimizer_step_impl(opt)

        assert "exp_avg_sq" in opt.state[p1_dtensor]
        assert mock_fused.call_count > first_call_count
