# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol

from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives


@pytest.fixture(scope="module", autouse=True)
def _fake_process_group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = "29713"
    dist.init_process_group("fake", rank=0, world_size=8)
    yield
    dist.destroy_process_group()


def test_all_reduce_on_meta_tensor_is_noop_and_recorded():
    t = torch.randn(16, 16, device="meta")
    with capture_fake_collectives() as recorder:
        result = dist.all_reduce(t)
    assert result is None
    assert len(recorder.events) == 1
    assert recorder.events[0].comm_primitive == "allreduce"
    assert recorder.events[0].tensor_shape == (16, 16)


def test_all_gather_into_tensor_on_meta_is_noop_and_recorded():
    input_t = torch.randn(4, device="meta")
    output_t = torch.empty(32, device="meta")
    with capture_fake_collectives() as recorder:
        dist.all_gather_into_tensor(output_t, input_t)
    assert output_t.shape == (32,)  # caller-preallocated shape untouched
    assert recorder.events[0].comm_primitive == "allgather"


def test_all_to_all_single_on_meta_is_noop_and_recorded():
    input_t = torch.randn(8, device="meta")
    output_t = torch.empty(8, device="meta")
    with capture_fake_collectives() as recorder:
        dist.all_to_all_single(output_t, input_t)
    assert recorder.events[0].comm_primitive == "all_to_all"


def test_funcol_all_gather_tensor_returns_correctly_shaped_new_tensor():
    t = torch.randn(4, 8, device="meta")
    with capture_fake_collectives() as recorder:
        out = funcol.all_gather_tensor(t, gather_dim=0, group=dist.group.WORLD)
    assert out.shape == (32, 8)  # 4 * world_size(8)
    assert recorder.events[0].comm_primitive == "allgather"


def test_funcol_all_to_all_single_respects_output_split_sizes():
    t = torch.randn(10, device="meta")
    with capture_fake_collectives() as recorder:
        out = funcol.all_to_all_single(t, [3, 4], [5, 5], group=dist.group.WORLD)
    assert out.shape == (7,)
    assert recorder.events[0].comm_primitive == "all_to_all"


def test_collectives_restored_after_context_exit():
    original_all_reduce = dist.all_reduce
    with capture_fake_collectives():
        assert dist.all_reduce is not original_all_reduce
    assert dist.all_reduce is original_all_reduce
