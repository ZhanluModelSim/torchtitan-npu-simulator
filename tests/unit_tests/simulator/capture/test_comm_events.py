# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torch.distributed.pipelining import schedules

from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives
from torchtitan_npu.simulator.meta_env import (
    _mark_p2p_ops,
    _pp_context,
    _local_tensor_for_shape,
    _split_meta_dtensor,
    patch_device_type_to_meta,
    unpatch_device_type_to_meta,
)


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


def test_meta_dtensor_split_preserves_backward_connectivity():
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate

    mesh = DeviceMesh.from_group(dist.group.WORLD, "meta", mesh=list(range(8)))
    local = torch.randn(2, 4, device="meta", requires_grad=True)
    tensor = DTensor.from_local(local, mesh, [Replicate()], run_check=False)

    left, right = _split_meta_dtensor(tensor, [2, 2], dim=-1)
    (left.sum() + right.sum()).backward()

    assert left.shape == (2, 2)
    assert right.shape == (2, 2)
    assert local.grad is not None
    assert local.grad.shape == local.shape


def test_dtensor_shape_checks_use_local_shape():
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    mesh = DeviceMesh.from_group(dist.group.WORLD, "meta", mesh=list(range(8)))
    local = torch.empty(2, 2, device="meta")
    tensor = DTensor.from_local(
        local,
        mesh,
        [Shard(1)],
        shape=torch.Size((2, 16)),
        stride=(16, 1),
        run_check=False,
    )

    assert tensor.shape == (2, 16)
    assert _local_tensor_for_shape(tensor).shape == (2, 2)


def test_hsdp_ep_mesh_info_uses_replicate_then_shard_axes():
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.fsdp._fully_shard._fsdp_common import HSDPMeshInfo
    import torchtitan.models.llama4.parallelize as llama4_parallelize

    mesh = DeviceMesh(
        "meta",
        torch.arange(8).reshape(2, 4),
        mesh_dim_names=("dp_replicate", "fsdp"),
    )
    try:
        patch_device_type_to_meta()
        mesh_info = llama4_parallelize.FSDPMeshInfo(mesh, shard_mesh_dim=0)
        assert isinstance(mesh_info, HSDPMeshInfo)
        assert mesh_info.replicate_mesh_dim == 0
        assert mesh_info.shard_mesh_dim == 1
    finally:
        unpatch_device_type_to_meta()


def test_collectives_restored_after_context_exit():
    original_all_reduce = dist.all_reduce
    with capture_fake_collectives():
        assert dist.all_reduce is not original_all_reduce
    assert dist.all_reduce is original_all_reduce


def test_disabled_memory_tracking_drops_fsdp_residency_events():
    with capture_fake_collectives(memory_tracking_enabled=False) as recorder:
        recorder.record_fsdp_residency(
            group_id="group",
            action="alloc",
            phase="forward",
            num_bytes=1024,
            tensor_ids=(1,),
            shard_world_size=1,
        )

    assert recorder.fsdp_residency_events == []
    assert len(recorder.fsdp_schedule_events) == 1
    assert recorder.fsdp_schedule_events[0].group_id == "group"
    assert recorder.fsdp_schedule_events[0].shard_world_size == 1


def test_mixed_p2p_batch_uses_each_op_pp_context():
    """Deferred 1F1B P2POps do not inherit the stale warmup context."""
    previous_context = dict(_pp_context)
    patch_device_type_to_meta()
    send_tensor = torch.empty(4, device="meta")
    recv_tensor = torch.empty(4, device="meta")
    try:
        # The first warmup recv executes before forward_one_chunk can stamp a
        # stage. Both P2POps must instead use their creation-time metadata.
        _pp_context.update(stage=-1, mb_idx=-1, phase="forward", comp_type="F")
        with capture_fake_collectives() as recorder:
            send_op = dist.P2POp(dist.isend, send_tensor, 1, dist.group.WORLD)
            recv_op = dist.P2POp(dist.irecv, recv_tensor, 1, dist.group.WORLD)
            _mark_p2p_ops(
                [send_op], stage=0, mb_idx=0, phase="forward", direction="send"
            )
            _mark_p2p_ops(
                [recv_op], stage=1, mb_idx=2, phase="backward", direction="recv"
            )
            schedules._batch_p2p([send_op, recv_op], desc="fwd_send_bwd_recv")

        assert [(event.p2p_stage, event.p2p_mb_idx, event.p2p_direction) for event in recorder.events] == [
            (0, 0, "forward_send"),
            (1, 2, "backward_recv"),
        ]
        assert [event.transfer_id for event in recorder.events] == [
            "pp:forward:s0->s1:mb0:t0",
            "pp:backward:s2->s1:mb2:t0",
        ]
    finally:
        _pp_context.clear()
        _pp_context.update(previous_context)
        unpatch_device_type_to_meta()
