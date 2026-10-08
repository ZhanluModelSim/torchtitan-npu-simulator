# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, _api, _dispatch, _redistribute
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta as SpecTensorMeta
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from torchtitan_npu.simulator import meta_env
from torchtitan_npu.simulator.capture.comm_events import capture_fake_collectives
from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture


@pytest.fixture
def redistribution_patch(monkeypatch):
    monkeypatch.setattr(meta_env, "_original_redistribute_local_tensor", meta_env._MISSING)
    monkeypatch.setattr(_redistribute, "redistribute_local_tensor", _redistribute.redistribute_local_tensor)
    for module in (_api, _dispatch):
        monkeypatch.setattr(module, "redistribute_local_tensor", module.redistribute_local_tensor)
    monkeypatch.setattr(meta_env, "_simulator_redistribute_wrapper", meta_env._MISSING)
    meta_env._patch_redistribute_local_tensor_for_meta()
    yield _redistribute.redistribute_local_tensor
    meta_env.unpatch_device_type_to_meta()


def _mesh(tmp_path, mesh_shape, coordinate, device_type="cpu"):
    rank = 0
    for size, index in zip(mesh_shape, coordinate):
        rank = rank * size + index
    dist.init_process_group(
        "fake", init_method=f"file://{tmp_path / 'store'}", rank=rank, world_size=math.prod(mesh_shape)
    )
    return DeviceMesh(device_type, torch.arange(math.prod(mesh_shape)).reshape(mesh_shape))


def _spec(mesh, shape, placements):
    strides = torch.empty(shape, device="meta").stride()
    return DTensorSpec(mesh, tuple(placements), tensor_meta=SpecTensorMeta(torch.Size(shape), strides, torch.float32))


@pytest.mark.parametrize(
    "length,mesh_shape,coordinate,expected",
    [
        (8, (2, 2), (0, 0), 2),
        (8, (2, 2), (1, 1), 2),
        (5, (2,), (0,), 3),
        (5, (2,), (1,), 2),
        (1, (4,), (0,), 1),
        (1, (4,), (1,), 0),
        (1, (4,), (2,), 0),
        (1, (4,), (3,), 0),
        (0, (4,), (0,), 0),
    ],
)
def test_target_shape_uses_rank_and_all_sharding_dimensions(
    redistribution_patch, tmp_path, length, mesh_shape, coordinate, expected
):
    mesh = _mesh(tmp_path, mesh_shape, coordinate)
    try:
        source = _spec(mesh, (length,), [Replicate() for _ in mesh_shape])
        target = _spec(mesh, (length,), [Shard(0) for _ in mesh_shape])
        result = redistribution_patch(torch.empty(length, device="meta"), source, target)
        assert result.shape == (expected,)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("mesh_device", ["cpu", "meta"])
@pytest.mark.parametrize(
    "source,target,expected_comm",
    [
        (Replicate(), Replicate(), []),
        (Replicate(), Shard(0), []),
        (Shard(0), Replicate(), ["allgather"]),
        (Partial(), Replicate(), ["allreduce"]),
        (Partial(), Shard(0), ["reduce_scatter"]),
        (Shard(0), Shard(1), ["allgather"]),
    ],
)
def test_real_placement_planner_preserves_dependencies(
    redistribution_patch, tmp_path, monkeypatch, mesh_device, source, target, expected_comm
):
    monkeypatch.setattr(torch, "meta", meta_env._MetaDeviceModule(), raising=False)
    mesh = _mesh(tmp_path, (2,), (0,), mesh_device)
    if mesh_device == "meta" and source == Shard(0) and target == Shard(1):
        expected_comm = ["all_to_all"]
    try:
        source_spec = _spec(mesh, (4, 4), [source])
        target_spec = _spec(mesh, (4, 4), [target])
        local_shape, _ = compute_local_shape_and_global_offset((4, 4), mesh, [source], skip_offset=True)
        x = torch.ones(local_shape, device="meta")
        capture = OpDispatchCapture()
        with capture, capture_fake_collectives() as recorder:
            produced = x.sin()
            result = redistribution_patch(produced, source_spec, target_spec)
            result.cos()
        assert [event.comm_primitive for event in recorder.events] == expected_comm
        nodes = list(capture.build_nodes().values())
        producer = next(node for node in nodes if node.annotations["raw_op_type"] == "aten.sin.default")
        consumer = next(node for node in nodes if node.annotations["raw_op_type"] == "aten.cos.default")

        def ancestors(node):
            by_id = {item.op_id: item for item in nodes}
            seen = set()
            pending = list(node.predecessors)
            while pending:
                ident = pending.pop()
                if ident in seen:
                    continue
                seen.add(ident)
                if ident in by_id:
                    pending.extend(by_id[ident].predecessors)
            return seen

        assert producer.op_id in ancestors(consumer)
        for event in recorder.events:
            assert event.op_id in ancestors(consumer)
            assert event.world_size == 2
        assert result.shape == compute_local_shape_and_global_offset((4, 4), mesh, [target], skip_offset=True)[0]
        if source == target:
            assert result is produced
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "gradient_placement,expected", [(Replicate(), ["allgather"]), (Partial(), ["allgather", "reduce_scatter"])]
)
def test_dtensor_cached_api_reference_uses_planner_in_forward_and_backward(
    redistribution_patch, tmp_path, gradient_placement, expected
):
    mesh = _mesh(tmp_path, (2,), (0,))
    try:
        x = torch.ones(2, 4, device="meta", requires_grad=True)
        phase = ["forward"]
        capture = OpDispatchCapture(phase_provider=lambda: phase[0])
        with capture, capture_fake_collectives() as recorder:
            distributed = DTensor.from_local(x, mesh, [Shard(0)], run_check=False, shape=(4, 4), stride=(4, 1))
            full = distributed.redistribute(placements=[Replicate()])
            local = full.to_local(grad_placements=[gradient_placement])
            loss = local.sin().sum()
            phase[0] = "backward"
            loss.backward()
        assert [event.comm_primitive for event in recorder.events] == expected
        assert x.grad.shape == x.shape
        assert all(event.op_id in capture.build_nodes() for event in recorder.events)
    finally:
        dist.destroy_process_group()


def test_patch_restores_cached_and_late_imported_references(redistribution_patch):
    import importlib

    original = meta_env._original_redistribute_local_tensor
    late_module = importlib.import_module("torch.distributed.tensor.experimental._tp_transform")
    assert late_module.redistribute_local_tensor is redistribution_patch
    assert _api.redistribute_local_tensor is redistribution_patch
    assert _dispatch.redistribute_local_tensor is redistribution_patch
    meta_env.unpatch_device_type_to_meta()
    assert all(module.redistribute_local_tensor is original for module in (_redistribute, _api, _dispatch, late_module))


def test_non_meta_local_slice_preserves_values(redistribution_patch, tmp_path):
    mesh = _mesh(tmp_path, (2,), (0,))
    try:
        source = _spec(mesh, (4,), [Replicate()])
        target = _spec(mesh, (4,), [Shard(0)])
        tensor = torch.arange(4, dtype=torch.float32)
        expected = meta_env._original_redistribute_local_tensor(tensor, source, target)
        result = redistribution_patch(tensor, source, target)
        assert torch.equal(result, expected)
        assert result.stride() == expected.stride()
        assert (result.untyped_storage()._cdata == tensor.untyped_storage()._cdata) == (
            expected.untyped_storage()._cdata == tensor.untyped_storage()._cdata
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("source,target", [(Shard(0), Replicate()), (Partial(), Shard(0)), (Shard(0), Shard(1))])
def test_uneven_planner_transport_and_unpadding(redistribution_patch, tmp_path, monkeypatch, rank, source, target):
    monkeypatch.setattr(torch, "meta", meta_env._MetaDeviceModule(), raising=False)
    mesh = _mesh(tmp_path, (2,), (rank,), "meta")
    try:
        shape = (5, 3)
        local_shape = compute_local_shape_and_global_offset(shape, mesh, [source], skip_offset=True)[0]
        tensor = torch.ones(local_shape, device="meta")
        capture = OpDispatchCapture()
        with capture, capture_fake_collectives() as recorder:
            result = redistribution_patch(tensor, _spec(mesh, shape, [source]), _spec(mesh, shape, [target]))
            result.sin()
        assert result.shape == compute_local_shape_and_global_offset(shape, mesh, [target], skip_offset=True)[0]
        assert len(recorder.events) == 1
        receive = next(event for event in capture.memory_events() if event.op_id == recorder.events[0].op_id)
        assert receive.outputs[0].shape != ()
        assert receive.inputs[0].dtype == receive.outputs[0].dtype == "float32"
    finally:
        dist.destroy_process_group()


def test_meta_axis_swap_records_backward_transport(redistribution_patch, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "meta", meta_env._MetaDeviceModule(), raising=False)
    mesh = _mesh(tmp_path, (2,), (0,), "meta")
    try:
        x = torch.ones(2, 4, device="meta", requires_grad=True)
        phase = ["forward"]
        capture = OpDispatchCapture(phase_provider=lambda: phase[0])
        with capture, capture_fake_collectives() as recorder:
            value = DTensor.from_local(x, mesh, [Shard(0)], run_check=False, shape=(4, 4), stride=(4, 1))
            output = value.redistribute(placements=[Shard(1)]).to_local()
            loss = output.sin().sum()
            phase[0] = "backward"
            loss.backward()
        assert [event.comm_primitive for event in recorder.events] == ["all_to_all", "all_to_all"]
        assert [
            next(node for node in capture._events if node.op_id == event.op_id).phase for event in recorder.events
        ] == ["forward", "backward"]
        assert x.grad.shape == x.shape
    finally:
        dist.destroy_process_group()
