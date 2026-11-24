# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from importlib import metadata

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchtitan_simulator.capture.comm_events import observe_collectives
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.compat.cp_exchange import observe_deepseek_v4_cp_exchange
from torchtitan_simulator.compat.patches import PatchScope
from torchtitan_simulator.metadata import to_meta_metadata


@pytest.fixture
def dispatcher(tmp_path):
    from torchtitan_npu.models.deepseek_v4 import token_dispatcher as cp

    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=3)
    try:
        dispatcher = cp.CPTokenDispatcher(cp.CPTokenDispatcher.Config())
        dispatcher.wire_meshes(cp_mesh=DeviceMesh.from_group(dist.group.WORLD, "cpu"))
        yield dispatcher
    finally:
        dist.destroy_process_group()


def test_native_cp_list_exchange_preserves_forward_and_reverse_full_group_geometry(dispatcher):
    from torchtitan_npu.models.deepseek_v4 import token_dispatcher as cp

    # Three ranks own positions 0–3, 4–7 and 8–11. Each list below contains
    # the foreign rows requested by one receiver in its actual plan order.
    required = [[4, 5, 9], [0, 8], [1, 2, 4]]
    sends = [[0, 1, 2], [2, 0, 1], [1, 1, 0]]
    receives = [[0, 2, 1], [1, 0, 1], [2, 1, 0]]
    original = cp.CPTokenDispatcher._all_to_all
    with PatchScope() as scope:
        host_splits = observe_deepseek_v4_cp_exchange(scope)
        routing = cp._routing_geometry(required, shard_len=4, cp_size=3)
        native = cp._build_exchange_plan(routing[0], "cpu")
        copies = [list(native.send_splits), list(native.recv_splits)]
        # Traverse nested aliases before either the dataclass or familiar field names.
        lifted = to_meta_metadata(
            {"aliases": [(native.send_splits, {"receive": native.recv_splits})], "copies": copies, "plan": native},
            host_splits=host_splits,
        )
        plan = lifted["plan"]
        first_send, first_receive = lifted["aliases"][0]
        assert first_send is plan.send_splits is native.send_splits
        assert first_receive["receive"] is plan.recv_splits is native.recv_splits
        assert lifted["copies"][0] is not first_send and lifted["copies"][1] is not plan.recv_splits
        assert plan.send_indices.device.type == "meta"
        capture = OpDispatchCapture()
        x = torch.empty(3, 4, device="meta", requires_grad=True)
        with observe_collectives() as recorder, capture:
            result = dispatcher._all_to_all(x, *plan.splits_for_collective())
            result.sum().backward()
            dispatcher._all_to_all(x.detach(), *lifted["copies"])
        nodes = capture.build_nodes()
        forward, backward, unproven = [nodes[event.op_id] for event in recorder.events]
        assert all(n.annotations["raw_op_type"] == "c10d.alltoall_.default" for n in (forward, backward, unproven))
        for node, send, recv in ((forward, sends, receives), (backward, receives, sends)):
            facts = node.attrs["collective_facts"]
            assert facts["capture_rank"] == 0
            assert list(facts["ordered_group"]) == [0, 1, 2]
            assert [list(row) for row in facts["send_rows"]] == send
            assert [list(row) for row in facts["recv_rows"]] == recv
            assert [list(shape) for shape in facts["rank_input_shapes"]] == [[sum(row), 4] for row in send]
            assert [list(shape) for shape in facts["rank_output_shapes"]] == [[sum(row), 4] for row in recv]
            assert facts["buffer_relation"] == "distinct"
        assert "collective_facts" not in unproven.attrs
    assert cp.CPTokenDispatcher._all_to_all is original


def test_unknown_cp_backward_warns_once_with_wheel_and_node_without_guessing(dispatcher, monkeypatch, caplog):
    from spmd_types import _collectives

    from torchtitan_npu.models.deepseek_v4 import token_dispatcher as cp

    # Change only the external node's identity. Its actual forward and backward
    # still dispatch the native list collective with opposite splits.
    class RenamedUneven(_collectives._AllToAllUneven):
        pass

    monkeypatch.setattr(_collectives, "_AllToAllUneven", RenamedUneven)
    observer = "torchtitan_simulator.compat.cp_exchange"
    with PatchScope() as scope:
        host_splits = observe_deepseek_v4_cp_exchange(scope)
        rows = cp._routing_geometry([[4, 5, 9], [0, 8], [1, 2, 4]], shard_len=4, cp_size=3)
        plan = to_meta_metadata(cp._build_exchange_plan(rows[0], "cpu"), host_splits=host_splits)
        splits = plan.splits_for_collective()
        x = torch.empty(3, 4, device="meta", requires_grad=True)
        with torch.no_grad():
            dispatcher._all_to_all(x, *splits)
        dispatcher._all_to_all(x.detach(), *splits)
        dispatcher._all_to_all(x, list(splits[0]), list(splits[1]))
        assert not any(record.name == observer for record in caplog.records)
        capture = OpDispatchCapture()
        with observe_collectives() as recorder, capture:
            for _ in range(2):
                dispatcher._all_to_all(x, *splits).sum().backward()
        nodes = capture.build_nodes()
        events = [nodes[event.op_id] for event in recorder.events]
        assert len(events) == 4
        for forward, backward in zip(events[::2], events[1::2], strict=True):
            assert forward.attrs["collective_facts"]["source"] == "deepseek_v4_cp_routing_geometry_v1"
            assert backward.annotations["raw_op_type"] == "c10d.alltoall_.default"
            assert "collective_facts" not in backward.attrs
        (record,) = [record for record in caplog.records if record.name == observer]
        assert record.levelno == logging.WARNING
        warning = record.getMessage()
        assert "RenamedUnevenBackward" in warning
        for package in ("torchtitan-npu", "spmd-types", "torch"):
            assert f"{package}={metadata.version(package)}" in warning
