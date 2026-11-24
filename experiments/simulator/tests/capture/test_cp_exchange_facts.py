# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchtitan_simulator.capture.comm_events import observe_collectives
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.compat.cp_exchange import observe_deepseek_v4_cp_exchange
from torchtitan_simulator.compat.patches import PatchScope
from torchtitan_simulator.metadata import to_meta_metadata


def test_native_cp_list_exchange_preserves_forward_and_reverse_full_group_geometry(tmp_path):
    from torchtitan_npu.models.deepseek_v4 import token_dispatcher as cp

    # Three ranks own positions 0–3, 4–7 and 8–11. Each list below contains
    # the foreign rows requested by one receiver in its actual plan order.
    required = [[4, 5, 9], [0, 8], [1, 2, 4]]
    sends = [[0, 1, 2], [2, 0, 1], [1, 1, 0]]
    receives = [[0, 2, 1], [1, 0, 1], [2, 1, 0]]
    original = cp.CPTokenDispatcher._all_to_all
    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=3)
    try:
        dispatcher = cp.CPTokenDispatcher(cp.CPTokenDispatcher.Config())
        dispatcher.wire_meshes(cp_mesh=DeviceMesh.from_group(dist.group.WORLD, "cpu"))
        with PatchScope() as scope:
            observe_deepseek_v4_cp_exchange(scope)
            routing = cp._routing_geometry(required, shard_len=4, cp_size=3)
            plan = to_meta_metadata(cp._build_exchange_plan(routing[0], "cpu"))
            capture = OpDispatchCapture()
            x = torch.empty(3, 4, device="meta", requires_grad=True)
            with observe_collectives() as recorder, capture:
                result = dispatcher._all_to_all(x, *plan.splits_for_collective())
                result.sum().backward()
                # Equal local geometry does not establish the other ranks' exchanges.
                dispatcher._all_to_all(x.detach(), list(plan.send_splits), list(plan.recv_splits))
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
    finally:
        dist.destroy_process_group()
