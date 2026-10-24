# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
from torchtitan_simulator.capture.fsdp_residency import (
    _uses_fsdp_sharding,
)


def test_fsdp_residency_distinguishes_sharded_and_replicated_param_groups():
    fsdp_mesh_info = object.__new__(FSDPMeshInfo)

    assert _uses_fsdp_sharding(SimpleNamespace(mesh_info=fsdp_mesh_info))
    assert not _uses_fsdp_sharding(SimpleNamespace(mesh_info=object()))


def test_root_cleanup_observes_staging_created_and_cleared_inside_post_forward(monkeypatch):
    from inspect import getattr_static

    from torch.distributed.fsdp._fully_shard._fsdp_param_group import AllGatherState, FSDPCommContext, FSDPParamGroup
    from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
    from torchtitan_simulator.capture import fsdp_residency
    from torchtitan_simulator.capture.comm_events import observe_collectives
    from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
    from torchtitan_simulator.compat.patches import PatchScope

    full = torch.empty(16, device="meta")
    stage = torch.empty_like(full)
    comm_ctx = object.__new__(FSDPCommContext)
    comm_ctx.all_gather_state = None
    group = SimpleNamespace(
        fsdp_params=[SimpleNamespace(all_gather_outputs=[full])],
        _module_fqn="layer",
        _training_state="forward",
        is_unsharded=False,
        _sim_active_transition_id="oracle",
        _all_gather_process_group=SimpleNamespace(size=lambda: 2),
        _all_gather_result=SimpleNamespace(all_gather_output=stage),
        comm_ctx=comm_ctx,
    )

    def wait(self):
        self.is_unsharded = True
        self.comm_ctx.all_gather_state = AllGatherState(self._all_gather_result, None)
        self._all_gather_result = None

    def root_post_forward(self):
        FSDPParamGroup.wait_for_unshard(group)
        self._comm_ctx.all_gather_state = None

    monkeypatch.setattr(FSDPParamGroup, "wait_for_unshard", wait)
    monkeypatch.setattr(FSDPState, "_post_forward", root_post_forward)
    monkeypatch.setattr(fsdp_residency, "_uses_fsdp_sharding", lambda group: True)
    previous_setattr = getattr_static(FSDPCommContext, "__setattr__")
    with PatchScope() as scope, observe_collectives() as recorder, OpDispatchCapture():
        fsdp_residency.install_fsdp_residency_hooks(scope)
        FSDPState._post_forward(SimpleNamespace(_comm_ctx=comm_ctx))
    markers = recorder.fsdp_residency_events
    assert [(marker.action, marker.num_bytes) for marker in markers] == [("alloc", 64), ("staging_free", 64)]
    assert markers[0].seq_idx < markers[1].seq_idx
    assert markers[1].transition_id == "oracle"
    assert getattr_static(FSDPCommContext, "__setattr__") is previous_setattr
