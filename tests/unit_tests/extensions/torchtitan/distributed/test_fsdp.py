# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Check the placements passed to FSDP without running device collectives."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.tensor import Shard
from torchtitan.models.common.activation import Sigmoid
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)


@pytest.fixture(scope="module")
def fsdp():
    # This definition-only module does not require package activation. Load it
    # directly to keep unrelated NPU runtime patches outside this CPU policy test.
    path = Path(__file__).resolve().parents[5] / "torchtitan_npu/extensions/torchtitan/distributed/fsdp.py"
    spec = importlib.util.spec_from_file_location("npu_fsdp_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StorageMeshStub(dict):
    """Expose only the axis names and sizes consumed by the EP=1 policy."""

    def __init__(self, dp_shard, cp=1):
        super().__init__(dp_shard=SimpleNamespace(size=lambda: dp_shard))
        if cp > 1:
            self["cp"] = SimpleNamespace(size=lambda: cp)
        self.mesh_dim_names = tuple(self)


def _moe(num_experts, *, enable_ep):
    return make_moe_config(
        num_experts=num_experts,
        router=make_router_config(dim=8, num_experts=num_experts, gate_param_init={}, score_func=Sigmoid.Config()),
        routed_experts=make_routed_experts_config(
            dim=8, hidden_dim=16, num_experts=num_experts, top_k=1, param_init={}, enable_ep=enable_ep
        ),
        shared_experts=make_ffn_config(dim=8, hidden_dim=16, w13_param_init={}, w2_param_init={}, enable_sp=False),
        enable_sp=False,
    ).build()


def _decoder_shell(block):
    model = nn.Module()
    model.enable_weight_tying = False
    model.tok_embeddings = model.norm = model.lm_head = None
    model.layers = nn.ModuleDict({"0": block})
    return model


@pytest.fixture
def shard_calls(monkeypatch, fsdp):
    calls = []

    def record_shard(module, **kwargs):
        placement_fn = kwargs.get("shard_placement_fn", lambda parameter: None)
        placements = {name: placement_fn(parameter) for name, parameter in module.named_parameters()}
        calls.append((module, placements, kwargs))

    monkeypatch.setattr(fsdp, "fully_shard", record_shard)
    return calls


@pytest.mark.parametrize(
    "dp_shard,cp,expected_w13,expected_w2",
    [(2, 1, 0, 0), (4, 1, 0, 0), (2, 4, 2, 1)],
    ids=["experts-fit", "experts-equal-shards", "cp-exceeds-experts"],
)
def test_ep1_expert_storage_overrides_stacked_projection_placement(
    fsdp, shard_calls, dp_shard, cp, expected_w13, expected_w2
):
    with torch.device("meta"):
        block = nn.Module()
        block.feed_forward = _moe(4, enable_ep=False)
        model = _decoder_shell(block)
        model.host_table = nn.Parameter(torch.empty(4, 8))
    mesh = _StorageMeshStub(dp_shard, cp)
    ignored = {model.host_table}

    fsdp.apply_fsdp_to_decoder(model, mesh, torch.bfloat16, torch.float32, pp_enabled=False, ignored_params=ignored)

    assert shard_calls[0][0] is block
    # W13 is [E, 2, F, D]; W2 is [E, D, F]. Expert-axis sharding
    # takes precedence until dp_shard * cp exceeds E.
    assert shard_calls[0][1] == {
        "feed_forward.router.gate.weight": None,
        "feed_forward.routed_experts.w13.weight": Shard(expected_w13),
        "feed_forward.routed_experts.w2.weight": Shard(expected_w2),
        "feed_forward.shared_experts.w13.weight": Shard(1),
        "feed_forward.shared_experts.w2.weight": None,
    }
    assert shard_calls[-1][0] is model
    assert all(kwargs["ignored_params"] is ignored for _, _, kwargs in shard_calls)
    assert all(kwargs["mesh"] is mesh for _, _, kwargs in shard_calls)


def test_dense_stacked_gate_up_shards_matrix_rows(fsdp, shard_calls):
    with torch.device("meta"):
        block = nn.Module()
        block.feed_forward = make_ffn_config(
            dim=8, hidden_dim=16, w13_param_init={}, w2_param_init={}, enable_sp=False
        ).build()
        model = _decoder_shell(block)

    fsdp.apply_fsdp_to_decoder(model, _StorageMeshStub(4), torch.bfloat16, torch.float32, pp_enabled=False)

    assert shard_calls[0][0] is block
    assert shard_calls[0][1] == {
        "feed_forward.w13.weight": Shard(1),
        "feed_forward.w2.weight": None,
    }


def test_native_dtype_override_reaches_fsdp_policy_with_ignored_parameters(fsdp, shard_calls):
    with torch.device("meta"):
        block = nn.Module()
        block.feed_forward = make_ffn_config(
            dim=8, hidden_dim=16, w13_param_init={}, w2_param_init={}, enable_sp=False
        ).build()
        model = _decoder_shell(block)
        model.host_table = nn.Parameter(torch.empty(4, 8))
    fp32_weight = block.feed_forward.w13.weight
    default_weight = block.feed_forward.w2.weight
    ignored = {model.host_table}

    def keep_selected_dtype(parameter):
        return parameter.dtype if parameter is fp32_weight else None

    fsdp.apply_fsdp_to_decoder(
        model,
        _StorageMeshStub(4),
        torch.bfloat16,
        torch.float32,
        pp_enabled=False,
        ignored_params=ignored,
        param_dtype_override_fn=keep_selected_dtype,
    )

    assert shard_calls
    for _, _, kwargs in shard_calls:
        policy = kwargs["mp_policy"]
        assert kwargs["ignored_params"] is ignored
        assert policy.param_dtype_override_fn is keep_selected_dtype
        assert policy._resolve_for_param(fp32_weight).param_dtype == torch.float32
        assert policy._resolve_for_param(default_weight).param_dtype == torch.bfloat16
        assert policy.reduce_dtype == torch.float32


@pytest.fixture
def ep2_storage_meshes():
    import torch.distributed as dist
    from torch.distributed.device_mesh import DeviceMesh
    from torch.testing._internal.distributed.fake_pg import FakeStore

    # Real mesh selection with no communication; fully_shard itself is recorded
    # by shard_calls, so this does not claim FSDP collective coverage.
    assert not dist.is_initialized()
    dist.init_process_group("fake", store=FakeStore(), rank=1, world_size=4)
    try:
        dense = DeviceMesh("cpu", torch.arange(4), mesh_dim_names=("dp_shard",))
        sparse = DeviceMesh("cpu", torch.arange(4).reshape(2, 2), mesh_dim_names=("edp_shard", "ep"))
        yield dense, sparse
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("num_experts,expected_axes", [(4, (0, 0)), (2, (2, 1))], ids=["experts-fit", "row-sharded"])
def test_ep2_selects_sparse_fsdp_mesh_and_preserves_dense_stacked_placement(
    fsdp, shard_calls, ep2_storage_meshes, num_experts, expected_axes
):
    from torch.distributed.fsdp import DataParallelMeshDims

    with torch.device("meta"):
        block = nn.Module()
        block.feed_forward = _moe(num_experts, enable_ep=True)
        model = _decoder_shell(block)
        model.host_table = nn.Parameter(torch.empty(4, 8))
    dense_mesh, sparse_mesh = ep2_storage_meshes
    ignored = {model.host_table}

    fsdp.apply_fsdp_to_decoder(
        model,
        dense_mesh,
        torch.bfloat16,
        torch.float32,
        pp_enabled=False,
        ep_degree=2,
        edp_mesh=sparse_mesh,
        dp_mesh_dims=DataParallelMeshDims(shard="dp_shard"),
        edp_mesh_dims=DataParallelMeshDims(shard="edp_shard"),
        ignored_params=ignored,
    )

    placements = shard_calls[0][1]
    for projection, axis in zip(("w13", "w2"), expected_axes, strict=True):
        placement = placements[f"feed_forward.routed_experts.{projection}.weight"]
        assert placement.placement == Shard(axis)
        assert placement.mesh_info.spmd_mesh is sparse_mesh
        assert placement.mesh_info.mesh.mesh.tolist() == [1, 3]
    for projection, axis in (("w13", 1), ("w2", 0)):
        placement = placements[f"feed_forward.shared_experts.{projection}.weight"]
        assert placement.placement == Shard(axis)
        assert placement.mesh_info.spmd_mesh is dense_mesh
        assert placement.mesh_info.mesh.mesh.tolist() == [0, 1, 2, 3]
    assert all(kwargs["ignored_params"] is ignored for _, _, kwargs in shard_calls)
