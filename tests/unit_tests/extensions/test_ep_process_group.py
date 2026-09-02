# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def ep_process_group():
    """Load the extension without importing the NPU package bootstrap."""
    spmd_types = types.ModuleType("torchtitan.distributed.spmd_types")
    registered = {}

    def set_spmd_meshes(**kwargs):
        registered.update(kwargs)

    spmd_types.set_spmd_meshes = set_spmd_meshes

    token_dispatcher = types.ModuleType("torchtitan.models.common.token_dispatcher")

    class BaseEPTokenDispatcher:
        def wire_meshes(self, *, ep_mesh, **kwargs):
            self.ep_mesh = ep_mesh
            self.kwargs = kwargs

    class DeepEPTokenDispatcher(BaseEPTokenDispatcher):
        pass

    token_dispatcher.BaseEPTokenDispatcher = BaseEPTokenDispatcher
    token_dispatcher.DeepEPTokenDispatcher = DeepEPTokenDispatcher

    parents = {
        "torchtitan": types.ModuleType("torchtitan"),
        "torchtitan.distributed": types.ModuleType("torchtitan.distributed"),
        "torchtitan.models": types.ModuleType("torchtitan.models"),
        "torchtitan.models.common": types.ModuleType("torchtitan.models.common"),
    }
    modules = {
        **parents,
        "torchtitan.distributed.spmd_types": spmd_types,
        "torchtitan.models.common.token_dispatcher": token_dispatcher,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    path = Path(__file__).resolve().parents[3] / "torchtitan_npu" / "extensions" / "ep_process_group.py"
    spec = importlib.util.spec_from_file_location("ep_process_group_test_impl", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._registered = registered
    yield module
    sys.modules.pop(spec.name, None)
    for name, previous_module in previous.items():
        if previous_module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous_module


class _FakeMesh:
    def __init__(self, group, size=2):
        self._group = group
        self._size = size

    def size(self):
        return self._size

    def get_group(self):
        return self._group


class _FakeSparseMesh:
    def __init__(self, mesh_dim_names, group_names):
        self.mesh_dim_names = tuple(mesh_dim_names)
        self._dim_group_names = list(group_names)

    def __getitem__(self, mesh_dim_name):
        index = self.mesh_dim_names.index(mesh_dim_name)
        return _FakeMesh(SimpleNamespace(group_name=self._dim_group_names[index]))

    def get_group(self, mesh_dim=None):
        if mesh_dim is None:
            if len(self._dim_group_names) != 1:
                raise RuntimeError("mesh dimension is required")
            index = 0
        else:
            index = self.mesh_dim_names.index(mesh_dim)
        return SimpleNamespace(group_name=self._dim_group_names[index])

    def size(self):
        return 2


@pytest.fixture(autouse=True)
def _reset_registry(ep_process_group):
    meshes = ep_process_group._EP_MESHES.copy()
    sparse_meshes = ep_process_group._SPARSE_MESHES.copy()
    ep_process_group._EP_MESHES.clear()
    ep_process_group._SPARSE_MESHES.clear()
    ep_process_group._DEEPEP_GROUPS.clear()
    yield
    ep_process_group._EP_MESHES.clear()
    ep_process_group._SPARSE_MESHES.clear()
    ep_process_group._DEEPEP_GROUPS.clear()
    ep_process_group._EP_MESHES.update(meshes)
    ep_process_group._SPARSE_MESHES.update(sparse_meshes)


def test_ep1_and_fake_groups_are_not_rebuilt(ep_process_group, monkeypatch):
    ep1_group = SimpleNamespace(group_name="ep1")
    ep1_mesh = _FakeMesh(ep1_group, size=1)
    assert ep_process_group._separate_ep_mesh(ep1_mesh) is ep1_mesh

    fake_group = SimpleNamespace(group_name="fake")
    fake_mesh = _FakeMesh(fake_group)
    monkeypatch.setattr(ep_process_group, "_is_fake_group", lambda group: True)
    assert ep_process_group._separate_ep_mesh(fake_mesh) is fake_mesh


def test_ep_mesh_is_created_once_and_group_is_resolved(ep_process_group, monkeypatch):
    source_group = SimpleNamespace(group_name="source")
    dedicated_group = SimpleNamespace(group_name="dedicated")
    source_mesh = _FakeMesh(source_group)
    dedicated_mesh = _FakeMesh(dedicated_group)
    calls = []
    warmed = []

    monkeypatch.setattr(ep_process_group, "_is_fake_group", lambda group: False)

    def warm(mesh):
        warmed.append(mesh)

    monkeypatch.setattr(ep_process_group, "_warm_spmd_mesh_axis", warm)

    def create(mesh):
        calls.append(mesh)
        return dedicated_mesh

    monkeypatch.setattr(ep_process_group, "_create_separate_ep_mesh", create)

    assert ep_process_group._separate_ep_mesh(source_mesh) is dedicated_mesh
    assert ep_process_group._separate_ep_mesh(source_mesh) is dedicated_mesh
    assert calls == [source_mesh]
    assert warmed == [dedicated_mesh]
    assert ep_process_group._EP_MESHES["source"] is dedicated_mesh


def test_deepep_group_is_resolved_for_an_ep_mesh(ep_process_group, monkeypatch):
    source_group = SimpleNamespace(group_name="source", size=lambda: 2)
    source_mesh = _FakeMesh(source_group)
    dedicated_group = SimpleNamespace(group_name="deepep")
    calls = []

    def create(group, **kwargs):
        calls.append((group, kwargs))
        return dedicated_group

    monkeypatch.setattr(ep_process_group, "_create_deepep_group", create)

    first = ep_process_group.get_deepep_group(
        source_mesh,
        hidden_dim=4096,
        num_max_tokens_per_rank=4096,
        num_experts=16,
        top_k=2,
    )
    assert first is dedicated_group
    assert calls == [
        (
            source_group,
            {
                "hidden_dim": 4096,
                "num_max_tokens_per_rank": 4096,
                "num_experts": 16,
                "top_k": 2,
            },
        ),
    ]


def test_deepep_group_requires_multiple_ep_ranks(ep_process_group):
    source_mesh = _FakeMesh(SimpleNamespace(group_name="ep1"), size=1)
    with pytest.raises(ValueError, match="more than one rank"):
        ep_process_group.get_deepep_group(
            source_mesh,
            hidden_dim=256,
            num_max_tokens_per_rank=128,
            num_experts=4,
            top_k=1,
        )


def test_deepep_group_creation_is_cached_and_checks_capacity(ep_process_group, monkeypatch):
    class _Options:
        def __init__(self):
            self.hccl_config = None

    class _ProcessGroupHCCL:
        Options = _Options

    fake_torch_npu = types.SimpleNamespace(
        _C=types.SimpleNamespace(_distributed_c10d=types.SimpleNamespace(ProcessGroupHCCL=_ProcessGroupHCCL))
    )

    class _ElasticBuffer:
        @staticmethod
        def get_moe_ep_ccl_buffer_size(world_size, capacity, hidden_dim, num_experts, top_k):
            del world_size, hidden_dim, num_experts, top_k
            return capacity

    fake_cann_ops_transformer = types.SimpleNamespace(ElasticBuffer=_ElasticBuffer)
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    monkeypatch.setitem(sys.modules, "cann_ops_transformer", fake_cann_ops_transformer)
    monkeypatch.setattr(ep_process_group.dist, "get_process_group_ranks", lambda group: [0, 1])
    created = []

    def new_group(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(group_name="deepep")

    monkeypatch.setattr(ep_process_group.dist, "new_group", new_group)
    source_group = SimpleNamespace(group_name="source", size=lambda: 2)

    first = ep_process_group._create_deepep_group(
        source_group,
        hidden_dim=256,
        num_max_tokens_per_rank=1024,
        num_experts=16,
        top_k=2,
    )
    second = ep_process_group._create_deepep_group(
        source_group,
        hidden_dim=256,
        num_max_tokens_per_rank=512,
        num_experts=16,
        top_k=2,
    )

    assert first is second
    assert len(created) == 1
    assert created[0]["pg_options"].hccl_config == {"hccl_buffer_size": 1024}
    with pytest.raises(RuntimeError, match="buffer is too small"):
        ep_process_group._create_deepep_group(
            source_group,
            hidden_dim=256,
            num_max_tokens_per_rank=2048,
            num_experts=16,
            top_k=2,
        )


def test_dedicated_mesh_is_not_recursively_recreated(ep_process_group, monkeypatch):
    source_group = SimpleNamespace(group_name="source")
    dedicated_group = SimpleNamespace(group_name="dedicated")
    source_mesh = _FakeMesh(source_group)
    dedicated_mesh = _FakeMesh(dedicated_group)
    ep_process_group._EP_MESHES["source"] = dedicated_mesh
    monkeypatch.setattr(
        ep_process_group,
        "_create_separate_ep_mesh",
        lambda mesh: (_ for _ in ()).throw(AssertionError("recreated")),
    )

    assert ep_process_group._separate_ep_mesh(dedicated_mesh) is dedicated_mesh


def test_dispatcher_wiring_receives_the_separated_mesh(ep_process_group, monkeypatch):
    source_group = SimpleNamespace(group_name="source")
    source_mesh = _FakeMesh(source_group)
    dedicated_mesh = _FakeMesh(SimpleNamespace(group_name="dedicated"))
    monkeypatch.setattr(ep_process_group, "_is_fake_group", lambda group: False)
    monkeypatch.setattr(
        ep_process_group,
        "_create_separate_ep_mesh",
        lambda mesh: dedicated_mesh,
    )

    token_dispatcher = sys.modules["torchtitan.models.common.token_dispatcher"]
    dispatcher = token_dispatcher.BaseEPTokenDispatcher()
    dispatcher.wire_meshes(ep_mesh=source_mesh, tp_mesh="tp")
    assert dispatcher.ep_mesh is dedicated_mesh
    assert dispatcher.kwargs == {"tp_mesh": "tp"}


def test_dispatcher_with_owned_group_skips_generic_mesh_creation(ep_process_group, monkeypatch):
    source_group = SimpleNamespace(group_name="source")
    source_mesh = _FakeMesh(source_group)
    dispatcher = sys.modules["torchtitan.models.common.token_dispatcher"].BaseEPTokenDispatcher()
    dispatcher._uses_custom_ep_process_group = True
    calls = []

    def separate(mesh):
        calls.append(mesh)
        return mesh

    monkeypatch.setattr(ep_process_group, "_separate_ep_mesh", separate)

    dispatcher.wire_meshes(ep_mesh=source_mesh)

    assert dispatcher.ep_mesh is source_mesh
    assert calls == []


def test_deepep_inherited_wiring_uses_the_separated_mesh(ep_process_group, monkeypatch):
    source_mesh = _FakeMesh(SimpleNamespace(group_name="source"))
    dedicated_mesh = _FakeMesh(SimpleNamespace(group_name="dedicated"))
    monkeypatch.setattr(ep_process_group, "_is_fake_group", lambda group: False)
    monkeypatch.setattr(ep_process_group, "_create_separate_ep_mesh", lambda mesh: dedicated_mesh)

    token_dispatcher = sys.modules["torchtitan.models.common.token_dispatcher"]
    dispatcher = token_dispatcher.DeepEPTokenDispatcher()
    dispatcher.wire_meshes(ep_mesh=source_mesh)

    assert dispatcher.ep_mesh is dedicated_mesh


def test_sparse_registration_replaces_only_ep_group(ep_process_group, monkeypatch):
    source_mesh = _FakeSparseMesh(
        ("dp_replicate", "efsdp", "ep"),
        ("dp-source", "efsdp-source", "ep-source"),
    )
    dedicated_mesh = _FakeSparseMesh(("ep",), ("ep-dedicated",))
    monkeypatch.setattr(ep_process_group, "_is_fake_group", lambda group: False)
    monkeypatch.setattr(
        ep_process_group,
        "_create_separate_ep_mesh",
        lambda mesh: dedicated_mesh,
    )

    spmd_types = sys.modules["torchtitan.distributed.spmd_types"]
    spmd_types.set_spmd_meshes(dense_mesh="dense", sparse_mesh=source_mesh)
    replaced = ep_process_group._registered["sparse_mesh"]

    assert replaced is not source_mesh
    assert replaced.get_group("dp_replicate").group_name == "dp-source"
    assert replaced.get_group("efsdp").group_name == "efsdp-source"
    assert replaced.get_group("ep").group_name == "ep-dedicated"
    assert ep_process_group._registered["dense_mesh"] == "dense"


def test_sparse_registration_reuses_cached_mesh(ep_process_group, monkeypatch):
    source_mesh = _FakeSparseMesh(
        ("dp_replicate", "efsdp", "ep"),
        ("dp-source", "efsdp-source", "ep-source"),
    )
    dedicated_mesh = _FakeSparseMesh(("ep",), ("ep-dedicated",))
    monkeypatch.setattr(ep_process_group, "_is_fake_group", lambda group: False)
    monkeypatch.setattr(
        ep_process_group,
        "_create_separate_ep_mesh",
        lambda mesh: dedicated_mesh,
    )

    first = ep_process_group._replace_sparse_ep_mesh(source_mesh)
    second = ep_process_group._replace_sparse_ep_mesh(source_mesh)

    assert first is second
