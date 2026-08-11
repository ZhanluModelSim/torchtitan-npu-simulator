# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import importlib
import os
import types
from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate
from torchtitan.components.optimizer import OptimizersContainer

from torchtitan_npu.config.configs import OptimizerConfig
from torchtitan_npu.patches.optimizer import swap_optimizer
from torchtitan_npu.patches.optimizer.optimizer_selector import (
    NpuOptimizerDispatcher,
    patch_npu_optimizer_framework,
)


@pytest.fixture(autouse=True)
def _reset_swap_global_state():
    swap_optimizer.SwapOptimizersContainer.param_to_cpu_states_map.clear()
    swap_optimizer.SwapOptimizersContainer.param_to_device_states_map.clear()
    swap_optimizer.SwapOptimizersContainer.swap_to_host_events_map.clear()
    swap_optimizer.SwapOptimizersContainer.swap_to_device_events_map.clear()
    swap_optimizer.SwapOptimizersContainer.param_update_events_map.clear()
    yield
    swap_optimizer.SwapOptimizersContainer.param_to_cpu_states_map.clear()
    swap_optimizer.SwapOptimizersContainer.param_to_device_states_map.clear()
    swap_optimizer.SwapOptimizersContainer.swap_to_host_events_map.clear()
    swap_optimizer.SwapOptimizersContainer.swap_to_device_events_map.clear()
    swap_optimizer.SwapOptimizersContainer.param_update_events_map.clear()


def _make_cpu_mesh():
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    return DeviceMesh("cpu", [0])


def test_unwrap_dtensor_returns_plain_tensor_for_non_dtensor():
    tensor = torch.randn(2, 2)

    result = swap_optimizer.unwrap_dtensor(tensor)

    assert result is tensor


def test_swap_config_delegates_to_base_container_when_disabled():
    """swap_optimizer=False → Config.build() returns a plain OptimizersContainer."""
    model = nn.Linear(2, 2)
    config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=False, name="AdamW", lr=1e-3, implementation="for-loop"
    )

    result = config.build(model_parts=[model])

    assert isinstance(result, OptimizersContainer)
    assert not isinstance(result, swap_optimizer.SwapOptimizersContainer)


def test_swap_config_rejects_unknown_optimizer():
    """Unknown optimizer name raises NotImplementedError before any NPU access."""
    config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=True,
        name="SGD",
        lr=1e-3,
        implementation="for-loop",
        swap_optimizer_times=8,
    )

    with pytest.raises(NotImplementedError, match="Optimizer SGD not added"):
        config.build(model_parts=[nn.Linear(2, 2)])


def test_swap_config_routes_to_swap_container_when_enabled(monkeypatch):
    """swap_optimizer=True → Config.build() instantiates the _owner class.

    Mocks _owner so the test verifies dispatch routing without running the
    real SwapOptimizersContainer.__init__ (which touches NPU streams).
    """
    instantiated = []

    class FakeOwner:
        def __init__(self, *, config, model_parts):
            instantiated.append((config, model_parts))

    monkeypatch.setattr(swap_optimizer.SwapOptimizersContainer.Config, "_owner", FakeOwner)

    config = swap_optimizer.SwapOptimizersContainer.Config(
        swap_optimizer=True,
        name="AdamW",
        lr=1e-3,
        implementation="fused",
        swap_optimizer_times=16,
    )

    result = config.build(model_parts=["model_part"])

    assert isinstance(result, FakeOwner)
    assert len(instantiated) == 1
    cfg, parts = instantiated[0]
    assert cfg.swap_optimizer is True
    assert cfg.swap_optimizer_times == 16
    assert cfg.name == "AdamW"
    assert parts == ["model_part"]


def test_importing_swap_optimizer_preserves_optimizer_dispatcher():
    patch_npu_optimizer_framework()
    try:
        importlib.reload(swap_optimizer)

        assert OptimizerConfig.build is NpuOptimizerDispatcher.dispatch_build
        with pytest.raises(ValueError, match="Cannot enable both"):
            OptimizerConfig(swap_optimizer=True, virtual_optimizer=True).build(model_parts=[])
    finally:
        patch_npu_optimizer_framework()


class _TiedEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 2)
        self.lm_head = torch.nn.Linear(2, 4, bias=False)
        self.lm_head.weight = self.embedding.weight


def _build_swapped_optimizer(
    monkeypatch,
    *,
    exp_avg,
    exp_avg_sq,
    step=None,
    model=None,
    param=None,
):
    if model is None:
        model = torch.nn.Linear(2, 2, bias=False)
    if param is None:
        param = model.weight
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    if step is not None:
        optimizer.param_groups[0]["step"] = torch.tensor(float(step))
    optimizer.param_to_group_map = {param: optimizer.param_groups[0]}

    state = optimizer.state[param]
    live_exp_avg = torch.zeros_like(param)
    live_exp_avg_sq = torch.zeros_like(param)
    live_exp_avg.untyped_storage().resize_(0)
    live_exp_avg_sq.untyped_storage().resize_(0)
    state["exp_avg"] = live_exp_avg
    state["exp_avg_sq"] = live_exp_avg_sq
    state["max_exp_avg_sq"] = None

    cpu_state = {
        "exp_avg": torch.full_like(param, exp_avg, device="cpu"),
        "exp_avg_sq": torch.full_like(param, exp_avg_sq, device="cpu"),
        "max_exp_avg_sq": None,
    }
    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "param_to_cpu_states_map",
        {param: cpu_state},
    )
    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "param_to_device_states_map",
        {param: state},
    )
    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "swap_to_host_stream",
        None,
    )

    container = swap_optimizer.SwapOptimizersContainer.__new__(swap_optimizer.SwapOptimizersContainer)
    container.model_parts = [model]
    container.optimizers = [optimizer]

    return types.SimpleNamespace(
        container=container,
        model=model,
        optimizer=optimizer,
        param=param,
        cpu_state=cpu_state,
        live_exp_avg=live_exp_avg,
        live_exp_avg_sq=live_exp_avg_sq,
    )


def test_state_dict_uses_cpu_cache_snapshot_and_restores_swap_state(monkeypatch):
    fixture = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=1.25,
        exp_avg_sq=2.5,
        step=5,
    )

    state_dict = fixture.container.state_dict()
    fixture.optimizer.param_groups[0]["step"].add_(1)

    assert torch.equal(
        state_dict["state.weight.exp_avg"],
        fixture.cpu_state["exp_avg"],
    )
    assert torch.equal(
        state_dict["state.weight.exp_avg_sq"],
        fixture.cpu_state["exp_avg_sq"],
    )
    assert state_dict["state.weight.exp_avg"].device.type == "cpu"
    assert not isinstance(state_dict["state.weight.exp_avg"], DTensor)
    assert "state.weight.max_exp_avg_sq" not in state_dict
    assert state_dict["state.weight.step"].item() == 5
    assert state_dict["param_groups.weight.step"].item() == 5
    assert state_dict["state.weight.step"] is not fixture.optimizer.param_groups[0]["step"]
    assert state_dict["param_groups.weight.step"] is not fixture.optimizer.param_groups[0]["step"]

    state = fixture.optimizer.state[fixture.param]
    assert state["exp_avg"] is fixture.live_exp_avg
    assert state["exp_avg"].untyped_storage().size() == 0
    assert state["exp_avg_sq"] is fixture.live_exp_avg_sq
    assert state["exp_avg_sq"].untyped_storage().size() == 0
    assert state["max_exp_avg_sq"] is None
    assert "step" not in state


def test_state_dict_synchronizes_pending_host_swap_before_reading_cpu_cache(
    monkeypatch,
):
    fixture = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=1.25,
        exp_avg_sq=2.5,
        step=5,
    )

    class FakeHostStream:
        def __init__(self):
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

    class FakeCurrentStream:
        def wait_stream(self, _stream):
            raise AssertionError("CPU checkpoint read must synchronize the host stream")

    host_stream = FakeHostStream()
    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "swap_to_host_stream",
        host_stream,
    )
    monkeypatch.setattr(
        swap_optimizer,
        "get_torch_device",
        lambda: types.SimpleNamespace(current_stream=lambda: FakeCurrentStream()),
    )

    fixture.container.state_dict()

    assert host_stream.synchronized


def test_state_dict_preserves_dtensor_layout_for_cpu_cache(monkeypatch):
    mesh = _make_cpu_mesh()
    local_param = torch.randn(2, 2)
    param = DTensor.from_local(local_param, mesh, [Replicate()])
    live_exp_avg = DTensor.from_local(torch.zeros_like(local_param), mesh, [Replicate()])
    cpu_exp_avg = torch.ones_like(local_param, device="cpu")
    state = {"exp_avg": live_exp_avg}

    monkeypatch.setattr(
        swap_optimizer.SwapOptimizersContainer,
        "param_to_cpu_states_map",
        {param: {"exp_avg": cpu_exp_avg}},
    )
    monkeypatch.setattr(
        swap_optimizer.DTensor,
        "from_local",
        staticmethod(
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("CPU-cache state_dict must not call DTensor.from_local")
            )
        ),
    )

    value = swap_optimizer.SwapOptimizersContainer._state_value_for_state_dict(
        param,
        state,
        "exp_avg",
    )

    assert isinstance(value, DTensor)
    assert value.device_mesh == param.device_mesh
    assert value.placements == param.placements
    local_value = value.to_local()
    assert local_value.device.type == "cpu"
    assert local_value.untyped_storage().data_ptr() == cpu_exp_avg.untyped_storage().data_ptr()


def test_state_dict_rejects_nonempty_cache_placeholder_without_storage():
    placeholder = torch.empty(4, 16384, device="cpu")
    placeholder.untyped_storage().resize_(0)

    try:
        swap_optimizer.SwapOptimizersContainer._tensor_for_state_dict(placeholder)
        raise AssertionError("Expected zero-storage nonempty tensor to be rejected")
    except RuntimeError as exc:
        assert "without CPU cache" in str(exc)


def test_load_state_dict_rebuilds_swap_state_without_mutating_checkpoint(monkeypatch):
    source = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=3.0,
        exp_avg_sq=4.0,
        step=5,
    )
    checkpoint_state_dict = source.container.state_dict()
    original_exp_avg = checkpoint_state_dict["state.weight.exp_avg"].clone()
    original_exp_avg_sq = checkpoint_state_dict["state.weight.exp_avg_sq"].clone()
    original_exp_avg_storage = checkpoint_state_dict["state.weight.exp_avg"].untyped_storage().size()
    original_exp_avg_sq_storage = checkpoint_state_dict["state.weight.exp_avg_sq"].untyped_storage().size()

    target = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=0.0,
        exp_avg_sq=0.0,
    )

    target.container.load_state_dict(checkpoint_state_dict)

    assert torch.equal(checkpoint_state_dict["state.weight.exp_avg"], original_exp_avg)
    assert torch.equal(checkpoint_state_dict["state.weight.exp_avg_sq"], original_exp_avg_sq)
    assert checkpoint_state_dict["state.weight.exp_avg"].untyped_storage().size() == original_exp_avg_storage
    assert checkpoint_state_dict["state.weight.exp_avg_sq"].untyped_storage().size() == original_exp_avg_sq_storage

    assert target.optimizer.param_groups[0]["step"].item() == 5
    target_cpu_state = swap_optimizer.SwapOptimizersContainer.param_to_cpu_states_map[target.param]
    assert torch.equal(target_cpu_state["exp_avg"], original_exp_avg)
    assert torch.equal(target_cpu_state["exp_avg_sq"], original_exp_avg_sq)
    assert target.optimizer.state[target.param]["exp_avg"].untyped_storage().size() == 0
    assert target.optimizer.state[target.param]["exp_avg_sq"].untyped_storage().size() == 0
    assert target.optimizer.state[target.param]["max_exp_avg_sq"] is None


def test_shared_parameter_aliases_use_canonical_save_and_alias_load(monkeypatch):
    source_model = _TiedEmbeddingModel()
    source = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=3.0,
        exp_avg_sq=4.0,
        step=5,
        model=source_model,
        param=source_model.embedding.weight,
    )

    fqns_by_param = swap_optimizer.SwapOptimizersContainer._fqns_by_param(source_model)
    assert fqns_by_param[source_model.embedding.weight] == (
        "embedding.weight",
        "lm_head.weight",
    )

    checkpoint_state_dict = source.container.state_dict()
    assert "state.embedding.weight.exp_avg" in checkpoint_state_dict
    assert "state.lm_head.weight.exp_avg" not in checkpoint_state_dict
    assert "param_groups.embedding.weight.lr" in checkpoint_state_dict
    assert "param_groups.lm_head.weight.lr" not in checkpoint_state_dict

    alias_state_dict = {}
    for key, value in checkpoint_state_dict.items():
        if key.startswith("state.embedding.weight."):
            key = key.replace("state.embedding.weight.", "state.lm_head.weight.", 1)
        elif key.startswith("param_groups.embedding.weight."):
            key = key.replace("param_groups.embedding.weight.", "param_groups.lm_head.weight.", 1)
        alias_state_dict[key] = value

    target_model = _TiedEmbeddingModel()
    target = _build_swapped_optimizer(
        monkeypatch,
        exp_avg=0.0,
        exp_avg_sq=0.0,
        model=target_model,
        param=target_model.embedding.weight,
    )
    target.optimizer.param_groups[0]["lr"] = 0.5

    target.container.load_state_dict(alias_state_dict)

    assert target.optimizer.param_groups[0]["lr"] == 1e-3
    assert target.optimizer.param_groups[0]["step"].item() == 5
    target_cpu_state = swap_optimizer.SwapOptimizersContainer.param_to_cpu_states_map[target.param]
    assert torch.equal(
        target_cpu_state["exp_avg"],
        alias_state_dict["state.lm_head.weight.exp_avg"],
    )
    assert torch.equal(
        target_cpu_state["exp_avg_sq"],
        alias_state_dict["state.lm_head.weight.exp_avg_sq"],
    )


def test_fqns_by_param_falls_back_when_private_get_fqns_is_unavailable(monkeypatch):
    model = _TiedEmbeddingModel()
    monkeypatch.setattr(swap_optimizer, "_get_fqns", None)

    fqns_by_param = swap_optimizer.SwapOptimizersContainer._fqns_by_param(model)

    assert fqns_by_param[model.embedding.weight] == (
        "embedding.weight",
        "lm_head.weight",
    )


def test_loaded_plain_cpu_state_rebuilds_dtensor_runtime_placeholder():
    mesh = _make_cpu_mesh()
    local_param = torch.randn(2, 2)
    param = DTensor.from_local(local_param, mesh, [Replicate()])
    cpu_exp_avg = torch.ones_like(local_param, device="cpu")

    placeholder = swap_optimizer.SwapOptimizersContainer._clone_loaded_state_for_device_placeholder(
        param,
        cpu_exp_avg,
    )

    assert isinstance(placeholder, DTensor)
    assert placeholder.device_mesh == param.device_mesh
    assert placeholder.placements == param.placements
    assert placeholder.to_local().untyped_storage().size() == 0


class _SwapScheduleEvent:
    def __init__(self, calls, index):
        self.calls = calls
        self.index = index

    def synchronize(self):
        self.calls.append(("release", self.index))


def _configure_adamw_swap_schedule(monkeypatch):
    params = [object() for _ in range(4)]
    buckets = [params[:2], params[2:3], params[3:]]
    calls = []
    indices = {id(param): index for index, param in enumerate(params)}

    def wait_event(event):
        return None

    def current_stream():
        return types.SimpleNamespace(record_event=object, wait_event=wait_event)

    def stream_context(stream):
        return nullcontext()

    fake_device = types.SimpleNamespace(current_stream=current_stream, stream=stream_context)

    def get_device():
        return fake_device

    monkeypatch.setattr(swap_optimizer, "get_torch_device", get_device)
    container = swap_optimizer.SwapOptimizersContainer

    def wait_swap(param):
        calls.append(("wait", indices[id(param)]))

    def swap_to_host(param):
        index = indices[id(param)]
        calls.append(("d2h", index))
        return _SwapScheduleEvent(calls, index)

    def update(param, state, group):
        calls.append(("update", indices[id(param)]))

    def load_bucket(bucket, *, wait_for_host):
        calls.append(("load", [indices[id(param)] for param in bucket], wait_for_host))

    monkeypatch.setattr(container, "swap_to_host_stream", object())
    monkeypatch.setattr(container, "wait_swap_to_device_event", wait_swap)
    monkeypatch.setattr(container, "swap_states_to_host", swap_to_host)
    monkeypatch.setattr(swap_optimizer, "param_update", update)
    monkeypatch.setattr(swap_optimizer, "_load_swap_bucket", load_bucket)
    group = {}
    optimizer = types.SimpleNamespace(
        state={param: {} for param in params},
        param_to_group_map={param: group for param in params},
        _swap_release_event=_SwapScheduleEvent(calls, "previous"),
    )
    return optimizer, buckets, calls


def test_swap_adamw_bounds_prefetch_to_two_buckets(monkeypatch):
    optimizer, buckets, calls = _configure_adamw_swap_schedule(monkeypatch)

    step_buckets = getattr(swap_optimizer, "_step_swap_buckets")
    step_buckets(optimizer, buckets)

    assert calls == [
        ("release", "previous"),
        ("load", [0, 1], True),
        ("wait", 0),
        ("update", 0),
        ("wait", 1),
        ("update", 1),
        ("load", [2], False),
        ("d2h", 0),
        ("d2h", 1),
        ("wait", 2),
        ("update", 2),
        ("release", 1),
        ("load", [3], False),
        ("d2h", 2),
        ("wait", 3),
        ("update", 3),
        ("release", 2),
        ("d2h", 3),
    ]
    assert getattr(optimizer, "_swap_release_event").index == 3


def test_swap_hybrid_waits_for_previous_adamw_release_before_muon_step():
    calls = []
    release_event = types.SimpleNamespace(synchronize=lambda: calls.append("release"))
    muon = types.SimpleNamespace(step=lambda *args, **kwargs: calls.append("muon"))
    adamw = types.SimpleNamespace(
        _swap_release_event=release_event,
        step=lambda *args, **kwargs: calls.append("adamw"),
    )
    container = swap_optimizer.SwapMuonHybridOptimizersContainer.__new__(
        swap_optimizer.SwapMuonHybridOptimizersContainer
    )
    container.optimizers = [muon, adamw]

    container.step()

    assert calls == ["release", "muon", "adamw"]
    assert getattr(adamw, "_swap_release_event") is None


def test_swap_adamw_d2h_records_source_stream_before_releasing_storage(
    monkeypatch,
):
    calls = []
    param = object()
    stream = types.SimpleNamespace(record_event=lambda: object())

    storage = MagicMock()
    storage.size.return_value = 1
    storage.resize_.side_effect = lambda size: calls.append(("resize", size))

    local_state = MagicMock()
    local_state.untyped_storage.return_value = storage
    local_state.record_stream.side_effect = lambda value: calls.append(("record_stream", value))
    cpu_state = MagicMock()
    cpu_state.copy_.side_effect = lambda source, non_blocking: calls.append(("copy", source, non_blocking))

    container = swap_optimizer.SwapOptimizersContainer
    container.param_to_device_states_map[param] = {"exp_avg": local_state}
    container.param_to_cpu_states_map[param] = {"exp_avg": cpu_state}
    monkeypatch.setattr(
        swap_optimizer,
        "get_torch_device",
        lambda: types.SimpleNamespace(current_stream=lambda: stream),
    )

    swap_optimizer.SwapOptimizersContainer.swap_states_to_host(param)

    assert calls == [
        ("copy", local_state, True),
        ("record_stream", stream),
        ("resize", 0),
    ]
