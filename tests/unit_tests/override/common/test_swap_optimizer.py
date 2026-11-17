# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

"""CPU contracts for the explicit optimizer-state swap override policy."""

import importlib
import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, OverrideConfig, apply_overrides

from torchtitan_npu.extensions.novaswap import swap_api as product_swap_api
from torchtitan_npu.override.common import optimizer as product_swap
from torchtitan_npu.override.common.optimizer import OptimizerStateSwapContainer

pytestmark = pytest.mark.cpu


def _supports_checkpointable_tensor_protocol() -> bool:
    try:
        protocol = getattr(
            importlib.import_module("torch.distributed.checkpoint.protocol"),
            "CheckpointableTensor",
        )
    except (ImportError, AttributeError):
        return False

    probe = torch.empty(0)
    vars(probe).update(
        global_shape=(0,),
        global_offsets=((0,),),
        local_offsets=((0,),),
        local_sizes=((0,),),
    )
    try:
        return isinstance(probe, protocol)
    except TypeError:
        return False


def _patch_container_global(monkeypatch, name: str, value) -> None:
    """Patch the module globals used by the imported container class.

    Other CPU tests can temporarily replace ``torchtitan_npu`` modules. Patch
    the class under test directly so fakes remain effective even when a package
    attribute points at a different module instance.
    """
    monkeypatch.setitem(
        OptimizerStateSwapContainer._swap_adamw.__globals__,
        name,
        value,
    )


def _patch_container_swap_api(monkeypatch, swap_api) -> None:
    if not hasattr(swap_api, "wait_for_device_release"):
        swap_api.wait_for_device_release = lambda _name: None
    monkeypatch.setattr(product_swap, "swap_api", swap_api)
    _patch_container_global(monkeypatch, "swap_api", swap_api)
    monkeypatch.setitem(product_swap._NovaSwapAdamW._submit.__globals__, "swap_api", swap_api)


def _install_adamw_swap(optimizer: torch.optim.AdamW) -> None:
    getattr(OptimizerStateSwapContainer, "_swap_adamw")(optimizer)


def _patch_fake_swap_runtime(
    monkeypatch,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[SimpleNamespace]]]:
    registered: dict[str, torch.Tensor] = {}
    handles: dict[str, tuple[SimpleNamespace]] = {}

    def register_tensor(tensor, name) -> None:
        registered[name] = tensor

    def execute(name, action) -> None:
        if action == "D2H":
            handles[name] = (
                SimpleNamespace(
                    swap_event=None,
                    is_completed=False,
                    tensor_cpu=registered[name].detach().view(torch.uint8).clone(),
                ),
            )
        elif action == "H2D":
            registered[name].view(torch.uint8).copy_(handles[name][0].tensor_cpu)

    def get_d2h_cpu_buffer(name):
        return handles[name][0].tensor_cpu

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=register_tensor,
            execute=execute,
            get_d2h_cpu_buffer=get_d2h_cpu_buffer,
        ),
    )
    return registered, handles


def test_swap_api_get_d2h_cpu_buffer_waits_for_live_handle(monkeypatch) -> None:
    tensor_cpu = torch.arange(8, dtype=torch.uint8)
    handle = SimpleNamespace(transfer="D2H", tensor_cpu=tensor_cpu)
    waited = []
    api = product_swap_api
    engine = api.SwapEngine
    wait_globals = getattr(getattr(engine.get_d2h_cpu_buffer, "__func__"), "__globals__")
    monkeypatch.setitem(engine._handles, "optimizer_state", (handle,))
    monkeypatch.setitem(wait_globals, "_wait", waited.append)

    result = api.get_d2h_cpu_buffer("optimizer_state")

    assert result is tensor_cpu
    assert waited == [handle]


def _build_checkpoint_model(seed: int):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.GELU(),
            torch.nn.Linear(3, 2),
        )
    named_parameters = list(model.named_parameters())
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for _, parameter in named_parameters],
                "param_names": [name for name, _ in named_parameters],
            }
        ],
        lr=0.03,
        foreach=False,
    )
    _install_adamw_swap(optimizer)
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]
    return model, optimizer, container


def _build_single_parameter_container(parameter: torch.nn.Parameter, lr: float):
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "param_names": ["weight"]}],
        lr=lr,
        foreach=False,
    )
    _install_adamw_swap(optimizer)
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]
    return optimizer, container


def _named_optimizer_parameters(optimizer):
    for group in optimizer.param_groups:
        yield from zip(group["params"], group["param_names"], strict=True)


def _clone_flat_tensor_state(optimizer) -> dict[str, torch.Tensor]:
    result = {}
    for parameter, fqn in _named_optimizer_parameters(optimizer):
        for state_name, value in optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor):
                result[f"state.{fqn}.{state_name}"] = value.detach().clone()
    return result


def _swapped_tensor_identities(optimizer) -> dict[tuple[torch.Tensor, str], torch.Tensor]:
    result = {}
    state_names = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    for parameter, state in optimizer.state.items():
        for state_name, value in state.items():
            if state_name in state_names:
                result[(parameter, state_name)] = value
    return result


def _assert_checkpoint_views(optimizer, flat_state, handles) -> None:
    parameter_fqns = dict(_named_optimizer_parameters(optimizer))
    locations = getattr(optimizer, "_torchtitan_npu_checkpoint_locations")
    for (parameter, state_name), (tensor_name, byte_offset) in locations.items():
        view = flat_state[f"state.{parameter_fqns[parameter]}.{state_name}"]
        raw = handles[tensor_name][0].tensor_cpu
        assert view.data_ptr() == raw.data_ptr() + byte_offset
        assert view.global_shape == tuple(view.shape)
        assert view.global_offsets == (tuple(0 for _ in view.shape),)
        assert view.local_offsets == (tuple(0 for _ in view.shape),)
        assert view.local_sizes == (tuple(view.shape),)


def _set_model_gradients(model, value: float) -> None:
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, value)


def _assert_named_tensors(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    for name, expected_tensor in expected.items():
        torch.testing.assert_close(actual[name], expected_tensor, rtol=0, atol=0)


class _Root(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )


def test_optimizer_state_swap_override_replaces_optimizer_config() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.optimizer.swap_optimizer",
            ]
        ),
        config,
    )

    assert len(replacements) == 1
    assert isinstance(config.optimizer, OptimizerStateSwapContainer.Config)


def test_virtual_optimizer_override_replaces_optimizer_config() -> None:
    config = _Root.Config()

    replacements = apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.common.optimizer.virtual",
            ]
        ),
        config,
    )

    assert len(replacements) == 1
    assert type(config.optimizer).__module__ == "torchtitan_npu.override.common.optimizer"
    assert type(config.optimizer).__qualname__ == "VirtualOptimizersContainer.Config"


def test_virtual_optimizers_container_registers_state_init_hook(monkeypatch) -> None:
    hooks = []
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    monkeypatch.setattr(optimizer, "register_step_pre_hook", hooks.append)

    def initialize_container(self, *, config, model_parts):
        self.optimizers = [optimizer]

    monkeypatch.setattr(
        product_swap.OptimizersContainer,
        "__init__",
        initialize_container,
    )

    product_swap.VirtualOptimizersContainer(config=object(), model_parts=[])

    assert hooks == [product_swap._swap_state_init_hook]


def test_optimizer_state_swap_conflicts_with_virtual_optimizer_override() -> None:
    config = _Root.Config()

    with pytest.raises(ValueError, match="both claim node 'optimizer'"):
        apply_overrides(
            OverrideConfig(
                imports=[
                    "torchtitan_npu.override.common.optimizer.virtual",
                    "torchtitan_npu.override.common.optimizer.swap_optimizer",
                ]
            ),
            config,
        )


def test_optimizer_state_swap_async_dcp_round_trip(monkeypatch, tmp_path) -> None:
    registered, handles = _patch_fake_swap_runtime(monkeypatch)
    source_model, source_optimizer, source_container = _build_checkpoint_model(seed=7)
    inputs = torch.arange(8, dtype=torch.float32).reshape(2, 4).div(7)
    source_model(inputs).square().sum().backward()
    source_optimizer.step()
    source_optimizer.param_groups[0]["lr"] = 0.017

    expected_parameters = {name: parameter.detach().clone() for name, parameter in source_model.named_parameters()}
    expected_state = _clone_flat_tensor_state(source_optimizer)
    source_flat_state = source_container.state_dict()
    _assert_checkpoint_views(source_optimizer, source_flat_state, handles)
    original_cpu_buffers = {name: handle[0].tensor_cpu.clone() for name, handle in handles.items()}

    checkpoint_dir = tmp_path / "dcp"
    future = dcp.async_save(
        {"model": source_model, "optimizer": source_container},
        checkpoint_id=checkpoint_dir,
        no_dist=True,
    )
    for handle in handles.values():
        handle[0].tensor_cpu.fill_(0xFF)
    future.result()

    for name, raw in original_cpu_buffers.items():
        handles[name][0].tensor_cpu.copy_(raw)
    _set_model_gradients(source_model, 0.125)
    source_optimizer.step()
    expected_next_parameters = {name: parameter.detach().clone() for name, parameter in source_model.named_parameters()}

    del source_flat_state
    registered.clear()
    handles.clear()
    del source_container, source_optimizer, source_model

    target_model, target_optimizer, target_container = _build_checkpoint_model(seed=19)
    target_container.state_dict()
    target_state_tensors = _swapped_tensor_identities(target_optimizer)
    dcp.load(
        {"model": target_model, "optimizer": target_container},
        checkpoint_id=checkpoint_dir,
        no_dist=True,
    )

    assert target_optimizer.param_groups[0]["lr"] == 0.017
    for key, tensor in target_state_tensors.items():
        assert target_optimizer.state[key[0]][key[1]] is tensor

    target_parameters = dict(target_model.named_parameters())
    _assert_named_tensors(target_parameters, expected_parameters)
    target_state = target_container.state_dict()
    _assert_named_tensors(target_state, expected_state)
    _set_model_gradients(target_model, 0.125)
    target_optimizer.step()
    _assert_named_tensors(target_parameters, expected_next_parameters)


def test_optimizer_state_swap_direct_state_dict_round_trip(monkeypatch) -> None:
    _patch_fake_swap_runtime(monkeypatch)
    source_parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    source_optimizer, source_container = _build_single_parameter_container(source_parameter, lr=0.03)
    source_parameter.grad = torch.tensor([0.25, -0.5])
    source_optimizer.step()
    source_optimizer.param_groups[0]["lr"] = 0.017
    expected_state = {name: value.detach().clone() for name, value in source_optimizer.state[source_parameter].items()}
    source_state = deepcopy(source_container.state_dict())

    target_parameter = torch.nn.Parameter(source_parameter.detach().clone())
    target_optimizer, target_container = _build_single_parameter_container(target_parameter, lr=0.5)
    target_container.state_dict()
    target_identities = {
        name: value
        for name, value in target_optimizer.state[target_parameter].items()
        if name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
    }

    target_container.load_state_dict(source_state)

    assert target_optimizer.param_groups[0]["lr"] == 0.017
    for name, tensor in target_identities.items():
        assert target_optimizer.state[target_parameter][name] is tensor
    target_state = target_container.state_dict()
    for name, expected in expected_state.items():
        torch.testing.assert_close(target_state[f"state.weight.{name}"], expected, rtol=0, atol=0)

    gradient = torch.tensor([-0.125, 0.375])
    source_parameter.grad = gradient.clone()
    target_parameter.grad = gradient.clone()
    source_optimizer.step()
    target_optimizer.step()

    torch.testing.assert_close(target_parameter, source_parameter, rtol=0, atol=0)
    for name, source_value in source_optimizer.state[source_parameter].items():
        torch.testing.assert_close(
            target_optimizer.state[target_parameter][name],
            source_value,
            rtol=0,
            atol=0,
        )


def test_optimizer_state_swap_materializes_only_missing_adamw_state(
    monkeypatch,
) -> None:
    _patch_fake_swap_runtime(monkeypatch)
    initialized = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    missing = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [initialized, missing],
                "param_names": ["initialized", "missing"],
            }
        ],
        lr=0.03,
        foreach=False,
    )
    _install_adamw_swap(optimizer)
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]

    initialized.grad = torch.tensor([0.25, -0.5])
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    initialized_state = {
        name: value.detach().clone()
        for name, value in optimizer.state[initialized].items()
    }
    assert not optimizer.state[missing]

    flat_state = container.state_dict()

    for name, expected in initialized_state.items():
        torch.testing.assert_close(
            optimizer.state[initialized][name], expected, rtol=0, atol=0
        )
    assert set(optimizer.state[missing]) == {"step", "exp_avg", "exp_avg_sq"}
    assert optimizer.state[missing]["step"].item() == 0
    for state_name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            optimizer.state[missing][state_name],
            torch.zeros_like(missing),
            rtol=0,
            atol=0,
        )
    assert missing.grad is None
    for state_name in ("exp_avg", "exp_avg_sq"):
        assert f"state.initialized.{state_name}" in flat_state
        assert f"state.missing.{state_name}" in flat_state
    locations = vars(optimizer)["_torchtitan_npu_checkpoint_locations"]
    assert (initialized, "exp_avg") in locations
    assert (missing, "exp_avg") in locations


def test_optimizer_state_swap_muon_produces_and_consumes_checkpoint_location(
    monkeypatch,
) -> None:
    registered, handles = _patch_fake_swap_runtime(monkeypatch)
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    class FakeMuon(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__(
                [
                    {
                        "params": [parameter],
                        "param_names": ["weight"],
                        "lr": 0.2,
                    }
                ],
                defaults={},
            )
            self._redistribution_runtime = SimpleNamespace(
                _enqueue_storage_to_compute=lambda *args, **kwargs: None
            )

        def _momentum(self, compute_layout, grad):
            momentum = grad.detach().clone()
            self.state[compute_layout.param]["momentum_buffer"] = momentum
            return momentum

        def _prepare_local(self, compute_layout, out) -> None:
            return None

    optimizer = FakeMuon()
    _patch_container_global(monkeypatch, "DistMuon", FakeMuon)
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)
    layout = SimpleNamespace(param=parameter, fqn="weight")
    momentum = optimizer._momentum(layout, torch.tensor([3.0, 4.0]))
    container = object.__new__(OptimizerStateSwapContainer)
    container.optimizers = [optimizer]

    flat_state = container.state_dict()

    tensor_name = "optimizer_state.0.weight.momentum_buffer"
    locations = vars(optimizer)["_torchtitan_npu_checkpoint_locations"]
    assert registered[tensor_name] is momentum
    assert locations[(parameter, "momentum_buffer")] == (tensor_name, 0)
    checkpoint_view = flat_state["state.weight.momentum_buffer"]
    assert checkpoint_view.data_ptr() == handles[tensor_name][0].tensor_cpu.data_ptr()
    torch.testing.assert_close(checkpoint_view, momentum, rtol=0, atol=0)
    assert flat_state["param_groups.weight.lr"] == 0.2


def _init_two_rank_cpu_mesh(rank: int, rendezvous: str) -> DeviceMesh:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    return init_device_mesh("cpu", (2,), mesh_dim_names=("dp_shard",))


def _checkpointable_local_copy(
    local: torch.Tensor,
    metadata: dict[str, Any],
    *,
    zero: bool = False,
) -> torch.Tensor:
    value = torch.zeros_like(local) if zero else local.clone()
    raw = value.view(torch.uint8).reshape(-1)
    return product_swap.make_checkpointable_view(
        raw,
        byte_offset=0,
        dtype=local.dtype,
        shape=tuple(local.shape),
        stride=tuple(local.stride()),
        **metadata,
    )


def _empty_optimizer_checkpoint_view(tensor: DTensor) -> torch.Tensor:
    parameter = torch.nn.Parameter(torch.empty(0))
    optimizer = torch.optim.AdamW(
        [{"params": [parameter], "param_names": ["weight"]}]
    )
    optimizer.state[parameter]["exp_avg"] = tensor
    vars(optimizer)["_torchtitan_npu_checkpoint_locations"] = {}
    flat_state = {"state.weight.exp_avg": tensor}

    OptimizerStateSwapContainer._replace_swapped_states_with_checkpoint_views(
        optimizer,
        flat_state,
    )
    view = flat_state["state.weight.exp_avg"]
    assert view.device.type == "cpu"
    assert view.numel() == 0
    assert view.untyped_storage().nbytes() == 0
    assert vars(view)["global_shape"] == (1,)
    assert vars(view)["global_offsets"] == ((1,),)
    assert vars(view)["local_offsets"] == ((0,),)
    assert vars(view)["local_sizes"] == ((0,),)
    return view


def _empty_dtensor_checkpoint_worker(
    rank: int,
    rendezvous: str,
    checkpoint_dir: str,
    result_prefix: str,
) -> None:
    mesh = _init_two_rank_cpu_mesh(rank, rendezvous)
    try:
        global_tensor = torch.tensor([7.0])
        local = global_tensor.narrow(0, rank, 1) if rank == 0 else global_tensor.narrow(0, 1, 0)
        tensor = DTensor.from_local(
            local.clone(),
            mesh,
            (Shard(0),),
            shape=global_tensor.shape,
            stride=global_tensor.stride(),
            run_check=False,
        )
        local = tensor.to_local()
        metadata = product_swap._checkpoint_metadata(tensor, local)

        if rank == 0:
            source_view = _checkpointable_local_copy(local, metadata)
        else:
            source_view = _empty_optimizer_checkpoint_view(tensor)

        dcp.save({"state": source_view}, checkpoint_id=checkpoint_dir)

        target_view = (
            _checkpointable_local_copy(local, metadata, zero=True)
            if rank == 0
            else source_view
        )

        dcp.load({"state": target_view}, checkpoint_id=checkpoint_dir)
        torch.testing.assert_close(target_view, local, rtol=0, atol=0)
        Path(f"{result_prefix}.{rank}").write_text("ok")
    finally:
        dist.destroy_process_group()


def test_optimizer_state_swap_replaces_empty_dtensor_shard(tmp_path) -> None:
    if not _supports_checkpointable_tensor_protocol():
        pytest.skip(
            "requires PyTorch DCP CheckpointableTensor protocol",
        )

    rendezvous = os.fspath(tmp_path / "empty-dtensor-rendezvous")
    checkpoint_dir = os.fspath(tmp_path / "empty-dtensor-checkpoint")
    result_prefix = os.fspath(tmp_path / "empty-dtensor-result")

    mp.spawn(
        _empty_dtensor_checkpoint_worker,
        args=(rendezvous, checkpoint_dir, result_prefix),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    assert Path(f"{result_prefix}.0").read_text() == "ok"
    assert Path(f"{result_prefix}.1").read_text() == "ok"


def _dtensor_checkpoint_worker(
    rank: int,
    rendezvous: str,
    checkpoint_dir: str,
    result_prefix: str,
) -> None:
    mesh = _init_two_rank_cpu_mesh(rank, rendezvous)
    try:
        global_tensor = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        local = global_tensor.narrow(0, rank * 3, 3).clone()
        tensor = DTensor.from_local(
            local,
            mesh,
            (Shard(0),),
            shape=global_tensor.shape,
            stride=global_tensor.stride(),
            run_check=False,
        )
        metadata = product_swap._checkpoint_metadata(tensor, local)
        assert metadata == {
            "global_shape": (6, 2),
            "global_offsets": ((rank * 3, 0),),
            "local_offsets": ((0, 0),),
            "local_sizes": ((3, 2),),
        }
        source_view = _checkpointable_local_copy(local, metadata)

        dcp.save({"state": source_view}, checkpoint_id=checkpoint_dir)

        target_view = _checkpointable_local_copy(local, metadata, zero=True)
        dcp.load({"state": target_view}, checkpoint_id=checkpoint_dir)

        torch.testing.assert_close(target_view, local, rtol=0, atol=0)
        Path(f"{result_prefix}.{rank}").write_text("ok")
    finally:
        dist.destroy_process_group()


def test_checkpoint_view_preserves_two_rank_dtensor_shards(tmp_path) -> None:
    if not _supports_checkpointable_tensor_protocol():
        pytest.skip(
            "requires PyTorch DCP CheckpointableTensor protocol",
        )

    rendezvous = os.fspath(tmp_path / "dtensor-rendezvous")
    checkpoint_dir = os.fspath(tmp_path / "dtensor-checkpoint")
    result_prefix = os.fspath(tmp_path / "dtensor-result")

    mp.spawn(
        _dtensor_checkpoint_worker,
        args=(rendezvous, checkpoint_dir, result_prefix),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    assert Path(f"{result_prefix}.0").read_text() == "ok"
    assert Path(f"{result_prefix}.1").read_text() == "ok"


def test_optimizer_state_swap_has_no_unowned_cleanup_facade() -> None:
    source = product_swap.__file__
    assert source is not None
    contents = Path(source).read_text()

    assert "close_swap_state" not in contents
    assert "_torchtitan_npu_close_swap_state" not in contents
    assert "_torchtitan_npu_swap_error" not in contents


def test_novaswap_wait_for_device_release_keeps_handle_and_filters_npu(monkeypatch) -> None:
    calls = []

    class ReleaseWorker:
        def wait_for_name(self, name, *, release_target=None) -> None:
            calls.append((name, release_target))

    engine = product_swap_api.SwapEngine
    monkeypatch.setattr(engine, "_ready", True)
    monkeypatch.setattr(engine, "_release_worker", ReleaseWorker())
    monkeypatch.setattr(engine, "_handles", {"state": ()})

    product_swap_api.wait_for_device_release("state")

    assert calls == [("state", "npu")]
    assert engine._handles == {"state": ()}


def test_optimizer_state_swap_adamw_pipelines_novaswap_buckets(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(4))
    parameter_b = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    events = []
    phases = {}

    def register_tensor(tensor, name) -> None:
        events.append(("register", name, tensor))

    def execute(name, action) -> None:
        events.append(("execute", name, action))
        if action == "D2H":
            phases[name] = "D2H"

    def wait_for_device_release(name) -> None:
        events.append(("wait_for_device_release", name))

    fake_swap_api = SimpleNamespace(
        register_tensor=register_tensor,
        execute=execute,
        get_handle_phase=lambda name: phases.get(name),
        remove_tensor=lambda name: events.append(("remove", name)),
        wait_for_device_release=wait_for_device_release,
    )
    _patch_container_swap_api(monkeypatch, fake_swap_api)
    OptimizerStateSwapContainer._swap_adamw(optimizer)

    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    optimizer.step()

    names = [
        f"adamw.{id(optimizer)}.bucket.0",
        f"adamw.{id(optimizer)}.bucket.1",
    ]
    assert [event[1] for event in events if event[0] == "register"] == names
    assert [event[2] for event in events if event[0] == "execute"] == [
        "D2H",
        "D2H",
    ]
    assert [event[1] for event in events if event[0] == "wait_for_device_release"] == []
    for parameter in (parameter_a, parameter_b):
        state = optimizer.state[parameter]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["exp_avg"].untyped_storage().data_ptr() == state["exp_avg_sq"].untyped_storage().data_ptr()
    assert "step" not in optimizer.param_groups[0]

    events.clear()
    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    optimizer.step()

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "WAIT_DEVICE",
        "H2D",
        "D2H",
        "WAIT_DEVICE",
        "D2H",
    ]


def test_optimizer_state_swap_adamw_installs_flat_state_before_stock_step(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.arange(4.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    events = []
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: events.append(("register", name, tensor)),
            execute=lambda name, action: events.append((action, name)),
        ),
    )
    swap = product_swap._NovaSwapAdamW(optimizer)
    original_step = swap._original_step

    def stock_step(closure=None):
        state = optimizer.state[parameter]
        assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
        assert state["step"].device.type == "cpu"
        assert state["exp_avg"].untyped_storage().data_ptr() == state["exp_avg_sq"].untyped_storage().data_ptr()
        assert events[0][0] == "register"
        return original_step(closure)

    swap._original_step = stock_step
    parameter.grad = torch.ones_like(parameter)
    swap.step()

    assert [event[0] for event in events] == ["register", "D2H"]
    assert torch.equal(optimizer.state[parameter]["step"], torch.tensor(1.0))


def test_optimizer_state_swap_adamw_lazily_adds_bucket_for_late_gradient(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter_b = torch.nn.Parameter(torch.tensor([3.0, 4.0]))
    swapped = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    reference_a = torch.nn.Parameter(parameter_a.detach().clone())
    reference_b = torch.nn.Parameter(parameter_b.detach().clone())
    reference = torch.optim.AdamW([reference_a, reference_b], lr=1e-3, foreach=False)
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: None,
        ),
    )
    OptimizerStateSwapContainer._swap_adamw(swapped)

    parameter_a.grad = torch.full_like(parameter_a, 0.25)
    reference_a.grad = parameter_a.grad.clone()
    swapped.step()
    reference.step()
    assert parameter_b not in swapped.state

    parameter_a.grad = torch.full_like(parameter_a, 0.5)
    parameter_b.grad = torch.full_like(parameter_b, 0.75)
    reference_a.grad = parameter_a.grad.clone()
    reference_b.grad = parameter_b.grad.clone()
    swapped.step()
    reference.step()

    for swapped_parameter, reference_parameter in (
        (parameter_a, reference_a),
        (parameter_b, reference_b),
    ):
        assert torch.equal(swapped_parameter, reference_parameter)
        for key, value in reference.state[reference_parameter].items():
            assert torch.equal(swapped.state[swapped_parameter][key], value)


def test_optimizer_state_swap_adamw_initializes_amsgrad_flat_state(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.arange(3.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False, amsgrad=True)
    registered = []
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: registered.append(tensor),
            execute=lambda name, action: None,
        ),
    )
    OptimizerStateSwapContainer._swap_adamw(optimizer)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert set(state) == {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
    assert registered[0].numel() == parameter.numel() * 3
    storage = state["exp_avg"].untyped_storage().data_ptr()
    assert state["exp_avg_sq"].untyped_storage().data_ptr() == storage
    assert state["max_exp_avg_sq"].untyped_storage().data_ptr() == storage


def test_optimizer_state_swap_adamw_matches_fused_step_placement(monkeypatch) -> None:
    parameter = torch.nn.Parameter(torch.arange(3.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False, fused=True)
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: None,
        ),
    )
    OptimizerStateSwapContainer._swap_adamw(optimizer)

    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert state["step"].device == parameter.device
    assert state["step"].dtype == torch.float32
    assert state["step"].item() == 1


def test_optimizer_state_swap_adamw_runs_external_step_hooks_once(monkeypatch) -> None:
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    hooks = []
    optimizer.register_step_pre_hook(lambda *_args: hooks.append("pre"))
    optimizer.register_step_post_hook(lambda *_args: hooks.append("post"))
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: None,
        ),
    )
    OptimizerStateSwapContainer._swap_adamw(optimizer)

    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    assert hooks == ["pre", "post"]


def test_optimizer_state_swap_adamw_rejects_preinitialized_state() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, foreach=False)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    with pytest.raises(ValueError, match="must be installed before AdamW initializes optimizer state"):
        OptimizerStateSwapContainer._swap_adamw(optimizer)


def test_optimizer_state_swap_adamw_prefetches_before_current_bucket_compute(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(4))
    parameter_b = torch.nn.Parameter(torch.ones(4))
    optimizer = torch.optim.AdamW([parameter_a, parameter_b], lr=1e-3, foreach=False)
    events = []
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: events.append((action, name)),
            wait_for_device_release=lambda name: events.append(("WAIT_PRIOR_RELEASE", name)),
        ),
    )
    swap = product_swap._NovaSwapAdamW(optimizer)

    parameter_a.grad = torch.ones_like(parameter_a)
    parameter_b.grad = torch.ones_like(parameter_b)
    swap.step()

    events.clear()
    swap._original_step = lambda closure=None: events.append(("AdamW", None))
    swap.step()

    assert [action for action, _ in events] == [
        "WAIT_PRIOR_RELEASE",
        "H2D",
        "WAIT_DEVICE",
        "WAIT_PRIOR_RELEASE",
        "H2D",
        "AdamW",
        "D2H",
        "WAIT_DEVICE",
        "AdamW",
        "D2H",
    ]


def test_optimizer_state_swap_adamw_waits_for_oldest_release_at_byte_budget(monkeypatch) -> None:
    events = []
    parameters = [torch.nn.Parameter(torch.ones(4)) for _ in range(2)]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, foreach=False)
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: events.append((action, name)),
            wait_for_device_release=lambda name: events.append(name),
        ),
    )
    swap = product_swap._NovaSwapAdamW(optimizer)
    monkeypatch.setattr(product_swap, "_ADAMW_SWAP_RESIDENT_TARGET_BUCKETS", 1)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)

    swap.step()

    assert events == [
        ("D2H", f"adamw.{id(optimizer)}.bucket.0"),
        f"adamw.{id(optimizer)}.bucket.0",
        ("D2H", f"adamw.{id(optimizer)}.bucket.1"),
    ]


def test_optimizer_state_swap_adamw_keeps_stock_state_and_updates(monkeypatch) -> None:
    swapped_parameters = [
        torch.nn.Parameter(torch.tensor([1.0, 2.0])),
        torch.nn.Parameter(torch.tensor([3.0, 4.0])),
    ]
    reference_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in swapped_parameters
    ]
    swapped = torch.optim.AdamW(swapped_parameters, lr=1e-3, foreach=False)
    reference = torch.optim.AdamW(reference_parameters, lr=1e-3, foreach=False)
    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=lambda name, action: None,
        ),
    )
    OptimizerStateSwapContainer._swap_adamw(swapped)

    for _ in range(2):
        original_param_groups = swapped.param_groups
        for swapped_parameter, reference_parameter in zip(
            swapped_parameters, reference_parameters, strict=True
        ):
            grad = torch.full_like(swapped_parameter, 0.25)
            swapped_parameter.grad = grad
            reference_parameter.grad = grad.clone()
        swapped.step()
        reference.step()
        assert swapped.param_groups is original_param_groups

    for swapped_parameter, reference_parameter in zip(
        swapped_parameters, reference_parameters, strict=True
    ):
        assert torch.equal(swapped_parameter, reference_parameter)
        swapped_state = swapped.state[swapped_parameter]
        reference_state = reference.state[reference_parameter]
        assert swapped_state.keys() == reference_state.keys()
        for key in swapped_state:
            assert torch.equal(swapped_state[key], reference_state[key])


def test_optimizer_state_swap_keeps_tensor_wait_and_local_offload_at_prepare(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(2))
    parameter_b = torch.nn.Parameter(torch.ones(3))
    layout_a = SimpleNamespace(param=parameter_a, fqn="a")
    layout_b = SimpleNamespace(param=parameter_b, fqn="b")
    events = []

    class FakeMuon:
        def __init__(self) -> None:
            self.state = {
                parameter_a: {"momentum_buffer": parameter_a.detach().clone()},
                parameter_b: {"momentum_buffer": parameter_b.detach().clone()},
            }

            self._redistribution_runtime = SimpleNamespace(
                _enqueue_storage_to_compute=lambda *args, **kwargs: None,
            )

        def _momentum(self, compute_layout, grad):
            return self.state[compute_layout.param]["momentum_buffer"]

        def _prepare_local(self, compute_layout, out):
            events.append(("prepare", compute_layout))

    def execute(name, action) -> None:
        events.append(("execute", name, action))

    phases = {}

    def get_handle_phase(name):
        return phases.get(name, "D2H")

    def execute_with_phase(name, action) -> None:
        execute(name, action)
        if action == "H2D":
            phases[name] = "H2D"
        elif action == "D2H":
            phases[name] = "D2H"

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=execute_with_phase,
            get_handle_phase=get_handle_phase,
        ),
    )

    optimizer = FakeMuon()
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)

    optimizer._prepare_local(layout_a, None)
    optimizer._prepare_local(layout_b, None)

    assert [event[2] for event in events if event[0] == "execute"] == [
        "H2D",
        "WAIT_DEVICE",
        "D2H",
        "H2D",
        "WAIT_DEVICE",
        "D2H",
    ]
    assert [event[0] for event in events] == [
        "execute",
        "execute",
        "prepare",
        "execute",
        "execute",
        "execute",
        "prepare",
        "execute",
    ]


def test_optimizer_state_swap_prefetches_plan_before_a2a_and_defers_offload(monkeypatch) -> None:
    parameter_a = torch.nn.Parameter(torch.ones(2))
    parameter_b = torch.nn.Parameter(torch.ones(3))
    parameter_c = torch.nn.Parameter(torch.ones(4))
    layout_a = SimpleNamespace(param=parameter_a, fqn="a")
    layout_b = SimpleNamespace(param=parameter_b, fqn="b")
    layout_c = SimpleNamespace(param=parameter_c, fqn="c")
    events = []
    phases = {}
    streams = []

    class FakeRuntime:
        def _enqueue_storage_to_compute(self, plan, slot, context, *, prepare):
            for layout in plan.redistributed_items:
                prepare(layout, None)
            events.append(("a2a", plan))
            return "work"

    class FakeMuon:
        def __init__(self) -> None:
            self.state = {
                parameter_a: {"momentum_buffer": parameter_a.detach().clone()},
                parameter_b: {"momentum_buffer": parameter_b.detach().clone()},
                parameter_c: {"momentum_buffer": parameter_c.detach().clone()},
            }
            self._redistribution_runtime = FakeRuntime()

        def _momentum(self, compute_layout, grad):
            return self.state[compute_layout.param]["momentum_buffer"]

        def _prepare_local(self, compute_layout, out):
            events.append(("prepare", compute_layout.fqn))

    def execute(name, action) -> None:
        events.append((action, name))
        phases[name] = action

    def stream_context(stream):
        streams.append(stream)
        return nullcontext()

    _patch_container_swap_api(
        monkeypatch,
        SimpleNamespace(
            register_tensor=lambda tensor, name: None,
            execute=execute,
            get_handle_phase=lambda name: phases.get(name, "D2H"),
        ),
    )
    _patch_container_global(
        monkeypatch,
        "torch_npu",
        SimpleNamespace(npu=SimpleNamespace(stream=stream_context)),
    )

    optimizer = FakeMuon()
    OptimizerStateSwapContainer._swap_muon(optimizer, model_part=0)
    plan = SimpleNamespace(
        redistributed_items=(layout_a, layout_b),
        unredistributed_items=(layout_c,),
    )

    transfer_stream = object()
    assert optimizer._redistribution_runtime._enqueue_storage_to_compute(
        plan,
        slot=None,
        context=SimpleNamespace(transfer_stream=transfer_stream),
        prepare=optimizer._prepare_local,
    ) == "work"
    assert streams == [transfer_stream]
    assert [event[0] for event in events] == [
        "H2D",
        "H2D",
        "H2D",
        "WAIT_DEVICE",
        "prepare",
        "WAIT_DEVICE",
        "prepare",
        "a2a",
        "D2H",
        "D2H",
    ]
    assert [event[1] for event in events[:3]] == [
        product_swap.make_swap_state_name("optimizer_state", 0, "a", "momentum_buffer"),
        product_swap.make_swap_state_name("optimizer_state", 0, "b", "momentum_buffer"),
        product_swap.make_swap_state_name("optimizer_state", 0, "c", "momentum_buffer"),
    ]


@pytest.mark.parametrize("override_name", ["virtual", "swap_optimizer"])
def test_swap_preserves_host_sparse_update(monkeypatch, override_name):
    from torchtitan.components.optimizer import ParamGroupConfig

    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable
    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer

    allocations = []

    def allocate(parameter):
        result = torch.empty_like(parameter)
        allocations.append(result)
        return result

    monkeypatch.setattr(product_swap, "_make_swap", allocate)
    # CPU allocations replace only the external NovaSwap storage/transfer API.
    _patch_container_swap_api(monkeypatch, SimpleNamespace(
        register_tensor=lambda *_args, **_kwargs: None,
        execute=lambda *_args, **_kwargs: None,
    ))

    def build():
        model = torch.nn.Module()
        model.register_parameter("dense", torch.nn.Parameter(torch.ones(4)))
        model.add_module("table", HostEngramTable.Config(
            vocab_size=16, layer_id=0, ngram_orders=(2,), num_heads=1,
            head_vocab_sizes=(11,), embedding_dim=4, num_embeddings=12,
            require_token_id_map=False, pin_memory=False,
        ).build())
        with torch.no_grad():
            model.table.weight.copy_(torch.arange(48).reshape(12, 4) / 16)
        root = _Root.Config(optimizer=HostSparseOptimizersContainer.Config(
            implementation="for-loop",
            param_groups=[
                ParamGroupConfig(pattern=r"table\.weight$", optimizer_name="SparseAdam", optimizer_kwargs={"lr": 0.05}),
                ParamGroupConfig(pattern=r".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.01}),
            ],
        ))
        apply_overrides(OverrideConfig(imports=[f"torchtitan_npu.override.common.optimizer.{override_name}"]), root)
        optimizer = root.optimizer.build(model_parts=[model])
        assert isinstance(optimizer, HostSparseOptimizersContainer)
        return model, optimizer

    def step(model, optimizer, ids):
        optimizer.zero_grad()
        (model.table._distributed_lookup(ids).sum() + model.dense.sum()).backward()
        assert model.table.weight.grad is None
        optimizer.step()

    model, optimizer = build()
    reference_dense = torch.nn.Parameter(model.dense.detach().clone())
    reference_table = torch.nn.Embedding.from_pretrained(model.table.weight.detach().clone(), freeze=False, sparse=True)
    dense_opt = torch.optim.AdamW([reference_dense], lr=0.01, foreach=False, fused=False)
    sparse_opt = torch.optim.SparseAdam(reference_table.parameters(), lr=0.05)
    ids = torch.tensor([0, 7, 7, 11])
    step(model, optimizer, ids)
    (reference_table(ids).sum() + reference_dense.sum()).backward()
    dense_opt.step()
    sparse_opt.step()
    torch.testing.assert_close(model.dense, reference_dense, rtol=0, atol=0)
    torch.testing.assert_close(model.table.weight, reference_table.weight, rtol=0, atol=0)
    if override_name == "virtual":
        assert len(allocations) == 2  # Only dense AdamW moments use swap storage.
    optimizer.zero_grad()
    assert model.table.pending_sparse_grad() is None
    assert model.table.weight.grad is None


@pytest.mark.parametrize("saved_steps", [0, 3], ids=["before-first-step", "after-training"])
def test_host_sparse_swap_checkpoint_resumes_exactly(monkeypatch, tmp_path, saved_steps):
    from torchtitan.components.optimizer import ParamGroupConfig

    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import (
        HostSparseOptimizersContainer,
    )
    from torchtitan_npu.models.deepseek_v4_1.engram.host import HostEngramTable

    _patch_fake_swap_runtime(monkeypatch)

    def build():
        model = torch.nn.Module()
        model.dense = torch.nn.Parameter(torch.arange(4.0) / 7)
        model.table = HostEngramTable.Config(
            vocab_size=16,
            layer_id=0,
            ngram_orders=(2,),
            num_heads=1,
            head_vocab_sizes=(11,),
            embedding_dim=4,
            num_embeddings=12,
            require_token_id_map=False,
            pin_memory=False,
        ).build()
        with torch.no_grad():
            model.table.weight.copy_(torch.arange(48).reshape(12, 4) / 16)
        root = _Root.Config(
            optimizer=HostSparseOptimizersContainer.Config(
                implementation="for-loop",
                param_groups=[
                    ParamGroupConfig(
                        pattern=r"table\.weight$", optimizer_name="SparseAdam", optimizer_kwargs={"lr": 0.05}
                    ),
                    ParamGroupConfig(pattern=r".*", optimizer_name="AdamW", optimizer_kwargs={"lr": 0.01}),
                ],
            )
        )
        apply_overrides(OverrideConfig(imports=["torchtitan_npu.override.common.optimizer.swap_optimizer"]), root)
        container = root.optimizer.build(model_parts=[model])
        assert isinstance(container, product_swap.HostSparseOptimizerStateSwapContainer)
        return model, container

    def step(model, container):
        container.zero_grad()
        loss = model.table._distributed_lookup(torch.tensor([0, 7, 7, 11])).square().sum() + model.dense.square().sum()
        loss.backward()
        container.step()
        container.zero_grad()

    source, source_optimizer = build()
    for _ in range(saved_steps):
        step(source, source_optimizer)
    native = next(opt for opt in source_optimizer.optimizers if isinstance(opt, torch.optim.SparseAdam))
    expected_state = deepcopy(native.state.get(source.table.weight, {}))
    dcp.save({"model": source, "optimizer": source_optimizer}, checkpoint_id=tmp_path)

    target, target_optimizer = build()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.fill_(-9)
    dcp.load({"model": target, "optimizer": target_optimizer}, checkpoint_id=tmp_path)

    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = next(opt for opt in target_optimizer.optimizers if isinstance(opt, torch.optim.SparseAdam))
    restored_state = restored.state[target.table.weight]
    assert restored_state["step"] == saved_steps
    for name in ("exp_avg", "exp_avg_sq"):
        expected = expected_state[name] if saved_steps else torch.zeros_like(target.table.weight)
        torch.testing.assert_close(restored_state[name], expected, rtol=0, atol=0)
    step(source, source_optimizer)
    step(target, target_optimizer)
    for expected, actual in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hostsparse_schema_keeps_sparse_lifecycle_under_npu_state(monkeypatch) -> None:
    """Engram schemas keep the sparse lifecycle under the NPU-state default.

    swap_optimizer must also keep routing a HostSparse schema to the
    HostSparse CPU-offload container after the NPU-state derive, and the
    combined container must resolve the offload-aware dense-gradient clip
    correction (``_scale_dense_gradients``).
    """
    from torchtitan_npu.config.configs import TrainingConfig
    from torchtitan_npu.extensions import trainer as trainer_module
    from torchtitan_npu.extensions.torchtitan.components.optimizer.optimizer import HostSparseOptimizersContainer
    from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
    from torchtitan_npu.override.common.optimizer import (
        CpuOffloadHostSparseNpuStateOptimizersContainer,
        CpuOffloadHostSparseOptimizersContainer,
        swap_optimizer,
    )
    from torchtitan_npu.patches.torch_npu import cpu_dtensor_init

    class ReachedUpstreamError(Exception):
        pass

    captured: dict[str, object] = {}

    def capture_runtime(self, runtime) -> None:
        captured["optimizer"] = runtime.optimizer
        raise ReachedUpstreamError

    monkeypatch.setattr(trainer_module, "set_allow_hf32", lambda _: None)
    monkeypatch.setattr(cpu_dtensor_init, "install", lambda: None)
    monkeypatch.setattr(trainer_module.Trainer, "__init__", capture_runtime)

    optimizer_config = HostSparseOptimizersContainer.Config(_cpu_offload=True)
    config = TrainerEx.Config(
        training=TrainingConfig(enable_cpu_offload=True),
        optimizer=optimizer_config,
    )
    with pytest.raises(ReachedUpstreamError):
        TrainerEx(config)

    derived = captured["optimizer"]
    assert isinstance(derived, CpuOffloadHostSparseNpuStateOptimizersContainer.Config)
    assert isinstance(derived, HostSparseOptimizersContainer.Config)
    assert derived._cpu_offload is True
    # The dense-gradient clip correction must resolve to the offload-aware
    # implementation, not the plain HostSparse fallback.
    assert (
        CpuOffloadHostSparseNpuStateOptimizersContainer._scale_dense_gradients
        is CpuOffloadHostSparseOptimizersContainer._scale_dense_gradients
    )

    cpu_state = swap_optimizer(derived)
    assert isinstance(cpu_state, CpuOffloadHostSparseOptimizersContainer.Config)
    assert not isinstance(cpu_state, CpuOffloadHostSparseNpuStateOptimizersContainer.Config)
