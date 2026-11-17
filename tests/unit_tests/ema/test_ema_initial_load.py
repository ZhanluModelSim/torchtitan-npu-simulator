#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regression coverage for EMA math and checkpoint initial-load behaviour."""

from __future__ import annotations

import importlib.machinery
import subprocess
import sys
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from collections.abc import Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
INITIAL_VALUE = 1.0
LOADED_VALUE = 3.14159
EMAModules = tuple[Any, Any, str, Any]
_TORCHTITAN_NPU_MODULE_PREFIX = "torchtitan_npu"
_PATCHED_CONFIG_ATTRIBUTES = ("_owner", "build")


def _is_torchtitan_npu_module(module_name: str) -> bool:
    return module_name == _TORCHTITAN_NPU_MODULE_PREFIX or module_name.startswith(f"{_TORCHTITAN_NPU_MODULE_PREFIX}.")


def _clear_torchtitan_npu_modules() -> None:
    for module_name in list(sys.modules):
        if _is_torchtitan_npu_module(module_name):
            del sys.modules[module_name]


def _restore_config_attributes(checkpoint_config: Any, original_attributes: dict[str, object], missing: object) -> None:
    for name, value in original_attributes.items():
        if value is missing:
            if name in vars(checkpoint_config):
                delattr(checkpoint_config, name)
        else:
            setattr(checkpoint_config, name, value)


def _register_packages(package_paths: list[tuple[str, Path]]) -> None:
    for package_name, package_path in package_paths:
        if package_name in sys.modules:
            continue
        module = types.ModuleType(package_name)
        module.__file__ = str(package_path / "__init__.py")
        module.__package__ = package_name
        module.__path__ = [str(package_path)]
        spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        spec.submodule_search_locations = [str(package_path)]
        module.__spec__ = spec
        sys.modules[package_name] = module


def _load_ema_modules() -> EMAModules:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    package_root = REPOSITORY_ROOT / "torchtitan_npu"
    _register_packages(
        [
            ("torchtitan_npu", package_root),
            ("torchtitan_npu.patches", package_root / "patches"),
            ("torchtitan_npu.patches.torchtitan", package_root / "patches" / "torchtitan"),
            (
                "torchtitan_npu.patches.torchtitan.components",
                package_root / "patches" / "torchtitan" / "components",
            ),
        ]
    )
    from torchtitan_npu.patches.torchtitan.components import checkpoint as checkpoint_module
    from torchtitan_npu.patches.torchtitan.components.checkpoint import OriginalCheckpointManager
    from torchtitan_npu.patches.torchtitan.components.ema import EMA_OPTIMIZER, EMAOptimizersContainer

    return checkpoint_module, OriginalCheckpointManager, EMA_OPTIMIZER, EMAOptimizersContainer


def _load_checkpoint_conversion_module() -> types.ModuleType:
    package_root = REPOSITORY_ROOT / "torchtitan_npu"
    _register_packages(
        [
            ("torchtitan_npu", package_root),
            ("torchtitan_npu.tools", package_root / "tools"),
            ("torchtitan_npu.tools.checkpoint_conversion", package_root / "tools" / "checkpoint_conversion"),
        ]
    )
    return importlib.import_module("torchtitan_npu.tools.checkpoint_conversion.convert_to_hf")


@pytest.fixture
def isolated_ema_environment():
    import torchtitan.components.checkpoint as upstream_checkpoint
    import torchtitan.components.checkpointer as upstream_checkpointer
    import torchtitan.trainer as upstream_trainer

    original_modules = {name: module for name, module in sys.modules.items() if _is_torchtitan_npu_module(name)}
    original_sys_path = list(sys.path)
    original_trainer = upstream_trainer.Trainer
    original_post_dataloading_process = original_trainer.post_dataloading_process
    # torchtitan v0.3.0 keeps CheckpointManager in the checkpointer package with
    # a legacy shim re-export; the EMA patch replaces the name in all three
    # namespaces, so all three are saved and restored here.
    original_checkpoint_managers = (
        upstream_checkpoint.CheckpointManager,
        upstream_checkpointer.CheckpointManager,
        upstream_checkpointer.dcp.CheckpointManager,
    )
    checkpoint_config = original_checkpoint_managers[0].Config
    missing = object()
    original_config_attributes = {
        name: vars(checkpoint_config).get(name, missing) for name in _PATCHED_CONFIG_ATTRIBUTES
    }
    _clear_torchtitan_npu_modules()

    try:
        yield
    finally:
        sys.path[:] = original_sys_path
        upstream_trainer.Trainer = original_trainer
        original_trainer.post_dataloading_process = original_post_dataloading_process
        upstream_checkpoint.CheckpointManager = original_checkpoint_managers[0]
        upstream_checkpointer.CheckpointManager = original_checkpoint_managers[1]
        upstream_checkpointer.dcp.CheckpointManager = original_checkpoint_managers[2]
        _restore_config_attributes(checkpoint_config, original_config_attributes, missing)
        _clear_torchtitan_npu_modules()
        sys.modules.update(original_modules)


@pytest.fixture
def ema_modules(isolated_ema_environment) -> EMAModules:
    return _load_ema_modules()


class _FakeStateful:
    @staticmethod
    def state_dict() -> dict[str, object]:
        return {}

    @staticmethod
    def load_state_dict(_state_dict: dict[str, object]) -> None:
        return None


class _IdentityHFAdapter:
    hf_assets_path = None

    @staticmethod
    def to_hf(state_dict: dict[str, object]) -> dict[str, object]:
        return state_dict

    @staticmethod
    def from_hf(state_dict: dict[str, object]) -> dict[str, object]:
        return state_dict

    @staticmethod
    def get_hf_storage_reader(_checkpoint_id: str, _from_quantized: bool) -> object:
        return object()


def _ema_tensors(ema_optimizer: Any) -> list[torch.Tensor]:
    return [
        parameter_state["ema_params"]
        for optimizer in ema_optimizer.optimizers
        for parameter_state in optimizer.state.values()
    ]


def _fill_module(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _fill_tensors(state_dict: dict[str, object], value: float) -> None:
    for tensor in state_dict.values():
        if torch.is_tensor(tensor):
            tensor.fill_(value)


def _assert_all_tensors(tensors: list[torch.Tensor], expected: float) -> None:
    for tensor in tensors:
        assert torch.allclose(tensor, torch.full_like(tensor, expected), atol=1e-6)


def _make_checkpoint_manager(
    ema_modules: EMAModules,
    tmp_path: Path,
    *,
    initial_load_in_hf: bool = False,
    state_dict_adapter: object | None = None,
) -> tuple[nn.Module, Any, Any]:
    checkpoint_module, _checkpoint_manager, _ema_optimizer_name, ema_optimizers_container = ema_modules
    initial_path = tmp_path / "initial"
    initial_path.mkdir()
    model = nn.Linear(2, 2)
    _fill_module(model, INITIAL_VALUE)
    ema_config = ema_optimizers_container.Config(enable=True, decay=0.9)

    config = checkpoint_module.EMACheckpointManager.Config(
        enable=True,
        folder="checkpoint",
        interval=1,
        keep_latest_k=0,
        last_save_model_only=False,
        export_dtype="float32",
        exclude_from_loading=[],
        initial_load_path=str(initial_path),
        initial_load_model_only=True,
        initial_load_in_hf=initial_load_in_hf,
        async_mode="disabled",
    )
    config.ema_weights = ema_config
    manager = config.build(
        dataloader=_FakeStateful(),
        model_parts=[model],
        optimizers=_FakeStateful(),
        lr_schedulers=_FakeStateful(),
        states={},
        sd_adapter=state_dict_adapter,
        base_folder=str(tmp_path / "output"),
    )
    ema_optimizer = manager.ema_optimizer
    return model, manager, ema_optimizer


def test_fixed_decay_updates_only_shadow(ema_modules: EMAModules) -> None:
    _, _, _, ema_optimizers_container = ema_modules
    model = nn.Linear(2, 2, bias=False)
    _fill_module(model, INITIAL_VALUE)
    ema_optimizer = ema_optimizers_container(
        ema_optimizers_container.Config(enable=True, decay=0.8), model_parts=[model]
    )
    _fill_module(model, 5.0)
    ema_optimizer.step(0)
    _assert_all_tensors(_ema_tensors(ema_optimizer), 1.8)
    _assert_all_tensors(list(model.parameters()), 5.0)


def test_disabled_ema_is_a_noop(ema_modules: EMAModules) -> None:
    _, _, _, ema_optimizers_container = ema_modules
    model = nn.Linear(2, 2, bias=False)
    ema_optimizer = ema_optimizers_container(
        ema_optimizers_container.Config(enable=False, decay=0.5), model_parts=[model]
    )
    _fill_module(model, 9.0)
    ema_optimizer.step(0)
    assert not _ema_tensors(ema_optimizer)
    assert ema_optimizer.state_dict() == {}


def test_schedule_parameters_control_updates(ema_modules: EMAModules) -> None:
    _, _, _, ema_optimizers_container = ema_modules
    model = nn.Linear(2, 2, bias=False)
    _fill_module(model, INITIAL_VALUE)
    ema_optimizer = ema_optimizers_container(
        ema_optimizers_container.Config(
            enable=True, decay=None, half_life_fraction=0.05, start_step=2, step_bias=3, update_every_n_steps=3
        ),
        model_parts=[model],
    )
    _fill_module(model, 4.0)
    before_start = [tensor.clone() for tensor in _ema_tensors(ema_optimizer)]
    ema_optimizer.step(1)
    assert all(
        torch.equal(before, after) for before, after in zip(before_start, _ema_tensors(ema_optimizer), strict=True)
    )
    ema_optimizer.step(2)
    decay = 2.0 ** (-1.0 / 0.05)
    _assert_all_tensors(_ema_tensors(ema_optimizer), decay * INITIAL_VALUE + (1.0 - decay) * 4.0)


def test_ema_state_roundtrip_and_cold_start(ema_modules: EMAModules) -> None:
    _, _, _, ema_optimizers_container = ema_modules
    model = nn.Linear(2, 2, bias=False)
    _fill_module(model, INITIAL_VALUE)
    source = ema_optimizers_container(ema_optimizers_container.Config(enable=True, decay=0.5), model_parts=[model])
    _fill_module(model, 3.0)
    source.step(0)
    restored_model = nn.Linear(2, 2, bias=False)
    restored = ema_optimizers_container(
        ema_optimizers_container.Config(enable=True, decay=0.5), model_parts=[restored_model]
    )
    restored.load_state_dict(source.state_dict())
    for source_tensor, restored_tensor in zip(_ema_tensors(source), _ema_tensors(restored), strict=True):
        assert torch.allclose(source_tensor, restored_tensor)
    _fill_module(restored_model, LOADED_VALUE)
    restored.load_state_dict({})
    _assert_all_tensors(_ema_tensors(restored), LOADED_VALUE)


def test_native_model_only_load_reseeds_ema(ema_modules: EMAModules, tmp_path: Path) -> None:
    _, checkpoint_manager, _, _ = ema_modules
    model, manager, ema_optimizer = _make_checkpoint_manager(ema_modules, tmp_path)

    def fake_dcp_load(_manager: object, state_dict: dict[str, object], *_args: object) -> None:
        _fill_tensors(state_dict, LOADED_VALUE)

    try:
        with patch.object(checkpoint_manager, "dcp_load", fake_dcp_load):
            assert manager.load()
        _assert_all_tensors(list(model.parameters()), LOADED_VALUE)
        _assert_all_tensors(_ema_tensors(ema_optimizer), LOADED_VALUE)
    finally:
        manager.close()


def test_hf_model_only_load_reseeds_ema(ema_modules: EMAModules, tmp_path: Path) -> None:
    model, manager, ema_optimizer = _make_checkpoint_manager(
        ema_modules,
        tmp_path,
        initial_load_in_hf=True,
        state_dict_adapter=_IdentityHFAdapter(),
    )

    def fake_dcp_load(state_dict: dict[str, object], **_kwargs: object) -> None:
        _fill_tensors(state_dict, LOADED_VALUE)

    try:
        with patch("torchtitan.components.checkpointer.dcp.dcp.load", fake_dcp_load):
            assert manager.load()
        _assert_all_tensors(list(model.parameters()), LOADED_VALUE)
        _assert_all_tensors(_ema_tensors(ema_optimizer), LOADED_VALUE)
    finally:
        manager.close()


def _fake_checkpoint_manager_type(checkpoint_manager: Any, name: str) -> type[Any]:
    class FakeCheckpointManager(checkpoint_manager):
        @dataclass(kw_only=True, slots=True)
        class Config(checkpoint_manager.Config):
            pass

        def __init__(self, config: Any, **kwargs: Any) -> None:
            super().__init__(config, **kwargs)

        def close(self) -> None:
            return None

    FakeCheckpointManager.__name__ = name
    return FakeCheckpointManager


def _fake_base_checkpoint_init(manager: Any, config: Any, **kwargs: Any) -> None:
    manager.enable = config.enable
    manager.stager = None
    if manager.enable:
        manager.states = kwargs["states"]


def _build_checkpoint_manager_with_ema(
    checkpoint_module: Any,
    checkpoint_manager: Any,
    config: Any,
    ema_config: Any,
    model: nn.Module,
) -> Any:
    checkpoint_module.apply()
    config.ema_weights = ema_config
    with patch.object(checkpoint_manager, "__init__", _fake_base_checkpoint_init):
        return config.build(model_parts=[model], states={})


@pytest.mark.parametrize("name", ["NPUCheckpointManager", "NPUVirtualCheckpointManager"])
def test_npu_checkpoint_manager_receives_ema_during_construction(ema_modules: EMAModules, name: str) -> None:
    checkpoint_module, checkpoint_manager, ema_optimizer_name, ema_optimizers_container = ema_modules
    manager_type = _fake_checkpoint_manager_type(checkpoint_module.EMACheckpointManager, name)
    model = nn.Linear(2, 2, bias=False)
    ema_config = ema_optimizers_container.Config(enable=True)

    manager = _build_checkpoint_manager_with_ema(
        checkpoint_module,
        checkpoint_manager,
        manager_type.Config(enable=True),
        ema_config,
        model,
    )

    assert isinstance(manager, manager_type)
    assert manager.ema_optimizer.optimizers
    assert manager.states[ema_optimizer_name] is manager.ema_optimizer


def test_checkpoint_manager_receives_ema_during_construction(ema_modules: EMAModules) -> None:
    checkpoint_module, checkpoint_manager, ema_optimizer_name, ema_optimizers_container = ema_modules
    model = nn.Linear(2, 2, bias=False)
    ema_config = ema_optimizers_container.Config(enable=True)

    manager = _build_checkpoint_manager_with_ema(
        checkpoint_module,
        checkpoint_manager,
        checkpoint_module.EMACheckpointManager.Config(enable=True),
        ema_config,
        model,
    )

    assert manager.ema_optimizer.optimizers
    assert manager.states[ema_optimizer_name] is manager.ema_optimizer


def test_parallel_file_system_reader_completes_empty_plan(isolated_ema_environment, tmp_path: Path) -> None:
    checkpoint_conversion = _load_checkpoint_conversion_module()
    reader = checkpoint_conversion.ParallelFileSystemReader(tmp_path)

    future = reader.read_data(types.SimpleNamespace(items=[]), object())

    assert future.wait() is None


def test_parallel_file_system_reader_keeps_nonempty_file_parallelism(isolated_ema_environment, tmp_path: Path) -> None:
    checkpoint_conversion = _load_checkpoint_conversion_module()
    reader = checkpoint_conversion.ParallelFileSystemReader(tmp_path, thread_count=16)
    reader.storage_data = {
        0: types.SimpleNamespace(relative_path="shard-0"),
        1: types.SimpleNamespace(relative_path="shard-1"),
    }
    worker_counts: list[int] = []
    dispatched_files: list[list[str]] = []

    class RecordingExecutor:
        def __init__(self, max_workers: int) -> None:
            worker_counts.append(max_workers)

        def __enter__(self) -> RecordingExecutor:
            return self

        @staticmethod
        def __exit__(*args: object) -> None:
            return None

        @staticmethod
        def map(_fn: object, items: Iterable[tuple[str, object]]) -> list[None]:
            dispatched_files.append([path for path, _ in items])
            return []

    plan = types.SimpleNamespace(items=[types.SimpleNamespace(storage_index=0), types.SimpleNamespace(storage_index=1)])
    with patch.object(checkpoint_conversion, "ThreadPoolExecutor", RecordingExecutor):
        assert reader.read_data(plan, object()).wait() is None

    assert worker_counts == [2]
    assert dispatched_files == [["shard-0", "shard-1"]]


def test_ema_waits_for_parameter_reads_before_optimizer_writes(
    ema_modules: EMAModules, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, _, ema_optimizers_container = ema_modules
    model = nn.Linear(2, 2, bias=False)
    ema_optimizer = ema_optimizers_container(ema_optimizers_container.Config(enable=True), model_parts=[model])
    waited_events: list[object] = []
    event = object()

    class RecordingStream:
        @staticmethod
        def wait_event(waited_event: object) -> None:
            waited_events.append(waited_event)

    stream = RecordingStream()

    class FakeDeviceModule:
        @staticmethod
        def current_stream() -> RecordingStream:
            return stream

    ema_module = sys.modules[ema_optimizers_container.__module__]
    monkeypatch.setattr(ema_module.utils, "device_module", FakeDeviceModule)
    vars(ema_optimizer)["_param_read_event"] = event

    ema_optimizer.wait_for_param_reads()

    assert waited_events == [event]
    assert vars(ema_optimizer)["_param_read_event"] is None


def test_invalid_ema_configuration_fails_fast(ema_modules: EMAModules) -> None:
    _, _, _, ema_optimizers_container = ema_modules
    for options in ("negative_decay", "large_decay", "zero_interval", "zero_half_life"):
        values = {
            "negative_decay": {"decay": -0.1},
            "large_decay": {"decay": 1.1},
            "zero_interval": {"update_every_n_steps": 0},
            "zero_half_life": {"half_life_fraction": 0.0},
        }[options]
        with pytest.raises(ValueError):
            ema_optimizers_container.Config(enable=True, **values)


def test_ema_schema_is_inherited_by_npu_trainer_ex_config() -> None:
    pytest.importorskip("triton")
    pytest.importorskip("torch_npu")

    # A full package import registers custom ops; sys.modules rollback cannot
    # restore the PyTorch dispatcher after registering the same ops again.
    script = textwrap.dedent("""
        from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
        from torchtitan_npu.patches.torchtitan import trainer as trainer_module
        from torchtitan_npu.patches.torchtitan.components.checkpoint import EMACheckpointManager
        import torchtitan.components.checkpoint as upstream_checkpoint
        import torchtitan.trainer as upstream_trainer
        from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

        assert upstream_trainer.Trainer is trainer_module.EMATrainer
        assert "ema_weights" in trainer_module.EMATrainer.Config.__dataclass_fields__
        assert issubclass(GraphTrainer.Config, trainer_module.EMATrainer.Config)
        assert "ema_weights" in GraphTrainer.Config.__dataclass_fields__
        config = TrainerEx.Config()
        assert isinstance(config, trainer_module.EMATrainer.Config)
        assert config.ema_weights.enable is False
    """)
    subprocess.run([sys.executable, "-c", script], cwd=REPOSITORY_ROOT, check=True, timeout=120)
