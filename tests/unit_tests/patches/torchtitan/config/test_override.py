# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for grouped override logs with folded FQN ranges (#6180, #6315)."""

import importlib
import importlib.machinery
import logging
import sys
import types
from collections import Counter
from dataclasses import dataclass, field

import pytest
from torchtitan.config import OverrideConfig, apply_overrides, derive, override
from torchtitan.config.configurable import Configurable
from torchtitan.models.common.nn_modules import RMSNorm

import torchtitan_npu  # noqa: F401
from torchtitan_npu.patches.torchtitan.config import override as override_patch
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

pytestmark = pytest.mark.cpu

_override_engine = importlib.import_module("torchtitan.config.override")


class _ComponentA(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 128


class _DerivedComponentA(_ComponentA):
    @dataclass(kw_only=True, slots=True)
    class Config(_ComponentA.Config):
        pass


class _ComponentB(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        extra: int = 0


class _UnmatchedComponent(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int = 64


class _Parent(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        child: _ComponentA.Config = field(default_factory=_ComponentA.Config)
        children: list[_ComponentA.Config] = field(default_factory=list)


class _TaggedRMSNorm(RMSNorm):
    @dataclass(kw_only=True, slots=True)
    class Config(RMSNorm.Config):
        pass


@pytest.fixture
def isolated_override_registry():
    saved = _override_engine._REGISTRY.copy()
    yield
    _override_engine._REGISTRY.clear()
    _override_engine._REGISTRY.update(saved)


def _swap_to_b(cfg: _ComponentA.Config, *, extra: int) -> _ComponentB.Config:
    return _ComponentB.Config(dim=cfg.dim, extra=extra)


def _swap_unmatched(cfg: _UnmatchedComponent.Config) -> _ComponentB.Config:
    return _ComponentB.Config(dim=cfg.dim)


def _swap_rmsnorm(cfg: RMSNorm.Config) -> _TaggedRMSNorm.Config:
    return derive(cfg, _TaggedRMSNorm.Config)


def test_package_import_patches_apply_overrides_and_folds_indexed_fqns(
    isolated_override_registry,
    caplog,
) -> None:
    target = f"{__name__}._swap_to_b"
    unmatched_target = f"{__name__}._swap_unmatched"
    override(target=_ComponentA.Config)(_swap_to_b)
    override(target=_UnmatchedComponent.Config)(_swap_unmatched)

    config_mod = importlib.import_module("torchtitan.config")
    trainer_mod = importlib.import_module("torchtitan.trainer")
    assert config_mod.apply_overrides is override_patch.patched_apply_overrides
    assert sys.modules["torchtitan.config.override"].apply_overrides is override_patch.patched_apply_overrides
    assert trainer_mod.apply_overrides is override_patch.patched_apply_overrides

    parent_cfg = _Parent.Config(
        children=[
            _ComponentA.Config(),
            _ComponentA.Config(),
            _ComponentA.Config(),
            _DerivedComponentA.Config(),
            _ComponentA.Config(),
            _ComponentA.Config(),
        ]
    )

    with caplog.at_level(logging.INFO, logger=_override_engine.logger.name):
        replacements = apply_overrides(OverrideConfig(imports=[(target, {"extra": 256})]), parent_cfg)

    override_messages = [
        rec.getMessage() for rec in caplog.records if rec.getMessage().startswith(("[Override] ", "Applied "))
    ]
    assert override_messages == [
        f"[Override] {target}: _ComponentA.Config -> _ComponentB.Config with extra=256"
        " matched=6 fqns=['child', 'children.{0..2,4,5}']",
        f"[Override] {target}: _DerivedComponentA.Config -> _ComponentB.Config with extra=256"
        " matched=1 fqns=['children.3']",
        "Applied 7 override(s)",
    ]
    assert replacements == [
        f"[Override] {target}: child _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.0 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.1 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.2 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.3 _DerivedComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.4 _ComponentA.Config -> _ComponentB.Config with extra=256",
        f"[Override] {target}: children.5 _ComponentA.Config -> _ComponentB.Config with extra=256",
    ]
    assert isinstance(parent_cfg.child, _ComponentB.Config)
    assert parent_cfg.child.extra == 256
    assert all(isinstance(node, _ComponentB.Config) and node.extra == 256 for node in parent_cfg.children)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_override_engine.logger.name):
        empty = apply_overrides(OverrideConfig(imports=[unmatched_target]), parent_cfg)
    assert empty == []
    assert any("produced no replacements" in rec.getMessage() for rec in caplog.records)


def test_apply_overrides_folds_layer_fqns_on_deepseek_v4_debugmodel(
    isolated_override_registry,
    caplog,
) -> None:
    target = f"{__name__}._swap_rmsnorm"
    override(target=RMSNorm.Config)(_swap_rmsnorm)

    cfg = deepseek_v4_debugmodel()
    assert cfg.model is not None
    matched_nodes = list(cfg.traverse(RMSNorm.Config, recurse=True))
    expected_fqns = [fqn for fqn, _, _, _ in matched_nodes]
    expected_counts = Counter(type(node).__qualname__ for _, node, _, _ in matched_nodes)
    num_layers = len(cfg.model.layers)
    assert num_layers >= 2
    layer_range = f"{{0..{num_layers - 1}}}"
    assert any(fqn.startswith("model.layers.0.") for fqn in expected_fqns)

    with caplog.at_level(logging.INFO, logger=_override_engine.logger.name):
        replacements = apply_overrides(OverrideConfig(imports=[target]), cfg)

    assert len(replacements) == len(expected_fqns)
    grouped_msgs = [rec.getMessage() for rec in caplog.records if rec.getMessage().startswith(f"[Override] {target}: ")]
    assert len(grouped_msgs) == len(expected_counts)
    for cls_name, count in expected_counts.items():
        msg = next(m for m in grouped_msgs if f"{cls_name} -> _TaggedRMSNorm.Config" in m)
        assert f"matched={count} " in msg
    rmsnorm_msg = next(m for m in grouped_msgs if "RMSNorm.Config -> _TaggedRMSNorm.Config" in m)
    unitscale_msg = next(m for m in grouped_msgs if "UnitScaleRMSNorm.Config -> _TaggedRMSNorm.Config" in m)
    assert f"model.layers.{layer_range}.attention_norm" in rmsnorm_msg
    assert "model.layers.0.attention_norm" not in rmsnorm_msg
    assert f"model.layers.{layer_range}.attention.q_head_norm" in unitscale_msg


def test_apply_rebinds_preloaded_caller_modules(monkeypatch) -> None:
    import torchtitan.config

    monkeypatch.setattr(
        sys.modules["torchtitan.config.override"],
        "apply_overrides",
        override_patch._original_apply_overrides,
    )
    monkeypatch.setattr(torchtitan.config, "apply_overrides", override_patch._original_apply_overrides)

    caller_names = override_patch._CALLER_MODULES[:2]
    preloaded: dict[str, types.ModuleType] = {}
    for module_name in caller_names:
        mod = types.ModuleType(module_name)
        mod.apply_overrides = override_patch._original_apply_overrides  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, module_name, mod)
        preloaded[module_name] = mod

    override_patch.apply()

    assert sys.modules["torchtitan.config.override"].apply_overrides is override_patch.patched_apply_overrides
    assert torchtitan.config.apply_overrides is override_patch.patched_apply_overrides
    for mod in preloaded.values():
        assert mod.apply_overrides is override_patch.patched_apply_overrides


def test_apply_skips_when_upstream_provides_fold_indexed_paths(monkeypatch) -> None:
    import torchtitan.config

    monkeypatch.setattr(
        sys.modules["torchtitan.config.override"],
        "apply_overrides",
        override_patch._original_apply_overrides,
    )
    monkeypatch.setattr(torchtitan.config, "apply_overrides", override_patch._original_apply_overrides)

    fake_format = types.ModuleType("torchtitan.config.format")
    fake_format.__spec__ = importlib.machinery.ModuleSpec("torchtitan.config.format", loader=None)
    fake_format.fold_indexed_paths = lambda paths: list(paths)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchtitan.config.format", fake_format)

    assert override_patch._upstream_has_folded_override_logs() is True
    override_patch.apply()

    assert sys.modules["torchtitan.config.override"].apply_overrides is override_patch._original_apply_overrides
    assert torchtitan.config.apply_overrides is override_patch._original_apply_overrides
