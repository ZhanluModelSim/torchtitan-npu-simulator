# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts for the upstream effect predicate backport."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint
from torch._higher_order_ops.effects import EffectType, _register_effectful_op, has_effects

_MODULE_PATH = Path(__file__).resolve().parents[4] / "torchtitan_npu/patches/torch/utils/checkpoint.py"


@pytest.fixture
def checkpoint_patch():
    missing = object()
    original = getattr(torch.utils.checkpoint, "_is_cacheable_effect", missing)
    if original is not missing:
        del torch.utils.checkpoint._is_cacheable_effect
    spec = importlib.util.spec_from_file_location("checkpoint_effect_predicate_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        if original is missing:
            del torch.utils.checkpoint._is_cacheable_effect
        else:
            torch.utils.checkpoint._is_cacheable_effect = original


@pytest.fixture
def registered_effects():
    libraries = [torch.library.Library("codex_checkpoint_effects", "DEF"), torch.library.Library("c10d", "FRAGMENT")]
    libraries[0].define("effect(Tensor x) -> Tensor")
    libraries[1].define("_codex_checkpoint_effect(Tensor x) -> Tensor")
    operators = (torch.ops.codex_checkpoint_effects.effect.default, torch.ops.c10d._codex_checkpoint_effect.default)
    handles = [_register_effectful_op(op, EffectType.ORDERED) for op in operators]
    try:
        yield operators
    finally:
        for handle in reversed(handles):
            handle.destroy()
        for library in reversed(libraries):
            library._destroy()


def test_predicate_uses_registered_effects_and_excludes_raw_c10d(checkpoint_patch, registered_effects):
    non_collective, raw_collective = registered_effects

    assert has_effects(non_collective)
    assert has_effects(raw_collective)
    assert torch.utils.checkpoint._is_cacheable_effect is checkpoint_patch._is_cacheable_effect
    assert checkpoint_patch._is_cacheable_effect(non_collective)
    assert not checkpoint_patch._is_cacheable_effect(raw_collective)
    assert not checkpoint_patch._is_cacheable_effect(torch.ops.aten.add.Tensor)
    assert not checkpoint_patch._is_cacheable_effect(lambda value: value)


def test_apply_preserves_a_native_predicate(checkpoint_patch, monkeypatch):
    native_predicate = lambda op: "native"
    monkeypatch.setattr(torch.utils.checkpoint, "_is_cacheable_effect", native_predicate)

    checkpoint_patch.apply()

    assert torch.utils.checkpoint._is_cacheable_effect is native_predicate
