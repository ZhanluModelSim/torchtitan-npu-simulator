# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests for the backported compact lossless config formatter patch (#6290)."""

import itertools
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import pytest
import torch.distributed as dist
import torchtitan.trainer
from torchtitan.config import ConfigLoader
from torchtitan.config.configurable import Configurable

import torchtitan_npu  # noqa: F401
from torchtitan_npu.extensions.torchtitan.trainer import TrainerEx
from torchtitan_npu.patches.torchtitan.config.format import format_config
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

pytestmark = pytest.mark.cpu


def _make_init_closure(scale: float) -> Callable[[float], float]:
    def _init(x: float) -> float:
        return x * scale

    return _init


def _scaled_fn(x: float, *, std: float) -> float:
    return x * std


def _expand_segment(seg: str) -> list[str]:
    if not (seg.startswith("{") and seg.endswith("}")):
        return [seg]
    expanded: list[str] = []
    for part in seg[1:-1].split(","):
        if ".." in part:
            start_s, end_s = part.split("..", 1)
            expanded.extend(str(i) for i in range(int(start_s), int(end_s) + 1))
        else:
            expanded.append(part)
    return expanded


def _expand_folded(mapping: dict[str, Any]) -> dict[str, Any]:
    """Expand every ``{...}`` index segment back into flat dotted keys."""
    expanded: dict[str, Any] = {}
    for key, val in mapping.items():
        seg_choices = [_expand_segment(seg) for seg in re.split(r"\.(?![^{}]*\})", key)]
        for combo in itertools.product(*seg_choices):
            flat_key = ".".join(combo)
            assert flat_key not in expanded, f"overlapping folded key: {flat_key}"
            expanded[flat_key] = val
    return expanded


class _DenseLinear(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        param_init: dict[str, Callable] = field(default_factory=dict)


class _QuantLinear(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        # Intentionally identical field names to _DenseLinear.Config.
        dim: int
        param_init: dict[str, Callable] = field(default_factory=dict)


class _MoEBlock(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        dim: int
        num_experts: int


@dataclass(frozen=True, slots=True)
class _HeadSpec:
    head_dim: int


class _Layer(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        ffn: Configurable.Config
        layer_id: int
        heads: list[_HeadSpec] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _OpaqueLayout:
    spec: str

    def to_dict(self) -> dict[str, str]:
        return {"repr": repr(self)}


class _Model(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        layers: list[_Layer.Config]
        sharding_by_fqn: dict[str, _OpaqueLayout] = field(default_factory=dict)


def test_format_config_losslessly_folds_repeated_entries() -> None:
    num_layers = 10
    layers = [
        _Layer.Config(
            layer_id=i,
            ffn=_DenseLinear.Config(
                dim=2048 if i == 3 else 4096,
                param_init={
                    "w1": _make_init_closure(1.0),
                    "w2": partial(_scaled_fn, std=0.02 / (i + 1)),
                },
            ),
            heads=[_HeadSpec(head_dim=64), _HeadSpec(head_dim=64)],
        )
        for i in range(num_layers)
    ]
    owned = _OpaqueLayout("Owned()")
    sharded = _OpaqueLayout("Shard(0)")
    cfg = _Model.Config(
        layers=layers,
        sharding_by_fqn={
            **{f"layers.{i}.wq.weight": owned for i in range(5)},
            "layers.0.wo.weight": owned,
            "layers.1.wo.weight": sharded,
            "layers.2.wo.weight": owned,
        },
    )

    formatted = format_config(cfg)
    assert " at 0x" not in formatted
    folded = json.loads(formatted)
    expanded = _expand_folded(folded)

    assert len(folded) < len(expanded)

    for i in range(num_layers):
        assert expanded[f"layers.{i}.layer_id"] == i
        assert expanded[f"layers.{i}.ffn.dim"] == (2048 if i == 3 else 4096)
        assert f"std={0.02 / (i + 1)}" in expanded[f"layers.{i}.ffn.param_init.w2"]
        assert expanded[f"layers.{i}.ffn.param_init.w1"] == "<function _make_init_closure.<locals>._init>"
        assert expanded[f"layers.{i}.heads.0.head_dim"] == 64
        assert expanded[f"layers.{i}.heads.1.head_dim"] == 64

    for i in range(5):
        assert expanded[f"sharding_by_fqn.layers.{i}.wq.weight"] == repr(owned)
    assert expanded["sharding_by_fqn.layers.0.wo.weight"] == repr(owned)
    assert expanded["sharding_by_fqn.layers.1.wo.weight"] == repr(sharded)
    assert expanded["sharding_by_fqn.layers.2.wo.weight"] == repr(owned)


def test_format_config_preserves_sibling_config_subclasses() -> None:
    layers = [
        _Layer.Config(layer_id=0, ffn=_DenseLinear.Config(dim=1024)),
        _Layer.Config(layer_id=1, ffn=_DenseLinear.Config(dim=1024)),
        _Layer.Config(layer_id=2, ffn=_QuantLinear.Config(dim=1024)),
        _Layer.Config(layer_id=3, ffn=_MoEBlock.Config(dim=1024, num_experts=8)),
        _Layer.Config(layer_id=4, ffn=_MoEBlock.Config(dim=1024, num_experts=8)),
    ]
    cfg = _Model.Config(layers=layers)

    folded = json.loads(format_config(cfg))
    expanded = _expand_folded(folded)

    assert expanded["layers.0.ffn.__class__"] == "_DenseLinear.Config"
    assert expanded["layers.1.ffn.__class__"] == "_DenseLinear.Config"
    assert expanded["layers.2.ffn.__class__"] == "_QuantLinear.Config"
    assert expanded["layers.3.ffn.__class__"] == "_MoEBlock.Config"
    assert expanded["layers.4.ffn.__class__"] == "_MoEBlock.Config"
    assert expanded["layers.3.ffn.num_experts"] == 8
    assert "layers.2.ffn.num_experts" not in expanded


def test_package_import_patches_config_loader_print_config(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        ConfigLoader().load(
            [
                "--module",
                "torchtitan_npu_recipes.tests.models.deepseek_v4",
                "--config",
                "deepseek_v4_debugmodel",
                "--print-config",
            ]
        )

    assert exc_info.value.code == 0
    stdout = capsys.readouterr().out
    assert " at 0x" not in stdout
    folded = json.loads(stdout)
    expanded = _expand_folded(folded)

    ref_cfg = deepseek_v4_debugmodel()
    assert ref_cfg.model is not None
    layers = ref_cfg.model.layers
    assert len(folded) < len(expanded)
    assert expanded["__class__"] == TrainerEx.Config.__qualname__
    for idx, layer in enumerate(layers):
        assert expanded[f"model.layers.{idx}.__class__"] == type(layer).__qualname__


def test_package_import_patches_trainer_maybe_log_and_keeps_full_saved_json(tmp_path, caplog, monkeypatch) -> None:
    cfg = deepseek_v4_debugmodel()
    cfg.dump_folder = str(tmp_path)
    cfg.debug.print_config = True
    cfg.debug.save_config_file = "config.json"

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)

    target_logger = torchtitan.trainer.logger.name
    with caplog.at_level(logging.INFO, logger=target_logger):
        cfg.maybe_log()

    marker = "Running with configs: "
    log_msg = next(rec.getMessage() for rec in caplog.records if marker in rec.getMessage())
    assert " at 0x" not in log_msg
    logged = json.loads(log_msg.split(marker, 1)[1])
    expanded = _expand_folded(logged)

    assert cfg.model is not None
    layers = cfg.model.layers
    assert len(logged) < len(expanded)
    assert expanded["__class__"] == type(cfg).__qualname__
    for idx, layer in enumerate(layers):
        assert expanded[f"model.layers.{idx}.__class__"] == type(layer).__qualname__

    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert isinstance(saved["model"]["layers"], list)
    assert len(saved["model"]["layers"]) == len(layers)
