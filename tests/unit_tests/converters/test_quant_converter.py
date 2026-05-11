# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace

from torchtitan_npu.converters import quant_converter


@dataclass
class QuantizeCallSpy:
    model: object = None
    config: object = None
    filter_fn: object = None

    def record(self, model, config, filter_fn):
        self.model = model
        self.config = config
        self.filter_fn = filter_fn


def _patch_quant_runtime(monkeypatch, *, linear_recipe=None, grouped_recipe=None):
    monkeypatch.setattr(quant_converter, "is_a5", lambda: True)
    if linear_recipe is not None:
        monkeypatch.setattr(
            quant_converter.TorchMXLinearConfig,
            "from_recipe_name",
            staticmethod(lambda recipe_name: linear_recipe(recipe_name)),
        )
    if grouped_recipe is not None:
        monkeypatch.setattr(
            quant_converter.TorchMoETrainingConfig,
            "from_recipe_name",
            staticmethod(lambda recipe_name: grouped_recipe(recipe_name)),
        )


def _build_mxfp8_converter_config():
    return quant_converter.NPUMXFP8Config(
        recipe_name="mxfp8",
        filter_fqns=["attention"],
        fqns=["layers.0.moe"],
    )


def _build_parallel_dims():
    return SimpleNamespace(ep_enabled=False)


def test_quant_converter_replace():
    from torchtitan.components.quantization.mx import MXFP8Converter

    assert MXFP8Converter.Config is quant_converter.NPUMXFP8Config
    assert MXFP8Converter.__init__ == quant_converter.npu_quant_mxfp8_converter_init
    assert MXFP8Converter.convert == quant_converter.npu_quant_mxfp8_converter


def test_npu_quant_mxfp8_converter_init_sets_runtime_fields(monkeypatch):
    _patch_quant_runtime(
        monkeypatch,
        linear_recipe=lambda recipe_name: {"linear_recipe": recipe_name},
        grouped_recipe=lambda recipe_name: {"grouped_recipe": recipe_name},
    )

    converter = SimpleNamespace()
    config = _build_mxfp8_converter_config()

    quant_converter.npu_quant_mxfp8_converter_init(
        converter,
        config,
        parallel_dims=_build_parallel_dims(),
        model_compile_enabled=False,
    )

    assert converter.enabled is True
    assert converter.filter_fqns == ["attention"]
    assert converter.moe_fqns == ["layers.0.moe"]
    assert converter.linear_config == {"linear_recipe": "mxfp8"}
    assert converter.grouped_mm_config == {"grouped_recipe": "mxfp8"}


def test_npu_quant_mxfp8_converter_calls_linear_and_grouped_quantize(monkeypatch):
    _patch_quant_runtime(
        monkeypatch,
        linear_recipe=lambda recipe_name: {"linear_recipe": recipe_name},
        grouped_recipe=lambda recipe_name: {"grouped_recipe": recipe_name},
    )

    linear_spy = QuantizeCallSpy()
    grouped_spy = QuantizeCallSpy()
    monkeypatch.setattr(quant_converter, "linear_quantize_", linear_spy.record)
    monkeypatch.setattr(quant_converter, "grouped_quantize_", grouped_spy.record)

    converter = SimpleNamespace()
    config = _build_mxfp8_converter_config()
    quant_converter.npu_quant_mxfp8_converter_init(
        converter,
        config,
        parallel_dims=_build_parallel_dims(),
        model_compile_enabled=False,
    )

    model = object()
    quant_converter.npu_quant_mxfp8_converter(converter, model)

    assert linear_spy.model is model
    assert linear_spy.config == {"linear_recipe": "mxfp8"}
    assert linear_spy.filter_fn is not None

    assert grouped_spy.model is model
    assert grouped_spy.config == {"grouped_recipe": "mxfp8"}
    assert grouped_spy.filter_fn is not None
