# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU coverage for converter selection and weight-wrapper dispatch to NPU ops."""

import importlib
import logging
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.fixture
def converter_module():
    # TorchAO-NPU is an optional dependency of the training plugin.
    pytest.importorskip("torchao_npu")
    importlib.import_module("torchtitan_npu")
    return importlib.import_module("torchtitan_npu.extensions.adaptor.torchao.converter")


@pytest.fixture(autouse=True)
def restore_quantized_module_cache(converter_module):
    from torchtitan_npu.extensions.adaptor.torchao.converter import _npu_quantized_module_cache

    original = _npu_quantized_module_cache.copy()
    yield
    _npu_quantized_module_cache.clear()
    _npu_quantized_module_cache.update(original)


@pytest.fixture
def quantization_calls(monkeypatch, converter_module):
    calls = []

    def quantize(module, config, *, filter_fn):
        calls.append((module, config, filter_fn))

    # Patch where the call sites live (generated modules in recipes + the DeepSeek
    # quantized-expert class in configs.deepseek_v4), not the facade, so the patched
    # quantize_ is what NpuQuantized*Module.__init__ resolves.
    for mod in ("torchtitan_npu.extensions.adaptor.torchao.recipes",
                "torchtitan_npu.extensions.adaptor.torchao.configs.deepseek_v4"):
        monkeypatch.setattr(importlib.import_module(mod), "quantize_", quantize)
    monkeypatch.setitem(sys.modules, "cann_ops_nn.ops", ModuleType("cann_ops_nn.ops"))
    with torch.random.fork_rng(devices=[]):
        yield calls


@pytest.mark.parametrize("source_kind", ["grouped", "asc"])
def test_quantized_grouped_experts_keep_config_and_skip_reconversion(converter_module, quantization_calls, source_kind):
    from torchtitan_npu.extensions.adaptor.torchao.converter import mxfp8_param_swap

    source_cls = converter_module.GroupedExperts if source_kind == "grouped" else converter_module.AscGroupedExperts
    source = source_cls.Config(dim=2, hidden_dim=4, num_experts=2, swiglu_limit=1.5)
    quant_config = mxfp8_param_swap()
    converter = converter_module.NpuQuantizeConverter.Config(
        base_config=quant_config,
        require_match=False,
        unified_npu_quant_modules=converter_module.UNIFIED_NPU_QUANTIZED_MODULES,
        specific_npu_quant_modules=converter_module.SPECIFIC_NPU_QUANTIZED_MODULES,
    ).build()

    converted = converter.convert(source)

    assert isinstance(converted, converter_module.NpuQuantizedAscGroupedExpertsModule.Config)
    assert (converted.dim, converted.hidden_dim, converted.num_experts, converted.swiglu_limit) == (2, 4, 2, 1.5)
    assert converter.convert(converted) is converted
    swiglu_module = importlib.import_module("torchtitan_npu.override.common.swiglu_group")
    assert swiglu_module.asc(converted) is converted
    experts = converted.build()
    assert len(quantization_calls) == 1
    module, received_config, filter_fn = quantization_calls[0]
    assert module is experts
    assert received_config is quant_config
    assert filter_fn(experts, "")
    assert not filter_fn(torch.nn.Identity(), "child")


def test_quantization_converter_preserves_custom_experts(converter_module, quantization_calls):
    from torchtitan_npu.extensions.adaptor.torchao.converter import mxfp8_param_swap

    class CustomExperts(converter_module.GroupedExperts):
        @dataclass(kw_only=True, slots=True)
        class Config(converter_module.GroupedExperts.Config):
            output_scale: float = 3.0

        def __init__(self, config):
            super().__init__(config)
            self.output_scale = config.output_scale

        def forward(self, x, num_tokens_per_expert, *, routed_scores_R=None):
            return x * self.output_scale

    source = CustomExperts.Config(dim=2, hidden_dim=4, num_experts=2, output_scale=5.0)
    converter = converter_module.NpuQuantizeConverter.Config(
        base_config=mxfp8_param_swap(),
        require_match=False,
        unified_npu_quant_modules=converter_module.UNIFIED_NPU_QUANTIZED_MODULES,
        specific_npu_quant_modules=converter_module.SPECIFIC_NPU_QUANTIZED_MODULES,
    ).build()

    converted = converter.convert(source)

    assert isinstance(converted, CustomExperts.Config)
    assert converted.output_scale == 5.0
    assert converter.convert(converted) is converted
    experts = converted.build()
    assert quantization_calls[0][0] is experts
    torch.testing.assert_close(experts(torch.tensor([[1.0, 2.0]]), torch.tensor([1, 0])), torch.tensor([[5.0, 10.0]]))


def test_quantization_converter_keeps_linear_forward(converter_module, quantization_calls):
    from torchtitan_npu.extensions.adaptor.torchao.converter import mxfp8_param_swap

    source = converter_module.Linear.Config(in_features=2, out_features=2, bias=False)
    quant_config = mxfp8_param_swap()
    converter = converter_module.NpuQuantizeConverter.Config(
        base_config=quant_config,
        require_match=False,
        unified_npu_quant_modules=converter_module.UNIFIED_NPU_QUANTIZED_MODULES,
        specific_npu_quant_modules=converter_module.SPECIFIC_NPU_QUANTIZED_MODULES,
    ).build()

    converted = converter.convert(source)
    assert converter.convert(converted) is converted
    linear = converted.build()

    assert isinstance(linear, converter_module.Linear)
    assert not isinstance(linear, converter_module.NpuQuantizedAscGroupedExpertsModule)
    assert quantization_calls[0][0] is linear
    assert quantization_calls[0][1] is quant_config
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    torch.testing.assert_close(linear(torch.tensor([[2.0, 3.0]])), torch.tensor([[8.0, 18.0]]))


def _v41_attention():
    """One layer's own numbers: the five widths, and the pool its selector declares."""
    return SimpleNamespace(
        n_heads=64,
        head_dim=512,
        indexer=SimpleNamespace(
            index_topk=512,
            num_index_heads=32,
            index_head_dim=128,
            selector=SimpleNamespace(candidate_topk_blocks=2048, candidate_block_size=8),
        ),
    )


@pytest.mark.parametrize(
    ("model_type", "expect_sparse_attention_converter"),
    [("v4", False), ("v41", True)],
)
def test_recipe_converters_select_filters_for_model_type(
    converter_module, model_type, expect_sparse_attention_converter
):
    converters = converter_module._recipe_converters(
        "mix",
        model_type=converter_module.ModelType(model_type),
        enable_sparse_attention_quantization=True,
        enable_mxfp4_qat=False,
        dst_type_max=0.0,
        enable_fsdp_prequantize=False,
        model_compile_enabled=False,
        # The sparse-attention replacement is sized from the stack's single geometry.
        model_config=SimpleNamespace(
            layers=[SimpleNamespace(attention=_v41_attention())],
            candidate_topk_blocks=2048,
            candidate_block_size=8,
        ),
    )

    sparse_attention_converters = [
        converter
        for converter in converters
        if converter.filter_fn is converter_module._DSV41_CONFIG_FILTERS["sparse_attention"]
    ]
    assert bool(sparse_attention_converters) is expect_sparse_attention_converter


def test_qwen35_w4a8_mxfp_recipe_quantizes_only_dense_modules(converter_module):
    converters = converter_module._recipe_converters(
        "W4A8_MXFP",
        model_type=converter_module.ModelType.QWEN3_5,
        enable_mxfp4_qat=False,
        dst_type_max=7.0,
        enable_fsdp_prequantize=False,
        model_compile_enabled=False,
    )

    assert len(converters) == 1
    converter = converters[0]
    param_swap = converter.base_config
    assert isinstance(param_swap.activation_config, converter_module.MXQuantizeConfig)
    assert isinstance(param_swap.weight_config, converter_module.BlockMXQuantizeConfig)
    mxfp4_config = param_swap.weight_config.mxfp4_fake_quantize_config
    assert isinstance(mxfp4_config, converter_module.MXQuantizeConfig)
    assert mxfp4_config.elem_dtype is torch.float4_e2m1fn_x2
    assert mxfp4_config.dst_type_max == 7.0
    assert converter.filter_fn(SimpleNamespace(), "layers.0.attention.wq")
    assert converter.filter_fn(SimpleNamespace(), "layers.0.delta_net.in_proj_q")
    assert converter.filter_fn(SimpleNamespace(), "layers.0.feed_forward.w1")
    assert not converter.filter_fn(SimpleNamespace(), "layers.0.moe.routed_experts.inner_experts")
    assert not converter.filter_fn(SimpleNamespace(), "layers.0.moe.shared_experts.w1")
    assert not converter.filter_fn(SimpleNamespace(), "layers.0.moe.router")
    assert not converter.filter_fn(SimpleNamespace(), "lm_head")


def test_qwen35_w8a8_mxfp8_recipe_quantizes_only_dense_modules(converter_module):
    converters = converter_module._recipe_converters(
        "W8A8_MXFP8",
        model_type=converter_module.ModelType.QWEN3_5,
        enable_mxfp4_qat=True,
        dst_type_max=7.0,
        enable_fsdp_prequantize=False,
        model_compile_enabled=False,
    )

    assert len(converters) == 1
    converter = converters[0]
    param_swap = converter.base_config
    assert isinstance(param_swap.activation_config, converter_module.MXQuantizeConfig)
    assert isinstance(param_swap.weight_config, converter_module.BlockMXQuantizeConfig)
    assert param_swap.weight_config.mxfp4_fake_quantize_config is None
    assert converter.filter_fn(SimpleNamespace(), "layers.0.attention.wq")
    assert converter.filter_fn(SimpleNamespace(), "layers.0.delta_net.in_proj_q")
    assert converter.filter_fn(SimpleNamespace(), "layers.0.feed_forward.w1")
    assert not converter.filter_fn(SimpleNamespace(), "layers.0.moe.routed_experts.inner_experts")
    assert not converter.filter_fn(SimpleNamespace(), "layers.0.moe.shared_experts.w1")
    assert not converter.filter_fn(SimpleNamespace(), "layers.0.moe.router")
    assert not converter.filter_fn(SimpleNamespace(), "lm_head")


@pytest.mark.parametrize("recipe", ["W4A8_MXFP", "W8A8_MXFP8"])
@pytest.mark.parametrize(
    "fqn",
    [
        "vision_encoder.block.attn.wq",
        "vision_encoder.block.attn.wk",
        "vision_encoder.block.attn.wv",
        "vision_encoder.block.attn.proj",
        "vision_encoder.block.mlp.fc1",
        "vision_encoder.block.mlp.fc2",
        "vision_encoder.merger.fc1",
        "vision_encoder.merger.fc2",
    ],
)
def test_qwen35_dense_recipes_exclude_vision_encoder(converter_module, recipe, fqn):
    converters = converter_module._recipe_converters(
        recipe,
        model_type=converter_module.ModelType.QWEN3_5,
        enable_mxfp4_qat=False,
        dst_type_max=7.0,
        enable_fsdp_prequantize=False,
        model_compile_enabled=False,
    )

    assert len(converters) == 1
    assert not converters[0].filter_fn(SimpleNamespace(), fqn)


@pytest.mark.parametrize("recipe", ["W4A8_MXFP", "W8A8_MXFP8"])
def test_qwen35_dense_prequantize_rejects_mxfp4_payload(converter_module, recipe):
    with pytest.raises(ValueError, match="supports only FP8 FSDP-prequantized dense weights"):
        converter_module._recipe_converters(
            recipe,
            model_type=converter_module.ModelType.QWEN3_5,
            enable_mxfp4_qat=False,
            dst_type_max=7.0,
            enable_fsdp_prequantize=True,
            fsdp_prequantize_dtype="mxfp4",
            model_compile_enabled=False,
        )


@pytest.mark.parametrize("model_type", ["v4", "v41"])
@pytest.mark.parametrize("recipe", ["W4A8_MXFP", "W8A8_MXFP8"])
def test_qwen35_dense_recipes_are_rejected_for_deepseek(converter_module, model_type, recipe):
    with pytest.raises(ValueError, match=rf"recipe for model '{model_type}'.*{recipe}"):
        converter_module._recipe_converters(
            recipe,
            model_type=converter_module.ModelType(model_type),
            enable_mxfp4_qat=False,
            dst_type_max=7.0,
            enable_fsdp_prequantize=False,
            model_compile_enabled=False,
        )


def test_v41_li_quantization_alone_is_refused(converter_module):
    """The quantized selector launches the quantized sparse attention's schedules."""
    with pytest.raises(ValueError, match=r"requires .*enable-sparse-attention-quantization"):
        converter_module._recipe_converters(
            "mix",
            model_type="v41",
            enable_sparse_attention_quantization=False,
            enable_mxfp4_qat=False,
            dst_type_max=0.0,
            enable_fsdp_prequantize=False,
            model_compile_enabled=False,
            li_quantization="mxfp4",
        )


def test_sparse_attention_component_triggers_on_real_v41_filters(converter_module, caplog):
    """On the real V4.1 filter set the sparse_attention component must trigger.

    V4.1 always registers a ``sparse_attention`` filter, so ``should_trigger``
    returns True and never warns. (Passing an empty filter dict -- the only way
    to reach the warning branch -- is not a real V4.1 model state.)
    """
    from torchtitan_npu.extensions.adaptor.torchao.components import instantiate_component_for_model_and_name
    from torchtitan_npu.extensions.adaptor.torchao.recipes import RecipeContext

    component = instantiate_component_for_model_and_name(converter_module.ModelType.DSV4_1, "sparse_attention")
    ctx = RecipeContext(
        model_type=converter_module.ModelType.DSV4_1,
        model_compile_enabled=False,
        enable_sparse_attention_quantization=True,
        fsdp_prequantize_dtype="fp8",
    )

    with caplog.at_level(logging.WARNING):
        triggered = component.should_trigger(converter_module._DSV41_CONFIG_FILTERS, ctx)

    assert triggered is True
    assert "has no sparse_attention filter" not in caplog.text


def test_model_type_rejects_unknown_model_spec_name(converter_module):
    with pytest.raises(ValueError, match=r"supports DeepSeek V4, DeepSeek V4\.1 and Qwen3\.5 model specs"):
        converter_module._model_type_for_spec(SimpleNamespace(name="deepseek_v3", model=object()))


def test_model_type_comes_from_model_spec_name(converter_module):
    kinds = {
        "deepseek_v4": converter_module.ModelType.DSV4,
        "deepseek_v4_1": converter_module.ModelType.DSV4_1,
        "qwen3_5": converter_module.ModelType.QWEN3_5,
    }
    for spec_name, expected in kinds.items():
        assert converter_module._model_type_for_spec(SimpleNamespace(name=spec_name, model=object())) is expected


def test_family_configs_are_split_and_registered(converter_module):
    from torchtitan_npu.extensions.adaptor.torchao.components import (
        component_names_for_model,
        instantiate_component_for_model_and_name,
    )
    from torchtitan_npu.extensions.adaptor.torchao.configs import get_model_config

    v4 = get_model_config(converter_module.ModelType.DSV4)
    v41 = get_model_config(converter_module.ModelType.DSV4_1)
    qwen35 = get_model_config(converter_module.ModelType.QWEN3_5)

    # Component ownership now comes from the components registry, not configs.
    assert set(component_names_for_model(converter_module.ModelType.DSV4)) == {
        "lightning_indexer",
        "v4_sparse_attention",
        "kv_norm",
    }
    assert set(component_names_for_model(converter_module.ModelType.DSV4_1)) == {
        "lightning_indexer",
        "compressor",
        "sparse_attention",
    }
    assert component_names_for_model(converter_module.ModelType.QWEN3_5) == ()
    assert tuple(qwen35.recipe_matrix) == ("W4A8_MXFP", "W8A8_MXFP8")
    assert tuple(qwen35.filters) == ("dense_model",)

    # Order is deterministic: component_names_for_model must preserve registration
    # order (a set would randomize across processes and break reproducible
    # converter ordering).
    assert component_names_for_model(converter_module.ModelType.DSV4) == (
        "lightning_indexer",
        "v4_sparse_attention",
        "kv_norm",
    )
    assert component_names_for_model(converter_module.ModelType.DSV4_1) == (
        "lightning_indexer",
        "compressor",
        "sparse_attention",
    )

    # The lightning_indexer suffix differs per family; sparse_attention is V4.1-only.
    assert converter_module._DSV4_CONFIG_FILTERS["lightning_indexer"] is not converter_module._DSV41_CONFIG_FILTERS[
        "lightning_indexer"
    ]
    assert "sparse_attention" not in converter_module._DSV4_CONFIG_FILTERS
    assert "sparse_attention" in converter_module._DSV41_CONFIG_FILTERS

    # V4.1 owns its own (distinct) recipe matrix object.
    assert v41.recipe_matrix is not v4.recipe_matrix

    # Per-model components resolve under their feature name for each family.
    assert (
        type(
            instantiate_component_for_model_and_name(converter_module.ModelType.DSV4_1, "sparse_attention")
        ).__name__
        == "DeepSeekV41QuantSparseAttentionComponent"
    )
    assert (
        type(
            instantiate_component_for_model_and_name(converter_module.ModelType.DSV4_1, "compressor")
        ).__name__
        == "DeepSeekV41QuantCompressorComponent"
    )
    assert (
        type(instantiate_component_for_model_and_name(converter_module.ModelType.DSV4, "kv_norm")).__name__
        == "KVNormComponent"
    )
    # lightning_indexer is split per model: V4 vs V4.1 have their own class.
    assert (
        type(instantiate_component_for_model_and_name(converter_module.ModelType.DSV4, "lightning_indexer")).__name__
        == "DeepSeekV4QuantLightningIndexerComponent"
    )
    assert (
        type(instantiate_component_for_model_and_name(converter_module.ModelType.DSV4_1, "lightning_indexer")).__name__
        == "DeepSeekV41QuantLightningIndexerComponent"
    )
    assert (
        type(instantiate_component_for_model_and_name(converter_module.ModelType.DSV4, "v4_sparse_attention")).__name__
        == "DeepSeekV4QuantSparseAttentionComponent"
    )


def test_recipe_converters_reject_unknown_model_type(converter_module):
    with pytest.raises(ValueError, match="unsupported quantized model type"):
        converter_module._recipe_converters(
            "mix",
            model_type="v3",  # type: ignore[arg-type]
            enable_mxfp4_qat=False,
            dst_type_max=0.0,
            enable_fsdp_prequantize=False,
            model_compile_enabled=False,
        )


@pytest.mark.parametrize(
    ("model_name", "expected_model_type"),
    [("deepseek_v4", "v4"), ("deepseek_v4_1", "v41"), ("qwen3_5", "qwen3_5")],
)
def test_apply_quantization_converter_passes_model_type(converter_module, monkeypatch, model_name, expected_model_type):
    from torchtitan_npu.config.configs import QuantizationExtensionConfig

    @dataclass(frozen=True)
    class FakeModelSpec:
        name: str
        model: object

    quantization_config = QuantizationExtensionConfig(enable_quantized_training=True, recipe="mix")
    calls = []
    monkeypatch.setattr(converter_module, "_recipe_converters", lambda *args, **kwargs: calls.append(kwargs) or [])
    monkeypatch.setattr(converter_module, "validate_converter_order", lambda converters: None)

    model_spec = FakeModelSpec(name=model_name, model=object())
    converted = converter_module.apply_quantization_converter(
        model_spec,
        quantization_config,
        model_compile_enabled=False,
    )

    assert converted is not model_spec
    assert calls[0]["model_type"] == converter_module.ModelType(expected_model_type)


@pytest.mark.parametrize("quant_format", ["mx", "block_mx"])
def test_quantized_experts_preserve_wrappers_through_forward(monkeypatch, converter_module, quant_format):
    from torchtitan_npu.extensions.adaptor.torchao.converter import block_fp8_param_swap, mxfp8_param_swap

    wrapper_module = importlib.import_module(f"torchao_npu.wrapper_tensors.{quant_format}_wrapper_tensor")
    if quant_format == "mx":
        quant_config = mxfp8_param_swap()
        wrapper_cls = wrapper_module.MXTrainingWeightWrapperTensor
    else:
        quant_config = block_fp8_param_swap()
        wrapper_cls = wrapper_module.BlockMXTrainingWeightWrapperTensor
    monkeypatch.setitem(sys.modules, "cann_ops_nn.ops", ModuleType("cann_ops_nn.ops"))
    source = converter_module.GroupedExperts.Config(dim=32, hidden_dim=32, num_experts=2)
    converter = converter_module.NpuQuantizeConverter.Config(
        base_config=quant_config,
        unified_npu_quant_modules=converter_module.UNIFIED_NPU_QUANTIZED_MODULES,
        specific_npu_quant_modules=converter_module.SPECIFIC_NPU_QUANTIZED_MODULES,
    ).build()

    # Keep real quantize_, parameter wrappers, casts and torch._grouped_mm
    # dispatch. Replace only the NPU grouped-matmul and SwiGLU boundaries.
    with torch.random.fork_rng(devices=[]):
        experts = converter.convert(source).build()
    for weight in (experts.w1_EFD, experts.w3_EFD, experts.w2_EDF):
        assert isinstance(weight, wrapper_cls)
        assert isinstance(weight.bfloat16().transpose(-2, -1), wrapper_cls)

    matmul_calls = []

    def quantized_grouped_mm(x, weight, offsets, activation_config, weight_config):
        matmul_calls.append((offsets, activation_config, weight_config))
        return torch.cat((x[:1] @ weight[0], x[1:] @ weight[1]))

    monkeypatch.setattr(wrapper_module, f"to_{quant_format}_then_grouped_mm", quantized_grouped_mm)
    swiglu_calls = []

    def swiglu(x, *, weight, group_index, clamp_limit):
        swiglu_calls.append((x, weight, group_index, clamp_limit))
        # This fixed boundary contract isolates dispatch from CANN numerics.
        return (x[..., :32].float() * weight.reshape(-1, 1)).to(x.dtype)

    monkeypatch.setattr(torch.ops, "cann_ops_nn", SimpleNamespace(swiglu_group=SimpleNamespace(default=swiglu)))
    ascendc = importlib.import_module("torchtitan_npu.override.common.swiglu_group.ascendc")
    monkeypatch.setattr(ascendc, "get_spmd_backend", lambda: "torch")
    with torch.no_grad():
        identity = torch.eye(32)
        experts.w1_EFD.copy_(torch.stack((identity, identity * 2)))
        experts.w3_EFD.copy_(torch.stack((identity * 0.5, identity * 1.5)))
        experts.w2_EDF.copy_(torch.stack((identity * 3, identity * 4)))
    scores = torch.tensor([0.25, 0.5, 0.75])

    output = experts(torch.full((3, 32), 0.25), torch.tensor([1, 2]), routed_scores_R=scores)

    torch.testing.assert_close(output, torch.tensor([[0.1875], [1.0], [1.5]]).expand(3, 32))
    assert len(matmul_calls) == 3
    for offsets, activation_config, weight_config in matmul_calls:
        torch.testing.assert_close(offsets, torch.tensor([1, 3], dtype=torch.int32))
        assert activation_config is quant_config.activation_config
        assert weight_config is quant_config.weight_config
    assert len(swiglu_calls) == 1
    packed, weight, group_index, clamp_limit = swiglu_calls[0]
    torch.testing.assert_close(packed[:, :32], torch.tensor([[0.25], [0.5], [0.5]], dtype=torch.bfloat16).expand(3, 32))
    torch.testing.assert_close(
        packed[:, 32:], torch.tensor([[0.125], [0.375], [0.375]], dtype=torch.bfloat16).expand(3, 32)
    )
    torch.testing.assert_close(weight, scores)
    assert group_index is None
    assert clamp_limit == -1.0


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
def test_v41_recipe_couples_source_compressor_and_sparse_attention(converter_module, enabled):
    from torchao_npu.configs import DeepSeekV41QuantCompressorConfig, DeepSeekV41QuantSparseAttentionConfig

    from torchtitan_npu.config.configs import QuantizationExtensionConfig
    from torchtitan_npu.models.deepseek_v4_1 import model_registry

    spec = model_registry("deepseek_v4_1_debugmodel")
    converted = converter_module.apply_quantization_converter(
        spec,
        QuantizationExtensionConfig(
            enable_quantized_training=True,
            enable_sparse_attention_quantization=enabled,
            recipe="all_mxfp8",
        ),
        model_compile_enabled=False,
    )

    for layer in converted.model.layers:
        compressor = layer.attention.compressor
        compressor_quant = getattr(compressor, "_torchao_npu_config", None)
        attention_quant = getattr(layer.attention.inner_attention, "_torchao_npu_config", None)
        if enabled and compressor.is_source:
            assert isinstance(compressor_quant, DeepSeekV41QuantCompressorConfig)
            assert compressor_quant.group_size == 16
            assert type(compressor)._owner.__name__ == "NpuQuantizedCompressor"
        else:
            assert compressor_quant is None
        assert isinstance(attention_quant, DeepSeekV41QuantSparseAttentionConfig) is enabled


def _source_compressor_layer(spec, ratio):
    return next(
        i
        for i, layer in enumerate(spec.model.layers)
        if layer.attention.compressor.is_source and layer.attention.compressor.compress_ratio == ratio
    )


def _build_cpu_compressor(config):
    module = config.build()
    module.init_states(buffer_device=torch.device("cpu"))
    return module.to(dtype=torch.bfloat16)


def _capture_compressor_encoding(monkeypatch, width):
    captured = []

    def encode(cache, rows, slots, **kwargs):
        captured.append(rows.detach().clone())
        assert kwargs["quant_group_size"] == 16
        assert kwargs["quant_mode"] == "mxfp4_bf16"
        cache[:, : width // 2] = 0x21
        scales = torch.full((rows.shape[0], width // 16), 0.5, dtype=torch.bfloat16)
        cache[:, width // 2 : width // 2 + 2 * (width // 16)] = scales.view(torch.uint8)

    monkeypatch.setattr(torch.ops.custom, "kv_compress_epilog_v2", SimpleNamespace(default=encode), raising=False)
    return captured


@pytest.mark.parametrize("li_enabled, sparse_enabled", [(False, False), (False, True), (True, True)])
def test_v41_teacher_pair_survives_each_supported_switch_combination(
    converter_module,
    li_enabled,
    sparse_enabled,
):
    """Every combination a V4.1 recipe may state keeps both sides of the teacher edge intact.

    ``li_enabled`` without the sparse-attention switch is not among them: the quantized selector
    launches the quantized attention's schedules, so the converter refuses that pairing outright.
    """
    from torchao_npu.configs import DeepSeekV41QuantLightningIndexerConfig, DeepSeekV41QuantSparseAttentionConfig
    from torchtitan.config.override import OverrideConfig, apply_overrides

    from torchtitan_npu.config.configs import QuantizationExtensionConfig
    from torchtitan_npu.models.deepseek_v4_1 import model_registry

    converted = converter_module.apply_quantization_converter(
        model_registry("deepseek_v4_1_debugmodel"),
        QuantizationExtensionConfig(
            enable_quantized_training=True,
            enable_sparse_attention_quantization=sparse_enabled,
            li_quantization="mxfp4" if li_enabled else None,
            recipe="all_mxfp8",
        ),
        model_compile_enabled=False,
    )
    apply_overrides(
        OverrideConfig(
            imports=[
                "torchtitan_npu.override.deepseek_v4_1.lightning_indexer.asc",
                "torchtitan_npu.override.deepseek_v4_1.sparse_attn.asc",
            ]
        ),
        converted.model,
    )

    for layer in converted.model.layers:
        selector = layer.attention.indexer.selector
        inner_attention = layer.attention.inner_attention
        selector_quant = getattr(selector, "_torchao_npu_config", None)
        attention_quant = getattr(inner_attention, "_torchao_npu_config", None)

        assert isinstance(selector_quant, DeepSeekV41QuantLightningIndexerConfig) is li_enabled
        assert isinstance(attention_quant, DeepSeekV41QuantSparseAttentionConfig) is sparse_enabled
        if inner_attention.compress_ratio > 0:
            # The trainer discovers instances, not configs, to assign the step denominator.
            selector_module = selector.build()
            assert isinstance(selector_module, converter_module.DeepSeekV41Selector)
            assert selector_module.num_global_queries is None


@pytest.mark.parametrize("li_quantization", ["mxfp8", "fp8", "hif8"])
def test_v41_recipe_rejects_non_mxfp4_lightning_indexer(converter_module, li_quantization):
    with pytest.raises(ValueError, match="supports only li_quantization='mxfp4'"):
        converter_module._recipe_converters(
            "mix",
            model_type=converter_module.ModelType.DSV4_1,
            enable_sparse_attention_quantization=True,
            enable_mxfp4_qat=False,
            dst_type_max=0.0,
            enable_fsdp_prequantize=False,
            model_compile_enabled=False,
            li_quantization=li_quantization,
        )


@pytest.mark.parametrize("ratio", [1, 2], ids=["full-resolution", "compressed"])
def test_converted_v41_compressor_quantizes_post_rope_and_preserves_gradients(converter_module, monkeypatch, ratio):
    from torchtitan_npu.config.configs import QuantizationExtensionConfig
    from torchtitan_npu.models.deepseek_v4_1 import model_registry

    # The CPU oracle replaces the custom kernel below; no NPU extension is loaded.
    monkeypatch.setitem(sys.modules, "custom_ops", ModuleType("custom_ops"))
    reference_spec = model_registry("deepseek_v4_1_debugmodel")
    converted = converter_module.apply_quantization_converter(
        model_registry("deepseek_v4_1_debugmodel"),
        QuantizationExtensionConfig(
            enable_quantized_training=True,
            enable_sparse_attention_quantization=True,
            recipe="all_mxfp8",
        ),
        model_compile_enabled=False,
    )
    layer_id = _source_compressor_layer(reference_spec, ratio)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        reference = _build_cpu_compressor(reference_spec.model.layers[layer_id].attention.compressor)
        quantized = _build_cpu_compressor(converted.model.layers[layer_id].attention.compressor)
    quantized.load_state_dict(reference.state_dict())
    d = reference_spec.model.layers[layer_id].attention.compressor.wkv.out_features
    captured = _capture_compressor_encoding(monkeypatch, d)
    dim = reference_spec.model.dim
    x_ref = torch.linspace(-1, 1, 4 * dim, dtype=torch.bfloat16).reshape(1, 4, dim).requires_grad_()
    x_quant = x_ref.detach().clone().requires_grad_()
    positions = torch.arange(4).unsqueeze(0)

    expected_kv, expected_latent = reference(x_ref, positions)
    actual_kv, actual_latent = quantized(x_quant, positions)

    assert len(captured) == 1
    assert torch.equal(captured[0], expected_kv.detach().reshape(-1, d))
    assert torch.equal(actual_latent, expected_latent)
    expected_qdq = torch.tensor([0.25, 0.5], dtype=actual_kv.dtype).repeat(actual_kv.numel() // 2).reshape_as(actual_kv)
    assert torch.equal(actual_kv, expected_qdq)
    assert actual_kv.dtype == expected_kv.dtype
    reuser = converted.model.layers[layer_id + 1].attention.compressor.build()
    shared, latent = reuser(x_quant, positions, actual_kv)
    assert shared is actual_kv
    assert latent is None
    assert len(captured) == 1

    kv_grad = torch.linspace(-0.5, 0.5, actual_kv.numel()).reshape_as(actual_kv)
    latent_grad = torch.linspace(0.25, 0.75, actual_latent.numel()).reshape_as(actual_latent)
    ((expected_kv.float() * kv_grad).sum() + (expected_latent.float() * latent_grad).sum()).backward()
    ((actual_kv.float() * kv_grad).sum() + (actual_latent.float() * latent_grad).sum()).backward()
    torch.testing.assert_close(x_quant.grad, x_ref.grad, rtol=0, atol=0)
    for (_, parameter), (_, reference_parameter) in zip(
        quantized.named_parameters(), reference.named_parameters(), strict=True
    ):
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)


def test_v4_quantized_indexer_adapter_preserves_backward_record_and_frames(converter_module):
    from torchao_npu.quantized_modules.deepseek_v4.attention_metadata import QuantV4Metadata

    from torchtitan_npu.models.deepseek_v4.metadata import build_v4_metadata
    from torchtitan_npu.override.deepseek_v4.metadata import (
        AicpuGeometry,
        AicpuMetadata,
        AscLightningIndexerMetadata,
        AscSparseAttentionMetadata,
    )

    positions = torch.arange(8).unsqueeze(0)
    row = build_v4_metadata(positions, ratio=4, build_slots=False)
    forward_record = torch.tensor([7], dtype=torch.int32)
    backward_record = torch.tensor([9], dtype=torch.int32)
    metadata = QuantV4Metadata(
        cu_seq_q=row.cu_seq_q,
        cu_seq_k=row.cu_seq_k,
        ratio=row.ratio,
        position_k=row.position_k,
        q=row.q,
        swa_k=row.swa_k,
        cmp_k=row.cmp_k,
        aicpu_metadata=AicpuMetadata(
            geometry=AicpuGeometry(
                num_heads=2, head_dim=8, window_size=8, index_topk=4, index_num_heads=2, index_head_dim=4
            ),
            sparse_attn=AscSparseAttentionMetadata(),
            lightning_indexer=AscLightningIndexerMetadata(fwd=forward_record, bwd=backward_record),
        ),
    )
    query = torch.zeros(1, 8, 2, 4, dtype=torch.bfloat16)
    key = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    weights = torch.zeros(1, 8, 2, dtype=torch.bfloat16)

    inputs = converter_module._prepare_quant_lightning_indexer_inputs(
        SimpleNamespace(index_topk=4), query, key, weights, metadata
    )

    assert inputs.query.shape == (8, 2, 4)
    assert inputs.key.shape == (2, 1, 4)
    assert inputs.weights.shape == (8, 2) and inputs.weights.dtype == torch.float32
    assert inputs.metadata is forward_record and inputs.backward_metadata is backward_record
    assert inputs.seqused_q is metadata.q.seqused
    assert inputs.seqused_k is metadata.cmp_k.seqused
    assert inputs.cu_seqlens_q is row.cu_seq_q
    assert inputs.cu_seqlens_k is row.cmp_k.cu_seqlens
    assert inputs.cmp_residual_k is row.cmp_k.residual
