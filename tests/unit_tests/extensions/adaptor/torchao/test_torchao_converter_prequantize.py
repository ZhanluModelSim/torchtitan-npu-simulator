# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU UT for the FSDP pre-quantize whitelist policy in the TorchAO-NPU converter.

Covers the explicitness contract from the pre-quantize design: the whitelist
(default or user-provided) decides per-node which weights keep
``fsdp_prequantize=True``, and a whitelist pattern matching nothing fails fast
instead of being silently ignored. Shard-alignment validation is a runtime
concern (the wrapper tensor's fail-fast guard), not a config-stage one.
"""

import pytest
from torchtitan.models.common.linear import Linear

# TorchAO-NPU is an optional dependency (see adaptor converter's
# module docstring): environments without it (e.g. the CI UT runner) must skip
# this module at collection instead of failing the import below, mirroring the
# importorskip in the converter_module fixture of test_converter.py.
pytest.importorskip("torchao_npu")

from torchtitan_npu.config.configs import QuantizationExtensionConfig
from torchtitan_npu.extensions.adaptor.torchao.converter import (
    _PREQUANTIZE_DEFAULT_FQNS,
    NpuQuantizedAscGroupedExpertsModule,
    _match_fqn_suffix,
    apply_quantization_converter,
)
from torchtitan_npu.models.deepseek_v4 import model_registry


def _quantization_config(**overrides) -> QuantizationExtensionConfig:
    return QuantizationExtensionConfig(enable_quantized_training=True, recipe="all_block_fp8", **overrides)


def _collect_flags(model_config) -> dict[str, bool]:
    """Collect {fqn: fsdp_prequantize} for every quantized Linear node."""
    flags: dict[str, bool] = {}
    for fqn, config, _, _ in model_config.traverse(Linear.Config):
        ao_config = getattr(config, "_torchao_npu_config", None)
        weight_config = getattr(ao_config, "weight_config", None)
        if weight_config is not None and hasattr(weight_config, "fsdp_prequantize"):
            flags[fqn] = weight_config.fsdp_prequantize
    return flags


def test_default_whitelist_keeps_prequantize_on_all_dense_projections():
    """fsdp_prequantize_fqns=None resolves to the recipe default whitelist."""
    spec = model_registry("debugmodel", num_mtp_layers=0)
    quantization = _quantization_config(enable_fsdp_prequantize=True)

    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    flags = _collect_flags(converted.model)
    assert flags, "debugmodel must expose quantized Linear nodes"
    assert all(flags.values()), f"default whitelist must enable every dense weight: {flags}"


def test_user_whitelist_disables_non_whitelisted_weights():
    """Only whitelisted FQNs keep fsdp_prequantize=True; the rest are explicitly off."""
    spec = model_registry("debugmodel", num_mtp_layers=0)
    quantization = _quantization_config(
        enable_fsdp_prequantize=True,
        fsdp_prequantize_fqns=[".attention.wq_a"],
    )

    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    flags = _collect_flags(converted.model)
    wq_a = [flag for fqn, flag in flags.items() if fqn.endswith(".attention.wq_a")]
    wkv = [flag for fqn, flag in flags.items() if fqn.endswith(".attention.wkv")]
    assert wq_a and all(wq_a), "whitelisted wq_a must stay enabled"
    assert wkv and not any(wkv), "non-whitelisted wkv must be explicitly disabled"


def test_whitelist_pattern_matching_nothing_raises():
    """A typo'd pattern is a silent no-op and must fail fast."""
    spec = model_registry("debugmodel", num_mtp_layers=0)
    quantization = _quantization_config(
        enable_fsdp_prequantize=True,
        fsdp_prequantize_fqns=[".attention.wq_a", ".attention.wk_b"],  # wk_b does not exist
    )

    with pytest.raises(ValueError, match="matched no quantized weight"):
        apply_quantization_converter(spec, quantization, model_compile_enabled=False)


def test_explicit_empty_whitelist_with_master_switch_raises():
    """An explicitly empty whitelist contradicts enable_fsdp_prequantize=True."""
    spec = model_registry("debugmodel", num_mtp_layers=0)
    quantization = _quantization_config(enable_fsdp_prequantize=True, fsdp_prequantize_fqns=[])

    with pytest.raises(ValueError, match="explicitly empty"):
        apply_quantization_converter(spec, quantization, model_compile_enabled=False)


def test_master_switch_off_leaves_policy_inert():
    """Without enable_fsdp_prequantize no whitelist policy is applied at all."""
    spec = model_registry("debugmodel", num_mtp_layers=0)
    quantization = _quantization_config(enable_fsdp_prequantize=False)

    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    flags = _collect_flags(converted.model)
    assert flags and not any(flags.values()), "disabled master switch must keep every weight off pre-quantize"


def test_default_whitelist_covers_every_quantized_projection():
    """The default whitelist must stay in sync with the recipe's quantized set.

    Every quantized Linear the converter produces on the real debugmodel tree
    matches a default entry, so a renamed or removed projection surfaces here
    (or as a zero-match error) instead of silently dropping pre-quantize
    coverage. Matching the config tree — not a literal copy of the constant —
    is what keeps the whitelist and the recipe in sync.
    """
    spec = model_registry("debugmodel", num_mtp_layers=0)
    quantization = _quantization_config(enable_fsdp_prequantize=True)
    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    all_patterns = tuple(pattern for scope in _PREQUANTIZE_DEFAULT_FQNS.values() for pattern in scope)
    flags = _collect_flags(converted.model)
    assert flags, "debugmodel must expose quantized Linear nodes"
    for fqn in flags:
        assert any(_match_fqn_suffix(fqn, pattern) for pattern in all_patterns), (
            f"quantized weight {fqn!r} matches no default whitelist entry; update _PREQUANTIZE_DEFAULT_FQNS"
        )


def test_mxfp4_communication_keeps_dense_fp8_and_experts_mxfp4():
    spec = model_registry("debugmodel", num_mtp_layers=0)
    config = _quantization_config(
        enable_mxfp4_qat=True,
        enable_fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
    )
    converted = apply_quantization_converter(spec, config, model_compile_enabled=False)
    dense_weights = [
        node._torchao_npu_config.weight_config
        for _, node, _, _ in converted.model.traverse(Linear.Config)
        if getattr(node, "_torchao_npu_config", None) is not None
    ]
    assert dense_weights
    for weight in dense_weights:
        assert weight.fsdp_prequantize
        assert weight.fsdp_prequantize_dtype == "fp8"
        assert weight.mxfp4_fake_quantize_config is None
    experts = list(converted.model.traverse(NpuQuantizedAscGroupedExpertsModule.Config))
    assert experts
    for _, node, _, _ in experts:
        weight = node._torchao_npu_config.weight_config
        assert weight.fsdp_prequantize
        assert weight.fsdp_prequantize_dtype == "mxfp4"
        assert weight.mxfp4_fake_quantize_config is not None


def test_mxfp4_communication_config_reaches_only_selected_experts():
    spec = model_registry("debugmodel", num_mtp_layers=0)
    config = _quantization_config(
        enable_mxfp4_qat=True,
        enable_fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        fsdp_prequantize_fqns=[".moe.routed_experts.inner_experts"],
    )
    converted = apply_quantization_converter(spec, config, model_compile_enabled=False)
    experts = list(converted.model.traverse(NpuQuantizedAscGroupedExpertsModule.Config))
    assert experts
    for _, node, _, _ in experts:
        weight = node._torchao_npu_config.weight_config
        assert weight.fsdp_prequantize
        assert weight.fsdp_prequantize_dtype == "mxfp4"
        assert weight.mxfp4_fake_quantize_config is not None
    assert not any(_collect_flags(converted.model).values())
