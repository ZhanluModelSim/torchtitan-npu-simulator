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

import logging
import re

import pytest
from torchtitan.config.parallelism import ParallelismConfig

# TorchAO and TorchAO-NPU are optional dependencies (see the adaptor converter's
# module docstring): the adaptor's configs/recipe modules import them at module
# top, so environments without them (e.g. the CI UT runner) must skip this
# module at collection *before* the imports below, instead of failing the
# import.  Mirrors the importorskip in the converter_module fixture of
# test_converter.py (which defers the heavy import into a fixture).
pytest.importorskip("torchao")
pytest.importorskip("torchao_npu")

from torchtitan.models.base.module import Module
from torchtitan.models.common.linear import GroupedLinear, Linear

from torchtitan_npu.config.configs import QuantizationExtensionConfig
from torchtitan_npu.extensions.adaptor.torchao.configs import get_model_config
from torchtitan_npu.extensions.adaptor.torchao.converter import (
    _match_fqn_suffix,
    _model_type_for_config,
    apply_quantization_converter,
)
from torchtitan_npu.models.deepseek_v4 import build_model_config
from torchtitan_npu.models.deepseek_v4_1 import model_registry as v41_model_registry
from torchtitan_npu_recipes.models.deepseek_v4 import deepseek_v4_flash_cpt_4k_a5


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


# Stacked gate/up weight: FSDP shards it on dim 1, where the multi-component
# Block MX payload cannot be reassembled, so the recipe default whitelist keeps
# it on the high-precision gather. It is still Block FP8 quantized after
# unshard, which is why it shows up in ``_collect_flags`` at all.
_STACKED_GATE_UP = ".feed_forward.shared_experts.w13"


@pytest.mark.parametrize("family", ["v4", "v41"])
def test_default_whitelist_excludes_stacked_gate_up(family):
    """fsdp_prequantize_fqns=None keeps every dim-0 projection and drops the stacked w13."""
    spec = (
        build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
        if family == "v4"
        else v41_model_registry("deepseek_v4_1_stripped")
    )
    quantization = _quantization_config(enable_fsdp_prequantize=True)

    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    flags = _collect_flags(converted)
    assert flags, "debugmodel must expose quantized Linear nodes"
    stacked = {fqn: flag for fqn, flag in flags.items() if fqn.endswith(_STACKED_GATE_UP)}
    assert stacked, f"{family} must expose the stacked gate/up weight"
    assert not any(stacked.values()), f"stacked gate/up must keep the high-precision gather: {stacked}"
    kept = {fqn: flag for fqn, flag in flags.items() if fqn not in stacked}
    assert kept and all(kept.values()), f"default whitelist must enable every dim-0 projection: {kept}"


def test_full_a5_graphtrainer_quantizes_shared_and_routed_experts():
    from torchtitan_npu.override.common.rms_norm import AscRMSNorm
    from torchtitan_npu.override.deepseek_v4.compressor import asc

    config = deepseek_v4_flash_cpt_4k_a5()
    converted = apply_quantization_converter(
        config.model,
        config.extension.quantization,
        model_compile_enabled=False,
        override_config=config.override,
    )
    for suffix, config_type, prequantize in (
        (".feed_forward.shared_experts.w13", Linear.Config, False),
        (".feed_forward.shared_experts.w2", Linear.Config, False),
        (".feed_forward.routed_experts.w13", GroupedLinear.Config, True),
        (".feed_forward.routed_experts.w2", GroupedLinear.Config, True),
    ):
        projections = [node for fqn, node, _, _ in converted.traverse(config_type) if fqn.endswith(suffix)]
        assert projections, suffix
        assert all(node._torchao_npu_config.weight_config.fsdp_prequantize is prequantize for node in projections)
    assert converted.layers[0].attention.kv_norm.nope_head_dim == 448
    hca_norms = [layer.attention.compressor_128.norm for layer in converted.layers if layer.attention.compressor_128]
    assert hca_norms
    assert all(norm.nope_head_dim == 448 for norm in hca_norms)
    for field in ("compressor", "compressor_128"):
        compressor = next(cfg for layer in converted.layers if (cfg := getattr(layer.attention, field)) is not None)
        fused_norm = asc(compressor).norm
        assert isinstance(fused_norm.norm, AscRMSNorm.Config)
        assert fused_norm.norm.normalized_shape == compressor.head_dim
        assert fused_norm._fake_quant_config is compressor.norm._fake_quant_config


def test_user_whitelist_disables_non_whitelisted_weights():
    """Only whitelisted FQNs keep fsdp_prequantize=True; the rest are explicitly off."""
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    quantization = _quantization_config(
        enable_fsdp_prequantize=True,
        fsdp_prequantize_fqns=[".attention.wq_a"],
    )

    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    flags = _collect_flags(converted)
    wq_a = [flag for fqn, flag in flags.items() if fqn.endswith(".attention.wq_a")]
    wkv = [flag for fqn, flag in flags.items() if fqn.endswith(".attention.wkv")]
    assert wq_a and all(wq_a), "whitelisted wq_a must stay enabled"
    assert wkv and not any(wkv), "non-whitelisted wkv must be explicitly disabled"


def test_whitelist_pattern_matching_nothing_raises():
    """A typo'd pattern is a silent no-op and must fail fast."""
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    quantization = _quantization_config(
        enable_fsdp_prequantize=True,
        fsdp_prequantize_fqns=[".attention.wq_a", ".attention.wk_b"],  # wk_b does not exist
    )

    with pytest.raises(ValueError, match="matched no quantized weight"):
        apply_quantization_converter(spec, quantization, model_compile_enabled=False)


def test_explicit_empty_whitelist_with_master_switch_raises():
    """An explicitly empty whitelist contradicts enable_fsdp_prequantize=True."""
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    quantization = _quantization_config(enable_fsdp_prequantize=True, fsdp_prequantize_fqns=[])

    with pytest.raises(ValueError, match="explicitly empty"):
        apply_quantization_converter(spec, quantization, model_compile_enabled=False)


def test_master_switch_off_leaves_policy_inert():
    """Without enable_fsdp_prequantize no whitelist policy is applied at all."""
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    quantization = _quantization_config(enable_fsdp_prequantize=False)

    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    flags = _collect_flags(converted)
    assert flags and not any(flags.values()), "disabled master switch must keep every weight off pre-quantize"


def test_default_whitelist_covers_every_prequantizable_projection():
    """The default whitelist must stay in sync with the recipe's quantized set.

    Every quantized Linear the converter produces on the real debugmodel tree
    matches a default entry -- except the stacked gate/up weight, which cannot
    pre-quantize -- so a renamed or removed projection surfaces here (or as a
    zero-match error) instead of silently dropping pre-quantize coverage.
    Matching the config tree, not a literal copy of the constant, is what keeps
    the whitelist and the recipe in sync.
    """
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    quantization = _quantization_config(enable_fsdp_prequantize=True)
    converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    defaults = get_model_config(_model_type_for_config(spec)).prequantize_default_fqns
    all_patterns = tuple(pattern for scope in defaults.values() for pattern in scope)
    flags = _collect_flags(converted)
    assert flags, "debugmodel must expose quantized Linear nodes"
    for fqn in flags:
        if fqn.endswith(_STACKED_GATE_UP):
            continue
        assert any(_match_fqn_suffix(fqn, pattern) for pattern in all_patterns), (
            f"quantized weight {fqn!r} matches no default whitelist entry; update the model's prequantize_default_fqns"
        )


def test_mxfp4_communication_keeps_dense_fp8_and_experts_mxfp4():
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    config = _quantization_config(
        enable_mxfp4_qat=True,
        enable_fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
    )
    converted = apply_quantization_converter(spec, config, model_compile_enabled=False)
    dense = [
        (fqn, node._torchao_npu_config.weight_config)
        for fqn, node, _, _ in converted.traverse(Linear.Config)
        if getattr(node, "_torchao_npu_config", None) is not None
    ]
    assert dense
    prequantized_dense = [(fqn, weight) for fqn, weight in dense if not fqn.endswith(_STACKED_GATE_UP)]
    assert prequantized_dense
    for _, weight in prequantized_dense:
        assert weight.fsdp_prequantize
        assert weight.fsdp_prequantize_dtype == "fp8"
        assert weight.mxfp4_fake_quantize_config is None
    experts = list(converted.traverse(GroupedLinear.Config))
    assert experts
    for _, node, _, _ in experts:
        weight = node._torchao_npu_config.weight_config
        assert weight.fsdp_prequantize
        assert weight.fsdp_prequantize_dtype == "mxfp4"
        assert weight.mxfp4_fake_quantize_config is not None


def test_mxfp4_communication_config_reaches_only_selected_experts():
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    config = _quantization_config(
        enable_mxfp4_qat=True,
        enable_fsdp_prequantize=True,
        fsdp_prequantize_dtype="mxfp4",
        fsdp_prequantize_fqns=[".feed_forward.routed_experts.w13", ".feed_forward.routed_experts.w2"],
    )
    converted = apply_quantization_converter(spec, config, model_compile_enabled=False)
    experts = list(converted.traverse(GroupedLinear.Config))
    assert experts
    for _, node, _, _ in experts:
        weight = node._torchao_npu_config.weight_config
        assert weight.fsdp_prequantize
        assert weight.fsdp_prequantize_dtype == "mxfp4"
        assert weight.mxfp4_fake_quantize_config is not None
    assert not any(_collect_flags(converted).values())


def test_prequantize_policy_groups_enabled_and_disabled_logs(caplog):
    spec = build_model_config("debugmodel", num_mtp_layers=0, parallelism=ParallelismConfig())
    quantization = _quantization_config(
        enable_fsdp_prequantize=True,
        fsdp_prequantize_fqns=[".attention.wq_a", ".attention.wq_b"],
    )

    with caplog.at_level(logging.DEBUG):
        converted = apply_quantization_converter(spec, quantization, model_compile_enabled=False)

    all_flags = {
        fqn: node._torchao_npu_config.weight_config.fsdp_prequantize
        for fqn, node, _, _ in converted.traverse(Module.Config, recurse=True)
        if getattr(getattr(node, "_torchao_npu_config", None), "weight_config", None) is not None
        and hasattr(node._torchao_npu_config.weight_config, "fsdp_prequantize")
    }
    enabled_count = sum(1 for enabled in all_flags.values() if enabled)
    disabled_count = sum(1 for enabled in all_flags.values() if not enabled)
    info_policy = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.INFO and "fsdp_prequantize=" in record.getMessage()
    ]
    debug_policy = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.DEBUG and "fsdp_prequantize=" in record.getMessage()
    ]

    enabled_lines = [msg for msg in info_policy if "fsdp_prequantize=enabled matched=" in msg]
    disabled_lines = [msg for msg in info_policy if "fsdp_prequantize=disabled (not in whitelist)" in msg]
    assert len(enabled_lines) == 1
    assert f"matched={enabled_count}" in enabled_lines[0]
    assert "layers.{0..3}.attention.wq_a" in enabled_lines[0]
    assert "layers.{0..3}.attention.wq_b" in enabled_lines[0]
    # One disabled summary per pre-quantizing converter scope (dense + routed experts).
    assert len(disabled_lines) == 2
    disabled_matched = sum(int(match.group(1)) for msg in disabled_lines if (match := re.search(r"matched=(\d+)", msg)))
    assert disabled_matched == disabled_count
    assert len(debug_policy) == len(all_flags)
    assert any("fsdp_prequantize=enabled (whitelist .attention.wq_a)" in msg for msg in debug_policy)
    assert any("fsdp_prequantize=enabled (whitelist .attention.wq_b)" in msg for msg in debug_policy)
