# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.qat import IntxFakeQuantizeConfig, QATStep
from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs import ParamSwapConfig
from torchao_npu.quantization.quant_configs import (
    BlockMXQuantizeConfig,
    HiF8QuantizeConfig,
    MXQuantizeConfig,
)
from torchao_npu.wrapper_tensors import HiF8TrainingWeightWrapperTensor


def test_prepare_registers_swap_handlers_without_explicit_wrapper_import():
    """A fresh interpreter that only imports ParamSwapConfig can run a prepare step.

    The swap-handler registry is populated lazily by ``_param_swap_config_transform``
    (it imports the wrapper tensor modules on first use), so no explicit
    ``import torchao_npu.wrapper_tensors`` is required of users.
    This must run in a subprocess: in-process, test collection has long since
    imported the wrapper modules and pre-filled the registry, which would mask
    a broken registration trigger.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import torch
        from torchao.quantization.qat.fake_quantize_config import Float8FakeQuantizeConfig
        from torchao.quantization.quant_api import quantize_
        from torchao_npu.configs import ParamSwapConfig

        model = torch.nn.Linear(4, 2)
        config = ParamSwapConfig(weight_config=Float8FakeQuantizeConfig(), step="prepare")
        quantize_(model, config, filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear))

        from torchao_npu.wrapper_tensors import Float8TrainingWeightWrapperTensor

        assert isinstance(model.weight.data, Float8TrainingWeightWrapperTensor)
    """)
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, f"prepare in a fresh interpreter failed:\n{result.stderr}"


# =========================================================================
# Test configs in the prepare step
# =========================================================================


def test_config_step_validation():
    """step must be 'prepare' or 'convert'."""
    weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    ParamSwapConfig(weight_config=weight_config, step="prepare")
    ParamSwapConfig(step="convert")
    with pytest.raises(ValueError, match=r"^`step` must be one of \['prepare', 'convert'\]$"):
        ParamSwapConfig(weight_config=weight_config, step="blah")


def test_config_requires_weight_config():
    """prepare step requires weight_config, activation_config, or base_config."""
    with pytest.raises(
        ValueError,
        match=r"^Must specify `base_config`, `activation_config`, or `weight_config` in the prepare step$",
    ):
        ParamSwapConfig(step="prepare")


def test_config_requires_weight_config_in_prepare():
    """ParamSwapConfig requires weight_config during prepare, even if activation_config is set."""
    act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    with pytest.raises(
        ValueError,
        match=r"^`weight_config` is required for the prepare step of ParamSwapConfig\.$",
    ):
        ParamSwapConfig(activation_config=act_config, step="prepare")


@pytest.mark.parametrize("granularity", [PerRow()])
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_float8_weight_config_variants(granularity, dtype):
    """All Float8FakeQuantizeConfig variants should be accepted."""
    config = Float8FakeQuantizeConfig(dtype=dtype, granularity=granularity)
    qat_config = ParamSwapConfig(weight_config=config, step="prepare")
    assert qat_config.step == QATStep.PREPARE


@pytest.mark.parametrize(
    "base_config, expected_weight_config, expected_activation_config",
    [
        (
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            Float8FakeQuantizeConfig,
            Float8FakeQuantizeConfig,
        ),
    ],
)
def test_config_infer_from_base_config(base_config, expected_weight_config, expected_activation_config):
    """ParamSwapConfig can infer fake quantize configs from a PTQ base_config."""
    qat_config = ParamSwapConfig(base_config=base_config, step="prepare")
    assert isinstance(qat_config.weight_config, expected_weight_config)
    assert isinstance(qat_config.activation_config, expected_activation_config)
    assert qat_config.base_config is None


def test_config_rejects_invalid_weight_config():
    """Invalid (non-FP8, non-NPU-MX, non-HiF8) weight configs should be rejected."""
    intx_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
    with pytest.raises(
        ValueError,
        match=(
            r"^Only `Float8FakeQuantizeConfig`, `MXQuantizeConfig`, `BlockMXQuantizeConfig`, or "
            r"`HiF8QuantizeConfig` is supported for `weight_config` in ParamSwapConfig yet\.$"
        ),
    ):
        ParamSwapConfig(weight_config=intx_config, step="prepare")


def test_config_rejects_invalid_activation_config():
    """Invalid (non-FP8, non-NPU-MX, non-HiF8) activation configs should be rejected."""
    weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    activation_config = IntxFakeQuantizeConfig(torch.int8, "per_channel")
    with pytest.raises(
        ValueError,
        match=(
            r"^Only `Float8FakeQuantizeConfig`, `MXQuantizeConfig`, or `HiF8QuantizeConfig` is supported for "
            r"`activation_config` in ParamSwapConfig yet\.$"
        ),
    ):
        ParamSwapConfig(
            weight_config=weight_config,
            activation_config=activation_config,
            step="prepare",
        )


def test_config_rejects_invalid_base_config_in_prepare_step():
    """Only Float8DynamicActivationFloat8WeightConfig is accepted as base_config."""
    from torchao.quantization import Int4WeightOnlyConfig

    base_config = Int4WeightOnlyConfig(group_size=32)
    with pytest.raises(
        ValueError,
        match=(
            r"^Only `Float8DynamicActivationFloat8WeightConfig` is supported for "
            r"`base_config` in ParamSwapConfig yet\.$"
        ),
    ):
        ParamSwapConfig(base_config=base_config, step="prepare")


@pytest.mark.parametrize(
    "elem_dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    ],
)
def test_config_accepts_npu_mx_weight_config(elem_dtype):
    """MXQuantizeConfig should be accepted as weight_config."""
    config = MXQuantizeConfig(elem_dtype=elem_dtype)
    qat_config = ParamSwapConfig(weight_config=config, step="prepare")
    assert qat_config.step == QATStep.PREPARE


def test_config_accepts_block_mx_weight_config():
    """BlockMXQuantizeConfig should be accepted as weight_config."""
    config = BlockMXQuantizeConfig()
    qat_config = ParamSwapConfig(weight_config=config, step="prepare")
    assert qat_config.step == QATStep.PREPARE


def test_config_accepts_npu_mx_weight_and_activation_config():
    """MXQuantizeConfig should be accepted for both weight and activation."""
    weight_config = MXQuantizeConfig()
    act_config = MXQuantizeConfig()
    qat_config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=act_config,
        step="prepare",
    )
    assert qat_config.weight_config is weight_config
    assert qat_config.activation_config is act_config


def test_config_accepts_block_weight_with_mx_activation():
    """BlockMXQuantizeConfig as weight_config with MXQuantizeConfig as activation_config."""
    weight_config = BlockMXQuantizeConfig()
    act_config = MXQuantizeConfig()
    qat_config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=act_config,
        step="prepare",
    )
    assert qat_config.weight_config is weight_config
    assert qat_config.activation_config is act_config


def test_param_swap_accepts_hif8_for_weight_and_activation():
    """HiF8QuantizeConfig should be accepted for both weight and activation."""
    weight_config = HiF8QuantizeConfig()
    activation_config = HiF8QuantizeConfig()

    config = ParamSwapConfig(
        weight_config=weight_config,
        activation_config=activation_config,
        step=QATStep.PREPARE,
    )

    assert config.weight_config is weight_config
    assert config.activation_config is activation_config
    assert config.step == QATStep.PREPARE


def test_hif8_config_rejects_non_hif8_dtype():
    with pytest.raises(ValueError, match=r"HiF8QuantizeConfig requires torch_npu\.hifloat8"):
        HiF8QuantizeConfig(elem_dtype=torch.float8_e4m3fn)


def test_quantize_prepares_hif8_parameter_wrapper():
    config = HiF8QuantizeConfig()
    module = torch.nn.Linear(64, 128)
    quantize_config = ParamSwapConfig(
        weight_config=config,
        activation_config=config,
        step=QATStep.PREPARE,
    )

    quantize_(module, quantize_config)

    assert isinstance(module.weight, torch.nn.Parameter)
    assert isinstance(module.weight, HiF8TrainingWeightWrapperTensor)
    assert module.weight.weight_config is config
    assert module.weight.activation_config is config


# =========================================================================
# Test configs in the convert step
# =========================================================================


def test_config_rejects_base_config_in_convert():
    """base_config is not supported in the convert step."""
    weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    ParamSwapConfig(weight_config=weight_config, step="prepare")
    base_config = Float8DynamicActivationFloat8WeightConfig()
    with pytest.raises(
        NotImplementedError,
        match=r"^Applying PTQ in the convert step is not implemented yet\.$",
    ):
        ParamSwapConfig(base_config=base_config, step="convert")


def test_config_rejects_weight_config_in_convert():
    """weight_config cannot be specified in the convert step."""
    weight_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    with pytest.raises(
        ValueError,
        match=r"^Cannot specify `weight_config` or `activation_config` in the convert step$",
    ):
        ParamSwapConfig(weight_config=weight_config, step="convert")


def test_config_rejects_activation_config_in_convert():
    """activation_config cannot be specified in the convert step."""
    act_config = Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow())
    with pytest.raises(
        ValueError,
        match=r"^Cannot specify `weight_config` or `activation_config` in the convert step$",
    ):
        ParamSwapConfig(activation_config=act_config, step="convert")


# =========================================================================
# Test params_filter_fn
# =========================================================================


def test_config_default_params_filter_fn():
    """Default params_filter_fn is _is_parameter (wraps all parameters)."""
    qat_config = ParamSwapConfig(
        weight_config=Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow()),
        step="prepare",
    )
    from torchao_npu.quantization.filters import _is_parameter

    assert qat_config.params_filter_fn is _is_parameter


def test_config_custom_params_filter_fn():
    """Custom params_filter_fn is stored and forwarded to the handler."""

    def custom_filter(param, fqn):
        return True

    qat_config = ParamSwapConfig(
        weight_config=Float8FakeQuantizeConfig(dtype=torch.float8_e4m3fn, granularity=PerRow()),
        step="prepare",
        params_filter_fn=custom_filter,
    )
    assert qat_config.params_filter_fn is custom_filter
