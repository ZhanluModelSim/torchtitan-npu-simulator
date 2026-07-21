# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest

torch_npu = pytest.importorskip("torch_npu", reason="requires torch_npu + CANN")

from torchtitan.components.quantization.mx import MXFP8Converter  # noqa: E402
from torchtitan.config import ConfigManager  # noqa: E402
from torchtitan_npu.converters.registry import has_npu_converter  # noqa: E402
from torchtitan_npu.models.deepseek_v4 import config_registry as model_configs  # noqa: E402
from torchtitan_npu.simulator import config_registry as simulator_configs  # noqa: E402


CONFIG_NAMES = (
    "deepseek_v4_flash_baseline_bf16",
    "deepseek_v4_flash_baseline_mxfp8",
    "deepseek_v4_pro_baseline_bf16",
    "deepseek_v4_pro_baseline_mxfp8",
    "deepseek_v4_pro_20t_baseline_bf16",
    "deepseek_v4_pro_20t_baseline_mxfp8",
    "deepseek_v4_smoketest",
)

NPU_CONVERTER_NAMES = (
    "npu_rms_norm",
    "npu_moe_dispatch",
    "npu_gmm",
    "npu_rope",
    "npu_smla",
    "npu_mhc_pre",
    "npu_mhc_post",
)


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_simulator_config_preserves_training_config(config_name):
    base_config = getattr(model_configs, config_name)()
    sim_config = getattr(simulator_configs, config_name)()

    for field in dataclasses.fields(base_config):
        if field.name == "compile":
            assert sim_config.compile.components == base_config.compile.components
            assert sim_config.compile.enable is False
        else:
            assert getattr(sim_config, field.name) == getattr(base_config, field.name)

    assert sim_config.simulation.output_dir == f"./simulator_output/{config_name}"
    assert sim_config.simulation.world_size is None


def test_smoketest_uses_production_npu_converter_path():
    config = model_configs.deepseek_v4_smoketest()

    assert config.model_spec.model.moe_args.use_grouped_mm is True
    assert len(config.model_converters.converters) == len(NPU_CONVERTER_NAMES)
    for converter_name in NPU_CONVERTER_NAMES:
        assert has_npu_converter(config.model_converters.converters, converter_name)


@pytest.mark.parametrize(
    "module",
    ("torchtitan_npu.models.deepseek_v4", "torchtitan_npu.simulator"),
)
def test_mxfp8_fqns_cli_override(module):
    config = ConfigManager().parse_args(
        [
            "--module",
            module,
            "--config",
            "deepseek_v4_flash_baseline_mxfp8",
            "--mxfp8-fqns",
            "moe.experts,post_attention.wo_a",
        ]
    )

    mxfp8_configs = [
        converter
        for converter in config.model_converters.converters
        if isinstance(converter, MXFP8Converter.Config)
    ]
    assert len(mxfp8_configs) == 1
    assert mxfp8_configs[0].fqns == ["moe.experts", "post_attention.wo_a"]


def test_mxfp8_fqns_cli_override_requires_mxfp8_recipe():
    with pytest.raises(ValueError, match="requires exactly one MXFP8 converter"):
        ConfigManager().parse_args(
            [
                "--module",
                "torchtitan_npu.models.deepseek_v4",
                "--config",
                "deepseek_v4_flash_baseline_bf16",
                "--mxfp8-fqns",
                "moe.experts",
            ]
        )
