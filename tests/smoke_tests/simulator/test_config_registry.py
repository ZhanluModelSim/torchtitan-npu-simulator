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
from torchtitan_npu.models.deepseek_v4.config_overrides import (  # noqa: E402
    DeepSeekV4ModelOverrides,
)
from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model  # noqa: E402
from torchtitan_npu.models.deepseek_v4.moe import MoEArgs  # noqa: E402
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


def test_selective_ac_save_ops_cli_override():
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.simulator",
            "--config",
            "deepseek_v4_smoketest",
            "--simulation.selective-ac-save-ops",
            "default",
            "gmm",
        ]
    )

    assert config.simulation.selective_ac_save_ops == ["default", "gmm"]


def test_fsdp_allgather_fp8_cli_override_is_simulator_only():
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.simulator",
            "--config",
            "deepseek_v4_smoketest",
            "--simulation.enable-fsdp-allgather-fp8",
        ]
    )

    assert config.simulation.enable_fsdp_allgather_fp8 is True
    assert config.training.mixed_precision_param == "bfloat16"


def test_ep_dispatch_fp8_cli_override_is_simulator_only():
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.simulator",
            "--config",
            "deepseek_v4_smoketest",
            "--simulation.enable-ep-dispatch-fp8",
        ]
    )

    assert config.simulation.enable_ep_dispatch_fp8 is True


def test_mixed_precision_reduce_accepts_bfloat16_from_cli():
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.simulator",
            "--config",
            "deepseek_v4_smoketest",
            "--training.mixed-precision-reduce",
            "bfloat16",
        ]
    )

    assert config.training.mixed_precision_reduce == "bfloat16"


def test_model_override_schema_tracks_all_deepseek_v4_fields():
    model_fields = {
        field.name for field in dataclasses.fields(DeepSeekV4Model.Config)
    }
    override_fields = {
        field.name for field in dataclasses.fields(DeepSeekV4ModelOverrides)
    }
    assert override_fields == model_fields

    config = model_configs.deepseek_v4_pro_20t_baseline_bf16()
    assert dataclasses.asdict(config.model_overrides) == dataclasses.asdict(
        config.model_spec.model
    )
    assert {
        field.name for field in dataclasses.fields(config.model_overrides.moe_args)
    } == {field.name for field in dataclasses.fields(MoEArgs)}


@pytest.mark.parametrize(
    "module",
    ("torchtitan_npu.models.deepseek_v4", "torchtitan_npu.simulator"),
)
def test_deepseek_v4_model_overrides_cli(module):
    config = ConfigManager().parse_args(
        [
            "--module",
            module,
            "--config",
            "deepseek_v4_smoketest",
            "--model-overrides.n-layers",
            "3",
            "--model-overrides.dim",
            "192",
            "--model-overrides.compress-ratios",
            "1",
            "4",
            "128",
            "--model-overrides.moe-args.num-experts",
            "16",
            "--model-overrides.moe-args.top-k",
            "4",
            "--model-overrides.moe-args.num-expert-groups",
            "4",
            "--model-overrides.moe-args.num-limited-groups",
            "None",
            "--model-overrides.use-smla",
        ]
    )

    model = config.model_spec.model
    assert model.n_layers == 3
    assert model.dim == 192
    assert model.compress_ratios == (1, 4, 128)
    assert model.moe_args.num_experts == 16
    assert model.moe_args.top_k == 4
    assert model.moe_args.num_expert_groups == 4
    assert model.moe_args.num_limited_groups is None
    assert model.use_smla is True


def test_deepseek_v4_model_overrides_allow_extra_compress_ratios():
    config = ConfigManager().parse_args(
        [
            "--module",
            "torchtitan_npu.simulator",
            "--config",
            "deepseek_v4_smoketest",
            "--model-overrides.n-layers",
            "3",
        ]
    )

    assert config.model_spec.model.n_layers == 3
    assert config.model_spec.model.compress_ratios == (1, 1, 4, 128)


@pytest.mark.parametrize(
    ("args", "message"),
    (
        (
            ["--model-overrides.n-layers", "5"],
            "compress_ratios must contain at least one value per main layer",
        ),
        (
            [
                "--model-overrides.moe-args.num-experts",
                "10",
                "--model-overrides.moe-args.num-expert-groups",
                "4",
            ],
            "num_expert_groups must divide num_experts",
        ),
        (
            ["--model-overrides.o-groups", "3"],
            "n_heads must be divisible by o_groups",
        ),
    ),
)
def test_deepseek_v4_model_overrides_validate_before_run(args, message):
    with pytest.raises(ValueError, match=message):
        ConfigManager().parse_args(
            [
                "--module",
                "torchtitan_npu.simulator",
                "--config",
                "deepseek_v4_smoketest",
                *args,
            ]
        )
