# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
import dataclasses
import functools
from types import SimpleNamespace

import pytest

from torchtitan_npu.models.deepseek_v32 import deepseekv32_configs


def _normalize_config_value(value):
    if isinstance(value, functools.partial):
        return (
            value.func,
            value.args,
            _normalize_config_value(value.keywords),
        )
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normalize_config_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {
            key: _normalize_config_value(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(_normalize_config_value(item) for item in value)
    return value


def test_update_from_config_keeps_grouped_mm_enabled():
    model_config = deepcopy(deepseekv32_configs["smoketest"]())
    moe_layers = [layer for layer in model_config.layers if layer.moe is not None]
    assert moe_layers
    for layer in moe_layers:
        layer.moe.experts.use_grouped_mm = True

    trainer_config = SimpleNamespace(
        training=SimpleNamespace(seq_len=4096),
        parallelism=SimpleNamespace(
            context_parallel_degree=1,
            expert_parallel_comm_backend="standard",
        ),
        debug=SimpleNamespace(moe_force_load_balance=False),
    )

    model_config.update_from_config(trainer_config=trainer_config)

    for layer in model_config.layers:
        if layer.moe is not None:
            assert layer.moe.experts.use_grouped_mm is True


@pytest.mark.parametrize(
    "flavor",
    ("smoketest", "671B_debug_4_layers", "671B_debug_128die"),
)
def test_model_overrides_round_trip_registered_presets(flavor):
    from torchtitan_npu.models.deepseek_v32.config_overrides import (
        DeepSeekV32ModelOverrides,
    )

    original = deepseekv32_configs[flavor]()
    overrides = DeepSeekV32ModelOverrides.from_model_config(original)
    rebuilt = overrides.to_model_config()

    assert _normalize_config_value(rebuilt) == _normalize_config_value(original)


@pytest.mark.parametrize(
    "module",
    ("torchtitan_npu.models.deepseek_v32", "torchtitan_npu.simulator"),
)
def test_model_overrides_cli_rebuilds_v32_layers(module):
    from torchtitan.config import ConfigManager

    config = ConfigManager().parse_args(
        [
            "--module",
            module,
            "--config",
            "deepseek_v32_smoketest",
            "--model-overrides.n-layers",
            "3",
            "--model-overrides.dim",
            "192",
            "--model-overrides.n-heads",
            "6",
            "--model-overrides.num-experts",
            "16",
            "--model-overrides.router-top-k",
            "4",
            "--model-overrides.index-topk",
            "64",
        ]
    )

    model = config.model_spec.model
    assert model.dim == 192
    assert len(model.layers) == 3
    assert model.layers[0].attention.n_heads == 6
    assert model.layers[0].attention.index_topk == 64
    assert model.layers[1].moe.num_experts == 16
    assert model.layers[1].moe.router.top_k == 4


@pytest.mark.parametrize(
    ("args", "message"),
    (
        (
            ["--model-overrides.router-top-k", "9"],
            "router_top_k must be <= num_experts",
        ),
        (
            ["--model-overrides.index-head-dim", "96"],
            "index_head_dim must be a power of two",
        ),
        (
            ["--model-overrides.num-mtp-modules", "-1"],
            "num_mtp_modules must be >= 0",
        ),
    ),
)
def test_model_overrides_validate_before_run(args, message):
    from torchtitan.config import ConfigManager

    with pytest.raises(ValueError, match=message):
        ConfigManager().parse_args(
            [
                "--module",
                "torchtitan_npu.models.deepseek_v32",
                "--config",
                "deepseek_v32_smoketest",
                *args,
            ]
        )


def test_simulator_config_reuses_training_config():
    from torchtitan_npu.models.deepseek_v32.config_registry import (
        deepseek_v32_smoketest as training_smoketest,
    )
    from torchtitan_npu.simulator.config_registry import (
        deepseek_v32_smoketest as simulator_smoketest,
    )

    training_config = training_smoketest()
    simulation_config = simulator_smoketest()

    for field in dataclasses.fields(training_config):
        if field.name == "compile":
            assert (
                simulation_config.compile.components
                == training_config.compile.components
            )
            assert simulation_config.compile.enable is False
        elif field.name == "model_spec":
            assert simulation_config.model_spec.name == training_config.model_spec.name
            assert simulation_config.model_spec.flavor == training_config.model_spec.flavor
            assert _normalize_config_value(
                simulation_config.model_spec.model
            ) == _normalize_config_value(training_config.model_spec.model)
        else:
            assert getattr(simulation_config, field.name) == getattr(
                training_config,
                field.name,
            )
    assert simulation_config.simulation.output_dir.endswith(
        "deepseek_v32_smoketest"
    )


def test_smoketest_uses_full_precision_without_fsdp():
    from torchtitan_npu.models.deepseek_v32.config_registry import (
        deepseek_v32_smoketest,
    )

    config = deepseek_v32_smoketest()
    assert config.training.dtype == "float32"
    assert config.training.mixed_precision_param == "float32"
    assert config.parallelism.data_parallel_shard_degree == -1


def test_tp_smoketest_disables_gmm_and_enables_tp():
    from torchtitan_npu.models.deepseek_v32.config_registry import (
        deepseek_v32_tp_smoketest,
    )

    config = deepseek_v32_tp_smoketest()
    converter_names = {
        converter._owner._model_config.name
        for converter in config.model_converters.converters
    }
    assert config.parallelism.tensor_parallel_degree == 2
    assert config.parallelism.disable_loss_parallel is True
    assert "npu_gmm" not in converter_names
    assert converter_names == {
        "npu_dsa",
        "npu_rms_norm",
        "npu_rope",
    }
