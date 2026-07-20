# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import pytest

torch_npu = pytest.importorskip("torch_npu", reason="requires torch_npu + CANN")

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
