# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

torch_npu = pytest.importorskip("torch_npu", reason="requires torch_npu + CANN (see Task 20 container setup)")

from torchtitan_npu.simulator.config_registry import (  # noqa: E402
    deepseek_v4_pro_simulate_16_layers,
    deepseek_v4_pro_simulate_61_layers,
)


def test_simulate_61_layers_config_matches_acceptance_target():
    from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_pro_debug_61_layers_4k_384die

    base_config = deepseek_v4_pro_debug_61_layers_4k_384die()
    sim_config = deepseek_v4_pro_simulate_61_layers()

    assert sim_config.model_spec.name == base_config.model_spec.name
    assert sim_config.model_spec.flavor == base_config.model_spec.flavor
    assert sim_config.parallelism.expert_parallel_degree == 192
    assert sim_config.debug.moe_force_load_balance is True  # already True on the acceptance config
    assert sim_config.optimizer.swap_optimizer == base_config.optimizer.swap_optimizer
    assert sim_config.simulation.output_dir == "./simulator_output/deepseek_v4_pro_61_layers"
    # base_config.compile.enable is True (real training uses inductor_npu_ext
    # compilation), but entry.py::main() checks config.compile.enable BEFORE
    # config.build() ever runs (and therefore before SimulationTrainer
    # .__init__'s own override takes effect), raising a hard RuntimeError
    # about a missing "inductor_npu_ext" module for any un-forced compiled
    # config -- found via the real 61-layer smoke run.
    assert base_config.compile.enable is True
    assert sim_config.compile.enable is False


def test_simulate_16_layers_config_is_a_smaller_variant():
    sim_config = deepseek_v4_pro_simulate_16_layers()
    assert sim_config.model_spec.flavor == "v4_pro_debug_16_layers"
