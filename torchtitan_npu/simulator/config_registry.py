# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Simulation-config factory functions, resolved via
`--module torchtitan_npu.simulator --config <name>` (mirrors how every
other torchtitan_npu model's `config_registry.py` is resolved by
`ConfigManager`). Each function takes the *exact* existing
`torchtitan_npu.models.deepseek_v4.config_registry` factory output and
copies every field into a `SimulationTrainerConfig` -- the model_spec,
parallelism degrees, and every NPU-specific sub-config value (e.g.
`optimizer.swap_optimizer`) are reused unchanged; see design doc §7."""

from __future__ import annotations

import dataclasses

from torchtitan_npu.models.deepseek_v4.config_registry import (
    deepseek_v4_pro_debug_16_layers,
    deepseek_v4_pro_debug_61_layers_4k_384die,
)
from torchtitan_npu.simulator.trainer import SimulationConfig, SimulationTrainerConfig


def _to_simulation_config(base_config: object, output_dir: str) -> SimulationTrainerConfig:
    base_fields = {f.name: getattr(base_config, f.name) for f in dataclasses.fields(base_config)}
    return SimulationTrainerConfig(**base_fields, simulation=SimulationConfig(output_dir=output_dir))


def deepseek_v4_pro_simulate_61_layers() -> SimulationTrainerConfig:
    """Acceptance-target config: 61 layers, 384 experts,
    `expert_parallel_degree=192`, `384die` world size -- see
    docs/superpowers/specs/2026-07-01-npu-simulator-design.md."""
    base_config = deepseek_v4_pro_debug_61_layers_4k_384die()
    return _to_simulation_config(base_config, output_dir="./simulator_output/deepseek_v4_pro_61_layers")


def deepseek_v4_pro_simulate_16_layers() -> SimulationTrainerConfig:
    """Smaller/faster variant for local smoke testing before running the
    full 61-layer acceptance config (Task 20)."""
    base_config = deepseek_v4_pro_debug_16_layers()
    return _to_simulation_config(base_config, output_dir="./simulator_output/deepseek_v4_pro_16_layers")
