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
    sim_config = SimulationTrainerConfig(**base_fields, simulation=SimulationConfig(output_dir=output_dir))
    # `entry.py::main()` checks `config.compile.enable` BEFORE `config.build()`
    # ever runs (and therefore before `SimulationTrainer.__init__`'s own
    # `config.compile.enable = False` override takes effect), raising
    # `RuntimeError: ... inductor_npu_ext is not available` for any base
    # config with compile enabled (e.g. the 61-layer acceptance config) --
    # found via the real 61-layer smoke run. Forcing it here, before the
    # config is ever returned to `ConfigManager`, closes that gap.
    sim_config.compile.enable = False
    return sim_config


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


def deepseek_v4_pro_simulate_61_layers_pp16_tp8_cp4_ep128() -> SimulationTrainerConfig:
    """Large-scale strategy: PP=16, TP=8, CP=4, EP=128, FSDP auto-shard.

    With ``data_parallel_shard_degree=-1`` torchtitan resolves
    ``dp_shard = world_size // (dp_replicate * cp * tp * pp)``.
    Setting ``world_size=2048`` yields ``dp_shard=4`` and
    ``efsdp = dp_shard * cp * tp // ep = 1``.  Total simulated dies = 2048.

    DeepSeekV4 does not support MTP together with PP, so ``num_mtp_modules``
    is forced to 0 for this PP-enabled strategy.
    """
    base_config = deepseek_v4_pro_debug_61_layers_4k_384die()
    base_config = dataclasses.replace(
        base_config,
        training=dataclasses.replace(
            base_config.training,
            num_mtp_modules=0,
            # PP=16 with microbatch_size=1 needs local_batch_size >= 16 so
            # that the 1F1B schedule receives at least 16 microbatches.
            # global_batch_size=384 remains divisible by (16 * dp_shard=4).
            local_batch_size=16,
        ),
        parallelism=dataclasses.replace(
            base_config.parallelism,
            pipeline_parallel_degree=16,
            tensor_parallel_degree=8,
            context_parallel_degree=4,
            expert_parallel_degree=128,
            data_parallel_shard_degree=-1,
        ),
    )
    return _to_simulation_config(
        base_config,
        output_dir="./simulator_output/deepseek_v4_pro_61_layers_pp16_tp8_cp4_ep128",
    )
