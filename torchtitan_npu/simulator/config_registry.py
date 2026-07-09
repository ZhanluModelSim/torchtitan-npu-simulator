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


def _to_simulation_config(base_config: object, output_dir: str, *, comm_mode: str = "fake_backend") -> SimulationTrainerConfig:
    base_fields = {f.name: getattr(base_config, f.name) for f in dataclasses.fields(base_config)}
    sim_config = SimulationTrainerConfig(**base_fields, simulation=SimulationConfig(output_dir=output_dir))
    sim_config.compile.enable = False
    sim_config.comm.mode = comm_mode
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


def deepseek_v4_pro_simulate_16_layers_cp4() -> SimulationTrainerConfig:
    """CP=4 variant for testing context parallel communication capture.

    Uses the 16-layer model with CP=4 to verify that _WindowExchange
    (P2P isend/irecv) and _allgather_seq (all_gather_tensor_autograd)
    appear in the captured L0 graph.
    """
    base_config = deepseek_v4_pro_debug_16_layers()
    base_config = dataclasses.replace(
        base_config,
        parallelism=dataclasses.replace(
            base_config.parallelism,
            context_parallel_degree=4,
        ),
    )
    return _to_simulation_config(
        base_config,
        output_dir="./simulator_output/deepseek_v4_pro_16_layers_cp4",
    )


def deepseek_v4_pro_simulate_16_layers_pp4_cp4() -> SimulationTrainerConfig:
    """PP=4 + CP=4 variant for multi-process CP+FSDP capture.

    Run with: ``NGPU=64 torchrun --nproc_per_node=4 -m torchtitan_npu.entry
    --module torchtitan_npu.simulator
    --config deepseek_v4_pro_simulate_16_layers_pp4_cp4
    --training.steps=1``

    Uses real parallel degrees (PP=4, CP=4, DP=4 → world_size=64).
    The gloo PG has only 4 processes (one per PP stage); CP/FSDP/TP/EP
    subgroups use FakeProcessGroup with the correct simulated size.
    ``TORCHTITAN_SIM_WORLD_SIZE=64`` env var tells init_distributed to
    return 64 (not gloo's 4) for ParallelDims validation.
    """
    base_config = deepseek_v4_pro_debug_16_layers()
    base_config = dataclasses.replace(
        base_config,
        training=dataclasses.replace(
            base_config.training,
            num_mtp_modules=0,
            local_batch_size=4,
        ),
        parallelism=dataclasses.replace(
            base_config.parallelism,
            pipeline_parallel_degree=4,
            # Real parallel degrees — ParallelDims validates
            # dp_replicate * dp_shard * cp * tp * pp == world_size
            # 1 * 4 * 4 * 1 * 4 = 64 == TORCHTITAN_SIM_WORLD_SIZE
            tensor_parallel_degree=1,
            context_parallel_degree=4,
            expert_parallel_degree=1,
            data_parallel_shard_degree=-1,  # auto: 64 / (1*4*1*4) = 4
        ),
    )
    sim_config = _to_simulation_config(
        base_config,
        output_dir="./simulator_output/deepseek_v4_pro_16_layers_pp4_cp4",
        comm_mode="multi_proc_meta",
    )
    sim_config.simulation.simulated_parallel_degrees = {
        "pp": 4, "tp": 1, "cp": 4, "ep": 1,
        "dp_replicate": 1, "dp_shard": 4,
        "etp": 1, "world_size": 64,
    }
    return sim_config
    """Multi-process version: uses gloo PG with 16 processes (one per PP stage).

    Each process runs one PP stage with real 1F1B scheduling. All
    communication is intercepted as no-op (meta device, no real data).
    Each process captures its own L0-L3 IR; rank 0 merges them.

    The mesh has 16 ranks (PP degree only). TP/CP/EP/DP are simulated
    by the comm_events interceptors. The config's parallel degrees are
    set to the full values (TP=8, CP=4, EP=128) for RankTable, but
    ParallelDims uses pp=16, others=1 for mesh creation.

    Run with: ``NGPU=2048 torchrun --nproc_per_node=16 -m torchtitan_npu.entry
    --module torchtitan_npu.simulator
    --config deepseek_v4_pro_simulate_61_layers_pp16_tp8_cp4_ep128_multiproc
    --training.steps=1``
    """
    base_config = deepseek_v4_pro_debug_61_layers_4k_384die()
    base_config = dataclasses.replace(
        base_config,
        training=dataclasses.replace(
            base_config.training,
            num_mtp_modules=0,
            local_batch_size=16,
        ),
        parallelism=dataclasses.replace(
            base_config.parallelism,
            pipeline_parallel_degree=16,
            # For mesh creation: only PP is real (16 procs).
            # Other degrees are simulated by comm_events interceptors.
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            expert_parallel_degree=1,
            data_parallel_shard_degree=1,
        ),
    )
    sim_config = _to_simulation_config(
        base_config,
        output_dir="./simulator_output/deepseek_v4_pro_61_layers_pp16_tp8_cp4_ep128_multiproc",
        comm_mode="multi_proc_meta",
    )
    # Store the "real" parallel degrees for RankTable computation
    sim_config.simulation.simulated_parallel_degrees = {
        "pp": 16, "tp": 8, "cp": 4, "ep": 128,
        "dp_replicate": 1, "dp_shard": 4,
        "etp": 1, "world_size": 2048,
    }
    return sim_config
