"""Utility to extract PP degree from a simulator config for torchrun nproc_per_node.

In multi_proc_meta mode, torchrun's --nproc_per_node must equal the PP degree
(not the full world_size). This helper reads the config's parallelism settings
and simulated_parallel_degrees to determine the correct nproc_per_node value.

Usage:
    from torchtitan_npu.simulator.utils import get_nproc_per_node

    nproc = get_nproc_per_node(config)
    # nproc = 4 for PP=4, or 1 for PP=1 (fake_backend mode)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SimulationRuntime:
    """Runtime settings resolved from the final, CLI-overridden config."""

    comm_mode: str
    pp_degree: int
    world_size: int
    parallel_degrees: dict[str, int]


def _parallel_degree(parallelism: Any, name: str) -> int:
    value = getattr(parallelism, name, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"parallelism.{name} must be a positive integer, got {value!r}")
    return value


def get_nproc_per_node(config: Any) -> int:
    """Return the final PP degree used as the real worker count."""
    parallelism = getattr(config, "parallelism", None)
    if parallelism is not None:
        return _parallel_degree(parallelism, "pipeline_parallel_degree")

    return 1


def get_world_size(config: Any) -> int:
    """Return the configured or derivable full simulated world size."""
    sim_config = getattr(config, "simulation", None)
    if sim_config is not None:
        world_size = getattr(sim_config, "world_size", None)
        if world_size is not None:
            return int(world_size)
        sim_degrees = getattr(sim_config, "simulated_parallel_degrees", {})
        if sim_degrees and "world_size" in sim_degrees:
            return int(sim_degrees["world_size"])

    # Fall back to computing from parallel degrees
    parallelism = getattr(config, "parallelism", None)
    if parallelism is not None:
        pp = _parallel_degree(parallelism, "pipeline_parallel_degree")
        tp = _parallel_degree(parallelism, "tensor_parallel_degree")
        cp = _parallel_degree(parallelism, "context_parallel_degree")
        dp_replicate = _parallel_degree(parallelism, "data_parallel_replicate_degree")
        dp_shard = getattr(parallelism, "data_parallel_shard_degree", 1)
        if dp_shard == -1:
            # Can't compute without knowing world_size; return 0 as signal
            return 0
        if isinstance(dp_shard, bool) or not isinstance(dp_shard, int) or dp_shard < 1:
            raise ValueError(
                "parallelism.data_parallel_shard_degree must be -1 or a "
                f"positive integer, got {dp_shard!r}"
            )
        # EP/ETP reinterpret the dense DP/CP/TP mesh and are not additional
        # world-size factors.
        return pp * dp_replicate * dp_shard * cp * tp

    return 1


def resolve_simulation_runtime(
    config: Any,
    *,
    world_size: int | None = None,
) -> SimulationRuntime:
    """Normalize simulator runtime state after all CLI overrides are applied.

    ``comm.mode`` is an implementation detail derived from the final PP
    degree. The repeated ``simulated_parallel_degrees`` mapping is rebuilt for
    compatibility so it cannot retain values from the registry config after a
    CLI override.
    """
    parallelism = getattr(config, "parallelism", None)
    simulation = getattr(config, "simulation", None)
    comm = getattr(config, "comm", None)
    if parallelism is None or simulation is None or comm is None:
        raise TypeError("config must provide parallelism, simulation, and comm sections")

    pp = _parallel_degree(parallelism, "pipeline_parallel_degree")
    tp = _parallel_degree(parallelism, "tensor_parallel_degree")
    cp = _parallel_degree(parallelism, "context_parallel_degree")
    ep = _parallel_degree(parallelism, "expert_parallel_degree")
    etp = _parallel_degree(parallelism, "expert_tensor_parallel_degree")
    dp_replicate = _parallel_degree(parallelism, "data_parallel_replicate_degree")
    dp_shard = getattr(parallelism, "data_parallel_shard_degree", 1)
    if dp_shard != -1 and (
        isinstance(dp_shard, bool) or not isinstance(dp_shard, int) or dp_shard < 1
    ):
        raise ValueError(
            "parallelism.data_parallel_shard_degree must be -1 or a "
            f"positive integer, got {dp_shard!r}"
        )

    resolved_world_size = get_world_size(config) if world_size is None else world_size
    if (
        isinstance(resolved_world_size, bool)
        or not isinstance(resolved_world_size, int)
        or resolved_world_size < 1
    ):
        raise ValueError(
            "simulation world_size could not be resolved; set "
            "--simulation.world-size or NGPU"
        )

    fixed_dense_degree = pp * dp_replicate * cp * tp
    if dp_shard == -1:
        if resolved_world_size % fixed_dense_degree != 0:
            raise ValueError(
                f"world_size {resolved_world_size} must be divisible by PP*DP-replicate*CP*TP "
                f"({fixed_dense_degree})"
            )
        dp_shard = resolved_world_size // fixed_dense_degree
        parallelism.data_parallel_shard_degree = dp_shard
    else:
        expected_world_size = fixed_dense_degree * dp_shard
        if expected_world_size != resolved_world_size:
            raise ValueError(
                f"parallel degrees require world_size {expected_world_size}, "
                f"but simulator world_size is {resolved_world_size}"
            )

    if etp not in (1, tp):
        raise ValueError(f"ETP must be 1 or equal TP ({tp}), got {etp}")
    expert_domain = dp_shard * cp * tp
    expert_parallel_degree = ep * etp
    if expert_domain % expert_parallel_degree != 0:
        raise ValueError(
            f"EP*ETP ({expert_parallel_degree}) must divide DP-shard*CP*TP "
            f"({expert_domain})"
        )

    comm_mode = "multi_proc_meta" if pp > 1 else "fake_backend"
    comm.mode = comm_mode
    simulation.world_size = resolved_world_size
    parallel_degrees = {
        "pp": pp,
        "tp": tp,
        "cp": cp,
        "ep": ep,
        "dp_replicate": dp_replicate,
        "dp_shard": dp_shard,
        "etp": etp,
        "world_size": resolved_world_size,
    }
    simulation.simulated_parallel_degrees = parallel_degrees

    return SimulationRuntime(
        comm_mode=comm_mode,
        pp_degree=pp,
        world_size=resolved_world_size,
        parallel_degrees=parallel_degrees,
    )


def resolve_simulation_runtime_from_environment(config: Any) -> SimulationRuntime:
    """Resolve runtime, using environment only when config cannot provide it."""
    internal_world_size = os.environ.get("TORCHTITAN_SIM_WORLD_SIZE")
    configured_world_size = get_world_size(config)
    ngpu_world_size = os.environ.get("NGPU")
    if configured_world_size > 0:
        world_size = configured_world_size
    elif internal_world_size:
        world_size = int(internal_world_size)
    elif ngpu_world_size:
        world_size = int(ngpu_world_size)
    else:
        world_size = configured_world_size
    runtime = resolve_simulation_runtime(config, world_size=world_size)
    os.environ["NGPU"] = str(runtime.world_size)
    os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(runtime.world_size)
    return runtime
