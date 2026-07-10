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

from typing import Any


def get_nproc_per_node(config: Any) -> int:
    """Determine the correct --nproc_per_node for torchrun.

    In multi_proc_meta mode, nproc_per_node = PP degree.
    In fake_backend mode, nproc_per_node = 1 (single process).

    Reads from (in priority order):
    1. config.simulation.simulated_parallel_degrees["pp"]  (explicit)
    2. config.parallelism.pipeline_parallel_degree           (config field)
    3. 1 (fallback for fake_backend / PP=1)
    """
    # Check comm mode
    comm_config = getattr(config, "comm", None)
    comm_mode = getattr(comm_config, "mode", "fake_backend") if comm_config else "fake_backend"

    if comm_mode != "multi_proc_meta":
        return 1

    # Try simulated_parallel_degrees first (most explicit)
    sim_config = getattr(config, "simulation", None)
    if sim_config is not None:
        sim_degrees = getattr(sim_config, "simulated_parallel_degrees", {})
        if sim_degrees and "pp" in sim_degrees:
            return int(sim_degrees["pp"])

    # Fall back to parallelism config
    parallelism = getattr(config, "parallelism", None)
    if parallelism is not None:
        pp = getattr(parallelism, "pipeline_parallel_degree", 1)
        return max(int(pp), 1)

    return 1


def get_world_size(config: Any) -> int:
    """Determine the full simulated world_size (NGPU env var).

    In multi_proc_meta mode, world_size = full simulated size (e.g. 64 for PP=4,CP=4,DP=4).
    In fake_backend mode, world_size = NGPU env var (single process simulates all ranks).
    """
    sim_config = getattr(config, "simulation", None)
    if sim_config is not None:
        sim_degrees = getattr(sim_config, "simulated_parallel_degrees", {})
        if sim_degrees and "world_size" in sim_degrees:
            return int(sim_degrees["world_size"])

    # Fall back to computing from parallel degrees
    parallelism = getattr(config, "parallelism", None)
    if parallelism is not None:
        pp = getattr(parallelism, "pipeline_parallel_degree", 1)
        tp = getattr(parallelism, "tensor_parallel_degree", 1)
        cp = getattr(parallelism, "context_parallel_degree", 1)
        ep = getattr(parallelism, "expert_parallel_degree", 1)
        etp = getattr(parallelism, "expert_tensor_parallel_degree", 1)
        dp_replicate = getattr(parallelism, "data_parallel_replicate_degree", 1)
        dp_shard = getattr(parallelism, "data_parallel_shard_degree", 1)
        if dp_shard == -1:
            # Can't compute without knowing world_size; return 0 as signal
            return 0
        return pp * tp * cp * ep * etp * dp_replicate * dp_shard

    return 1
