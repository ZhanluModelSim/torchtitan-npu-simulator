#!/usr/bin/env python3
"""Launch simulator with mp.spawn instead of torchrun.

Usage:
    python3 scripts/run_simulator_spawn.py \
        --config deepseek_v4_pro_simulate_16_layers_pp4_cp4 \
        --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer

This script:
1. Parses the config to determine PP degree (nproc_per_node)
2. Parses the config to determine full world_size (NGPU)
3. Uses mp.spawn to launch nproc_per_node processes
4. Each process sets RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT env vars
5. Each process runs torchtitan_npu.entry main()

Equivalent to:
    NGPU=64 torchrun --nproc_per_node=4 -m torchtitan_npu.entry \
        --module torchtitan_npu.simulator \
        --config deepseek_v4_pro_simulate_16_layers_pp4_cp4 \
        --training.steps=1 \
        --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer
"""

import argparse
import os
import sys


def _worker_fn(rank: int, nproc: int, sim_world_size: int, config_name: str, extra_args: list[str]) -> None:
    """Worker function for each spawned process."""
    # Set environment variables that torchrun would normally set
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nproc)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["NGPU"] = str(sim_world_size)
    os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(sim_world_size)

    # Build config directly (bypass entry.py arg parsing)
    from torchtitan_npu.simulator.config_registry import __dict__ as registry
    config_fn = registry[config_name]
    config = config_fn()

    # Apply extra args (e.g. hf_assets_path)
    for arg in extra_args:
        if arg.startswith("--hf_assets_path="):
            config.hf_assets_path = arg.split("=", 1)[1]

    # Run trainer
    from torchtitan_npu.simulator.trainer import SimulationTrainer
    trainer = SimulationTrainer(config)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch simulator with mp.spawn")
    parser.add_argument(
        "--config", required=True, help="Simulator config function name"
    )
    parser.add_argument(
        "--module", default="torchtitan_npu.simulator", help="Module name"
    )
    parser.add_argument("--training.steps", type=int, default=1, help="Training steps")
    parser.add_argument(
        "--hf_assets_path", default=None, help="Path to tokenizer assets"
    )
    parser.add_argument(
        "--master_port", type=int, default=29500, help="Master port for gloo"
    )
    args, extra = parser.parse_known_args()

    # Build the config to extract PP degree and world_size
    # We need to import the config registry and call the config function
    config_module = __import__(
        "torchtitan_npu.simulator.config_registry",
        fromlist=[args.config],
    )
    config_fn = getattr(config_module, args.config)
    config = config_fn()

    # Extract PP degree (nproc_per_node) and world_size (NGPU)
    from torchtitan_npu.simulator.utils import get_nproc_per_node, get_world_size

    nproc = get_nproc_per_node(config)
    world_size = get_world_size(config)

    if world_size == 0:
        # dp_shard=-1 (auto), compute from parallel degrees
        parallelism = config.parallelism
        pp = parallelism.pipeline_parallel_degree
        tp = parallelism.tensor_parallel_degree
        cp = parallelism.context_parallel_degree
        ep = parallelism.expert_parallel_degree
        etp = parallelism.expert_tensor_parallel_degree
        dp_replicate = parallelism.data_parallel_replicate_degree
        # dp_shard = world_size / (pp * tp * cp * ep * etp * dp_replicate)
        # But we don't know world_size... use NGPU env or ask user
        ngpu_env = os.environ.get("NGPU")
        if ngpu_env:
            world_size = int(ngpu_env)
        else:
            print(f"ERROR: Cannot determine world_size. dp_shard=-1 (auto).")
            print(f"  Please set NGPU env var or pass --world_size")
            sys.exit(1)

    print(f"[spawn] config={args.config}")
    print(f"[spawn] nproc_per_node (PP degree) = {nproc}")
    print(f"[spawn] world_size (NGPU) = {world_size}")
    print(f"[spawn] comm_mode = {config.comm.mode}")

    # Build extra args (hf_assets_path etc.)
    extra_args = []
    if args.hf_assets_path:
        extra_args.append(f"--hf_assets_path={args.hf_assets_path}")

    if nproc == 1:
        # Single process, no spawn needed
        print(f"[spawn] Single process mode (PP=1), running directly")
        _worker_fn(0, 1, world_size, args.config, extra_args)
    else:
        # Spawn nproc processes
        import torch.multiprocessing as mp

        print(f"[spawn] Spawning {nproc} processes...")
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(nproc):
            p = ctx.Process(
                target=_worker_fn,
                args=(rank, nproc, world_size, args.config, extra_args),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        print(f"[spawn] All processes finished")


if __name__ == "__main__":
    main()
