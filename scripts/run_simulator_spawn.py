#!/usr/bin/env python3
"""Launch simulator with mp.spawn instead of torchrun.

Usage:
    python3 scripts/run_simulator_spawn.py \
        --config deepseek_v4_pro_simulate_16_layers_pp4_cp4 \
        --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer

This script:
1. Parses the config to determine PP degree (nproc_per_node) and world_size (NGPU)
2. Spawns PP degree processes using mp.Process with spawn method
3. Each process sets RANK/WORLD_SIZE/LOCAL_RANK env vars
4. Each process calls torchtitan_npu.entry.main() (same as torchrun)
5. WorkloadGraph objects are returned to the main process via mp.Queue

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


def _worker_fn(
    rank: int,
    nproc: int,
    sim_world_size: int,
    config_name: str,
    extra_args: list[str],
    result_queue: object | None = None,
) -> None:
    """Worker function for each spawned process.

    Calls torchtitan_npu.entry.main() by setting sys.argv, same as torchrun.
    After training, puts the WorkloadGraph onto result_queue if provided.
    """
    # Set environment variables that torchrun would normally set
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nproc)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["NGPU"] = str(sim_world_size)
    os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(sim_world_size)

    # Build sys.argv for entry.py's ConfigManager.parse_args()
    argv = [
        "torchtitan_npu.entry",
        "--module", "torchtitan_npu.simulator",
        "--config", config_name,
        "--training.steps", "1",
    ]
    argv.extend(extra_args)
    sys.argv = argv

    # Patch SimulationTrainer.train to capture workload_graph before
    # trainer.close() destroys it. The graph is put onto result_queue.
    if result_queue is not None:
        import torchtitan_npu.simulator.trainer as trainer_mod

        _orig_train = trainer_mod.SimulationTrainer.train

        def _patched_train(self):
            _orig_train(self)
            if self.workload_graph is not None:
                try:
                    result_queue.put({
                        "rank": rank,
                        "workload_graph": self.workload_graph,
                    })
                except Exception:
                    # WorkloadGraph may not be picklable (contains unpicklable
                    # objects like RankTable). Fall back to output_dir path.
                    result_queue.put({
                        "rank": rank,
                        "output_dir": self.simulation_config.output_dir,
                    })

        trainer_mod.SimulationTrainer.train = _patched_train

    # Call entry.main() — same code path as torchrun
    from torchtitan_npu.entry import main as entry_main
    entry_main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch simulator with mp.spawn")
    parser.add_argument(
        "--config", required=True, help="Simulator config function name"
    )
    parser.add_argument(
        "--hf_assets_path", default=None, help="Path to tokenizer assets"
    )
    parser.add_argument(
        "--master_port", type=int, default=29500, help="Master port for gloo"
    )
    args, extra = parser.parse_known_args()

    # Build the config to extract PP degree and world_size
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
        ngpu_env = os.environ.get("NGPU")
        if ngpu_env:
            world_size = int(ngpu_env)
        else:
            print("ERROR: Cannot determine world_size. dp_shard=-1 (auto).")
            print("  Please set NGPU env var.")
            sys.exit(1)

    print(f"[spawn] config={args.config}")
    print(f"[spawn] nproc_per_node (PP degree) = {nproc}")
    print(f"[spawn] world_size (NGPU) = {world_size}")
    print(f"[spawn] comm_mode = {config.comm.mode}")

    # Build extra args for entry.py (space-separated key value pairs)
    extra_args = []
    if args.hf_assets_path:
        extra_args.extend(["--hf_assets_path", args.hf_assets_path])
    extra_args.extend(extra)

    # Set master addr/port in env for workers
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)

    if nproc == 1:
        # Single process, no spawn needed
        print("[spawn] Single process mode (PP=1), running directly")
        _worker_fn(0, 1, world_size, args.config, extra_args)
    else:
        # Spawn nproc processes with a result queue
        import torch.multiprocessing as mp

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        print(f"[spawn] Spawning {nproc} processes...")
        procs = []
        for rank in range(nproc):
            p = ctx.Process(
                target=_worker_fn,
                args=(rank, nproc, world_size, args.config, extra_args, result_queue),
            )
            p.start()
            procs.append(p)

        # Collect results from the queue while processes run
        results = []
        for _ in range(nproc):
            try:
                result = result_queue.get(timeout=600)
                results.append(result)
                print(f"[spawn] Received result from rank {result.get('rank', '?')}")
            except Exception:
                print("[spawn] Warning: timeout waiting for result from a process")

        for p in procs:
            p.join()
        print(f"[spawn] All processes finished")

        # Process collected results
        for r in results:
            if "workload_graph" in r:
                wg = r["workload_graph"]
                print(f"[spawn] rank {r['rank']}: WorkloadGraph {wg.workload_id}, "
                      f"{len(wg.step_templates)} step templates")
            elif "output_dir" in r:
                print(f"[spawn] rank {r['rank']}: output at {r['output_dir']}")

    print("[spawn] Done")


if __name__ == "__main__":
    main()
