#!/usr/bin/env python3
"""Launch simulator with mp.spawn instead of torchrun.

Usage:
    python3 scripts/run_simulator_spawn.py \
        --config deepseek_v4_pro_baseline_bf16 \
        --parallelism.pipeline-parallel-degree 4 \
        --training.num-mtp-modules 0 \
        --simulation.world-size 256

This script:
1. Parses the config to determine PP degree (nproc_per_node) and world_size (NGPU)
2. Spawns PP degree processes using mp.Process with spawn method
3. Each process sets RANK/WORLD_SIZE/LOCAL_RANK env vars
4. Each process calls torchtitan_npu.entry.main() (same as torchrun)
5. Each process writes its rank-local graph and returns a completion record

Equivalent to:
    NGPU=256 torchrun --nproc_per_node=4 -m torchtitan_npu.entry \
        --module torchtitan_npu.simulator \
        --config deepseek_v4_pro_baseline_bf16 \
        --parallelism.pipeline-parallel-degree 4 \
        --training.num-mtp-modules 0 \
        --training.steps=1 \
        --simulation.world-size 256
"""

import argparse
import os
import queue
import sys
import time


_WORKER_TIMEOUT_SECONDS = 600


def _worker_failures(processes: list[object]) -> list[tuple[int, int]]:
    """Return ``(rank, exitcode)`` for workers that exited unsuccessfully."""
    return [
        (rank, process.exitcode)
        for rank, process in enumerate(processes)
        if process.exitcode not in (None, 0)
    ]


def _format_worker_failures(failures: list[tuple[int, int]]) -> str:
    return ", ".join(f"rank {rank} (exitcode {exitcode})" for rank, exitcode in failures)


def _collect_worker_results(
    result_queue: object,
    processes: list[object],
    expected_count: int,
    timeout: float = _WORKER_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    """Collect lightweight completion records while monitoring worker health."""
    deadline = time.monotonic() + timeout
    results: list[dict[str, object]] = []

    while len(results) < expected_count:
        failures = _worker_failures(processes)
        if failures:
            raise RuntimeError(
                "simulator worker failed: " + _format_worker_failures(failures)
            )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out after {timeout:g}s waiting for simulator workers "
                f"({len(results)}/{expected_count} results received)"
            )

        try:
            result = result_queue.get(timeout=min(0.5, remaining))
        except queue.Empty:
            if all(process.exitcode is not None for process in processes):
                break
            continue

        if not isinstance(result, dict):
            raise RuntimeError(
                f"simulator worker returned an invalid result: {result!r}"
            )
        results.append(result)

    return results


def _validate_worker_results(
    results: list[dict[str, object]],
    processes: list[object],
    expected_count: int,
) -> list[dict[str, object]]:
    """Verify that every PP rank completed capture and export exactly once."""
    failures = _worker_failures(processes)
    if failures:
        raise RuntimeError(
            "simulator worker failed: " + _format_worker_failures(failures)
        )

    ranks = [result.get("rank") for result in results]
    invalid_ranks = [rank for rank in ranks if not isinstance(rank, int)]
    if invalid_ranks:
        raise RuntimeError(f"worker results contain invalid ranks: {invalid_ranks}")

    expected_ranks = set(range(expected_count))
    actual_ranks = set(ranks)
    duplicate_ranks = sorted(rank for rank in actual_ranks if ranks.count(rank) > 1)
    missing_ranks = sorted(expected_ranks - actual_ranks)
    unexpected_ranks = sorted(actual_ranks - expected_ranks)
    if duplicate_ranks or missing_ranks or unexpected_ranks:
        raise RuntimeError(
            "incomplete simulator worker results: "
            f"missing={missing_ranks}, duplicate={duplicate_ranks}, "
            f"unexpected={unexpected_ranks}"
        )

    for result in results:
        rank = result["rank"]
        template_count = result.get("step_template_count")
        if not isinstance(template_count, int) or template_count <= 0:
            raise RuntimeError(
                f"rank {rank} captured no step templates: {template_count!r}"
            )

        template_ids = result.get("step_template_ids")
        if not isinstance(template_ids, list) or not all(
            isinstance(template_id, str) for template_id in template_ids
        ):
            raise RuntimeError(
                f"rank {rank} returned invalid step template IDs: {template_ids!r}"
            )
        required_templates = {f"s{rank}_F", f"s{rank}_B"}
        missing_templates = sorted(required_templates - set(template_ids))
        if missing_templates:
            raise RuntimeError(
                f"rank {rank} capture is incomplete; missing step templates "
                f"{missing_templates}"
            )

        output_dir = result.get("output_dir")
        if not isinstance(output_dir, str) or not os.path.isdir(output_dir):
            raise RuntimeError(
                f"rank {rank} output directory is missing: {output_dir!r}"
            )

    return sorted(results, key=lambda result: result["rank"])


def _terminate_workers(processes: list[object]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


def _join_workers(
    processes: list[object], timeout: float = _WORKER_TIMEOUT_SECONDS
) -> None:
    """Join all workers within one shared deadline."""
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))

    alive_ranks = [
        rank for rank, process in enumerate(processes) if process.is_alive()
    ]
    if alive_ranks:
        raise TimeoutError(
            f"timed out after {timeout:g}s waiting for worker ranks "
            f"{alive_ranks} to exit"
        )


def _build_entry_args(
    config_name: str,
    hf_assets_path: str | None,
    extra_args: list[str],
) -> list[str]:
    args = [
        "--module",
        "torchtitan_npu.simulator",
        "--config",
        config_name,
        "--training.steps",
        "1",
    ]
    if hf_assets_path:
        args.extend(["--hf_assets_path", hf_assets_path])
    args.extend(extra_args)
    return args


def _resolve_launch_config(entry_args: list[str]):  # noqa: ANN202
    """Parse the exact worker CLI before deciding how many workers to spawn."""
    from torchtitan.config import ConfigManager
    from torchtitan_npu.simulator.utils import (
        resolve_simulation_runtime_from_environment,
    )

    config = ConfigManager().parse_args(entry_args)
    runtime = resolve_simulation_runtime_from_environment(config, cli_args=entry_args)
    return config, runtime


def _worker_fn(
    rank: int,
    nproc: int,
    sim_world_size: int,
    entry_args: list[str],
    result_queue: object | None = None,
) -> None:
    """Worker function for each spawned process.

    Calls torchtitan_npu.entry.main() by setting sys.argv, same as torchrun.
    After training, puts a lightweight completion record onto result_queue.
    """
    # Set environment variables that torchrun would normally set
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nproc)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29500")
    os.environ["NGPU"] = str(sim_world_size)
    os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(sim_world_size)

    # Use the same CLI that the parent resolved before spawning.
    sys.argv = ["torchtitan_npu.entry", *entry_args]

    # Patch SimulationTrainer.train to report capture/export completion before
    # trainer.close() destroys the graph. Keep the queue payload lightweight:
    # multiprocessing.Queue serializes in a background thread, so attempting
    # to send WorkloadGraph directly cannot provide a reliable fallback.
    if result_queue is not None:
        import torchtitan_npu.simulator.trainer as trainer_mod

        _orig_train = trainer_mod.SimulationTrainer.train

        def _patched_train(self):
            _orig_train(self)
            if self.workload_graph is None:
                raise RuntimeError(f"rank {rank} finished without a WorkloadGraph")

            output_dir = os.path.abspath(
                os.path.join(self.simulation_config.output_dir, f"rank_{rank}")
            )
            result_queue.put({
                "rank": rank,
                "workload_id": str(self.workload_graph.workload_id),
                "step_template_count": len(self.workload_graph.step_templates),
                "step_template_ids": sorted(self.workload_graph.step_templates),
                "output_dir": output_dir,
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

    entry_args = _build_entry_args(args.config, args.hf_assets_path, extra)
    _config, runtime = _resolve_launch_config(entry_args)
    nproc = runtime.pp_degree
    world_size = runtime.world_size

    print(f"[spawn] config={args.config}")
    print(f"[spawn] nproc_per_node (PP degree) = {nproc}")
    print(f"[spawn] world_size (NGPU) = {world_size}")
    print(f"[spawn] comm_mode = {runtime.comm_mode}")
    print(f"[spawn] parallel_degrees = {runtime.parallel_degrees}")

    # Set master addr/port in env for workers
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(args.master_port)

    if nproc == 1:
        # Single process, no spawn needed
        print("[spawn] Single process mode (PP=1), running directly")
        _worker_fn(0, 1, world_size, entry_args)
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
                args=(rank, nproc, world_size, entry_args, result_queue),
            )
            p.start()
            procs.append(p)

        completed = False
        try:
            results = _collect_worker_results(result_queue, procs, nproc)
            _join_workers(procs)
            completed = True
        finally:
            if not completed:
                _terminate_workers(procs)

        results = _validate_worker_results(results, procs, nproc)
        print(f"[spawn] All {nproc} processes finished successfully")
        for result in results:
            print(
                f"[spawn] rank {result['rank']}: WorkloadGraph "
                f"{result['workload_id']}, {result['step_template_count']} "
                f"step templates, output at {result['output_dir']}"
            )

    print("[spawn] Done")


if __name__ == "__main__":
    main()
