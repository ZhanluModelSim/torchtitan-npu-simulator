#!/usr/bin/env python3
"""Run DualPipeV (or any multi-proc config) with csv output enabled, so
schedule_plan.csv is written per rank. Used to verify the L2 SchedulePlan CSV export."""
import os
import sys
import torch.multiprocessing as mp


def _worker(rank, config_name, hf):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29512", NGPU="2",
                      TORCHTITAN_SIM_WORLD_SIZE="2")
    sys.argv = ["torchtitan_npu.entry", "--module", "torchtitan_npu.simulator",
                "--config", config_name, "--training.steps", "1"]
    if hf:
        sys.argv += ["--hf_assets_path", hf]
    import torchtitan_npu.simulator.config_registry as cr
    orig = getattr(cr, config_name)

    def wrapped():
        c = orig()
        c.simulation.output_formats = ["csv"]
        c.simulation.output_dir = "./simulator_output/_plan_dp"
        return c
    setattr(cr, config_name, wrapped)
    from torchtitan_npu.entry import main
    main()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hf_assets_path", default=None)
    a = p.parse_args()
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=_worker, args=(r, a.config, a.hf_assets_path)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


if __name__ == "__main__":
    main()
