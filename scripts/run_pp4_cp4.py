#!/usr/bin/env python3
"""Run DeepSeek V4 PP4+CP4 simulation with output formats enabled so the
output_dir is populated (the base config leaves output_formats empty)."""
import os
import sys
import torch.multiprocessing as mp


def _worker(rank, nproc, sim_ws):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nproc)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29503")
    os.environ["NGPU"] = str(sim_ws)
    os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(sim_ws)
    sys.argv = [
        "torchtitan_npu.entry", "--module", "torchtitan_npu.simulator",
        "--config", "deepseek_v4_pro_simulate_16_layers_pp4_cp4",
        "--training.steps", "1",
        "--hf_assets_path", "./tests/assets/tokenizer/deepseekv3_tokenizer",
    ]
    import torchtitan_npu.simulator.trainer as tm
    orig_build = tm.SimulationTrainerConfig.build

    def patched_build(self, *a, **kw):
        trainer = orig_build(self, *a, **kw)
        trainer.simulation_config.output_formats = ["csv", "text"]
        return trainer

    tm.SimulationTrainerConfig.build = patched_build
    from torchtitan_npu.entry import main as entry_main
    entry_main()


def main():
    nproc = 4
    sim_ws = 64
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29503"
    mp.spawn(_worker, args=(nproc, sim_ws), nprocs=nproc,
             join=True, start_method="spawn")


if __name__ == "__main__":
    main()
