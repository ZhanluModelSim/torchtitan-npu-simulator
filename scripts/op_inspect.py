#!/usr/bin/env python3
"""Inspect captured L0 op_type distribution to determine whether the simulator
sees aten decompositions or NPU fused ops (the autofusion baseline)."""
import os
import sys
import torch.multiprocessing as mp
from collections import Counter


def _worker(rank, nproc, sim_ws, config_name, extra_args):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(nproc)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
    os.environ["NGPU"] = str(sim_ws)
    os.environ["TORCHTITAN_SIM_WORLD_SIZE"] = str(sim_ws)
    sys.argv = ["torchtitan_npu.entry", "--module", "torchtitan_npu.simulator",
                "--config", config_name, "--training.steps", "1"] + extra_args
    import torchtitan_npu.simulator.trainer as tm
    orig = tm.SimulationTrainer.train

    def patched(self):
        orig(self)
        wg = self.workload_graph
        print(f"\n##### rank {rank} #####", flush=True)
        for tid, sg in getattr(wg, "step_templates", {}).items():
            c = Counter(n.op_type for n in sg.nodes.values())
            print(f"  {tid} ({sg.step_type}) nodes={len(sg.nodes)}:", flush=True)
            for t, n in c.most_common():
                print(f"    {t}: {n}", flush=True)
            # show a few raw_op_type annotations for elementwise/unknown ops
            raws = Counter(n.annotations.get("raw_op_type", "")
                           for n in sg.nodes.values()
                           if n.op_type in ("unknown", "view", "reshape", "transpose", "cat", "split"))
            if raws:
                print(f"    [raw_op_type for view/unknown/etc]:", flush=True)
                for t, n in raws.most_common(12):
                    print(f"      {t}: {n}", flush=True)

    tm.SimulationTrainer.train = patched
    from torchtitan_npu.entry import main as entry_main
    entry_main()


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else "deepseek_v4_pro_simulate_16_layers"
    extra = sys.argv[2:]
    if not any(a.startswith("--hf_assets_path") for a in extra):
        extra += ["--hf_assets_path", "./tests/assets/tokenizer/deepseekv3_tokenizer"]
    nproc = int(os.environ.get("NGPU", "16"))
    sim_ws = int(os.environ.get("TORCHTITAN_SIM_WORLD_SIZE", nproc))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "29501")
    mp.spawn(_worker, args=(nproc, sim_ws, config, extra), nprocs=nproc,
             join=True, start_method="spawn")


if __name__ == "__main__":
    main()
