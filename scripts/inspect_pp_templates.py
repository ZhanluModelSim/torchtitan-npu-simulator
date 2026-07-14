#!/usr/bin/env python3
"""Per-rank template inspector: runs the simulator with mp.spawn (spawn
context) and prints each rank's captured step_templates, instance counts, and
timeline comp_type distribution directly from inside each worker (the
WorkloadGraph is not picklable across the queue)."""
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
        sch = wg.iteration.schedule
        ct = Counter(e.comp_type for e in sch.execution_timeline)
        print(f"\n##### rank {rank} #####", flush=True)
        print("step_templates:", list(wg.step_templates.keys()), flush=True)
        for tid, sg in wg.step_templates.items():
            hist = Counter(n.annotations.get("raw_op_type", "").split(".")[0]
                            if "." in n.annotations.get("raw_op_type", "")
                            else n.annotations.get("op_type", "") for n in sg.nodes.values())
            stages = Counter(n.annotations.get("pp_stage", -1) for n in sg.nodes.values())
            print(f"  {tid}: step_type={sg.step_type} nodes={len(sg.nodes)} "
                  f"pp_stage_dist={dict(stages)} top_kinds={dict(hist.most_common(5))}", flush=True)
        print("instances:", [(i.instance_id, i.comp_type, i.micro_batch_idx, i.pipeline_stage)
                             for i in sch.instances], flush=True)
        print("timeline total:", len(sch.execution_timeline), "comp_type dist:", dict(ct), flush=True)

    tm.SimulationTrainer.train = patched
    from torchtitan_npu.entry import main as entry_main
    entry_main()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hf_assets_path", default=None)
    args, extra = p.parse_known_args()
    cfg_mod = __import__("torchtitan_npu.simulator.config_registry", fromlist=[args.config])
    cfg = getattr(cfg_mod, args.config)()
    from torchtitan_npu.simulator.utils import get_nproc_per_node, get_world_size
    nproc = get_nproc_per_node(cfg)
    sim_ws = get_world_size(cfg)
    extra_args = ["--training.steps", "1"]
    if args.hf_assets_path:
        extra_args += ["--hf_assets_path", args.hf_assets_path]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    if nproc == 1:
        _worker(0, 1, sim_ws, args.config, extra_args)
        return
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(nproc):
        p = ctx.Process(target=_worker, args=(rank, nproc, sim_ws, args.config, extra_args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
