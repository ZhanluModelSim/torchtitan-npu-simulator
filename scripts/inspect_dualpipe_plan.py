"""Inspect DualPipeV's pipeline_order_with_comms to verify OVERLAP_F_B and
V-schedule local transfers (set_local_fwd_input — no P2P)."""
import os
import sys
import torch.multiprocessing as mp
from collections import Counter


def _worker(rank, nproc, sim_ws, config_name, extra_args, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(nproc), LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29510", NGPU=str(sim_ws),
                      TORCHTITAN_SIM_WORLD_SIZE=str(sim_ws))
    sys.argv = ["torchtitan_npu.entry", "--module", "torchtitan_npu.simulator",
                "--config", config_name, "--training.steps", "1"] + extra_args
    import torchtitan_npu.simulator.trainer as tm
    orig_init = tm.SimulationTrainer.__init__

    def patched_init(self, config):
        orig_init(self, config)
        sched = getattr(self, "pp_schedule", None)
        if sched is not None and getattr(sched, "pipeline_order_with_comms", None):
            q.put({"rank": rank, "plan": _serialize(sched.pipeline_order_with_comms)})

    tm.SimulationTrainer.__init__ = patched_init
    try:
        from torchtitan_npu.entry import main as entry_main
        entry_main()
    except Exception as e:
        q.put({"rank": rank, "error": repr(e)})


def _serialize(plan):
    out = {}
    for r, actions in plan.items():
        out[str(r)] = [repr(a) for a in actions]
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hf_assets_path", default=None)
    args = p.parse_known_args()[0]
    cfg_mod = __import__("torchtitan_npu.simulator.config_registry", fromlist=[args.config])
    cfg = getattr(cfg_mod, args.config)()
    from torchtitan_npu.simulator.utils import get_nproc_per_node, get_world_size
    nproc = get_nproc_per_node(cfg); sim_ws = get_world_size(cfg)
    extra = ["--training.steps", "1"]
    if args.hf_assets_path:
        extra += ["--hf_assets_path", args.hf_assets_path]
    os.environ["MASTER_ADDR"] = "127.0.0.1"; os.environ["MASTER_PORT"] = "29510"
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, nproc, sim_ws, args.config, extra, q)) for r in range(nproc)]
    for p in procs: p.start()
    for p in procs: p.join()
    results = {}
    while not q.empty():
        r = q.get()
        results[r["rank"]] = r
    for rank in sorted(results):
        r = results[rank]
        if "error" in r:
            print(f"rank {rank} ERROR: {r['error'][:200]}"); continue
        plan = r["plan"]
        print(f"\n===== rank {rank} pipeline_order_with_comms =====")
        ctr = Counter()
        for a in plan[str(rank)]:
            # action repr like "0F1" / "(1F0;3B0)OVERLAP_F_B" / "1UNSHARD"
            if "OVERLAP" in a: ctr["OVERLAP_F_B"] += 1
            elif "UNSHARD" in a: ctr["UNSHARD"] += 1
            elif "RESHARD" in a: ctr["RESHARD"] += 1
            elif "REDUCE_GRAD" in a: ctr["REDUCE_GRAD"] += 1
            elif "SEND_F" in a: ctr["SEND_F"] += 1
            elif "RECV_F" in a: ctr["RECV_F"] += 1
            elif "SEND_B" in a: ctr["SEND_B"] += 1
            elif "RECV_B" in a: ctr["RECV_B"] += 1
            elif a and a[0].isdigit():
                ct = a[1:].rstrip("0123456789")
                ctr[ct] += 1
        print("action type counts:", dict(ctr))
        print("first 25 actions:", plan[str(rank)][:25])


if __name__ == "__main__":
    main()
