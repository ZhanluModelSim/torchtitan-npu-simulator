#!/usr/bin/env python3
"""Inspect the relationship between UNSHARD/RESHARD plan actions and the
s-1_F (unattributed) L1 template: do the unshard/reshard L0 comm ops land
in s-1_F, and do the plan actions reference them?"""
import os
import sys
import torch.multiprocessing as mp
from collections import Counter


def _worker(rank, config_name, hf, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE="4", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29513", NGPU="4",
                      TORCHTITAN_SIM_WORLD_SIZE="4")
    sys.argv = ["torchtitan_npu.entry", "--module", "torchtitan_npu.simulator",
                "--config", config_name, "--training.steps", "1"]
    if hf:
        sys.argv += ["--hf_assets_path", hf]
    import torchtitan_npu.simulator.trainer as tm
    orig = tm.SimulationTrainer.train

    def patched(self):
        orig(self)
        wg = self.workload_graph
        plan = wg.schedule_plan
        out = {"rank": rank}
        # 1. s-1_F template contents
        sg = wg.step_templates.get("s-1_F")
        if sg:
            out["s-1_F_nodes"] = len(sg.nodes)
            out["s-1_F_raw"] = Counter(n.annotations.get("raw_op_type", "")
                                       for n in sg.nodes.values())
        # 2. UNSHARD/RESHARD actions
        out["unshard_actions"] = []
        out["reshard_actions"] = []
        for a in plan.actions:
            if a.action_type in ("UNSHARD", "RESHARD"):
                out[f"{a.action_type.lower()}_actions"].append({
                    "id": a.action_id, "stage": a.stage, "template_ref": a.template_ref,
                    "comm_op_id": a.comm_op_id, "is_noop": a.is_noop,
                    "produces": a.produces, "consumes": a.consumes, "seq": a.seq_idx,
                })
        # 3. the param_full/param_shard DataSlots these produce/consume
        out["fsdp_slots"] = []
        for sid in set(s for a in plan.actions for s in (a.produces + a.consumes)):
            s = plan.data_slots.get(sid)
            if s and s.kind in ("param_full", "param_shard"):
                out["fsdp_slots"].append({
                    "id": sid, "kind": s.kind, "stage_src": s.src_stage,
                    "shape": list(s.shape), "comm": s.comm_primitive,
                    "src_exit_op": s.src_exit_op, "dst_entry_op": s.dst_entry_op,
                    "producer": s.producer_action_id, "consumers": s.consumer_action_ids,
                })
        q.put(out)

    tm.SimulationTrainer.train = patched
    from torchtitan_npu.entry import main
    main()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hf_assets_path", default=None)
    a = p.parse_args()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(r, a.config, a.hf_assets_path, q)) for r in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    import json
    while not q.empty():
        r = q.get()
        print(json.dumps(r, default=str, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
