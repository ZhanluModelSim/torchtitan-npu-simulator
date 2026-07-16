#!/usr/bin/env python3
"""Dump one OVERLAP_F_B action's sub_actions (template_ref / stage / mb / comp_type)
and the DataSlots its sub-actions produce/consume, to ground the replay recipe."""
import os
import sys
import torch.multiprocessing as mp


def _worker(rank, config_name, hf, q):
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29514", NGPU="2",
                      TORCHTITAN_SIM_WORLD_SIZE="2")
    sys.argv = ["torchtitan_npu.entry", "--module", "torchtitan_npu.simulator",
                "--config", config_name, "--training.steps", "1"]
    if hf:
        sys.argv += ["--hf_assets_path", hf]
    import torchtitan_npu.simulator.trainer as tm
    orig = tm.SimulationTrainer.train

    def patched(self):
        orig(self)
        plan = self.workload_graph.schedule_plan
        out = {"rank": rank, "overlaps": []}
        for a in plan.actions:
            if a.action_type != "OVERLAP_F_B" or not a.sub_actions:
                continue
            subs = []
            for s in a.sub_actions:
                slots_out = []
                for sid in s.produces + s.consumes:
                    sl = plan.data_slots.get(sid)
                    if sl:
                        slots_out.append({"kind": sl.kind, "s%d->s%d" % (sl.src_stage, sl.dst_stage):
                                          f"mb{sl.mb_idx} {sl.comm_primitive or 'local'} {sl.volume_bytes}B"})
                subs.append({"stage": s.stage, "mb": s.mb_idx, "ct": s.comp_type,
                             "tmpl": s.template_ref, "seq": s.seq_idx,
                             "produces": s.produces, "consumes": s.consumes,
                             "slot_info": slots_out})
            out["overlaps"].append({"action_id": a.action_id, "seq": a.seq_idx, "subs": subs})
            if len(out["overlaps"]) >= 2:
                break
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
        print(json.dumps(q.get(), default=str, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
