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
        plan = wg.schedule_plan
        if plan is not None:
            print("  comm_events_summary:", plan.annotations.get("comm_events_summary"), flush=True)
        if plan is not None:
            at = Counter(a.action_type for a in plan.actions)
            # flatten OVERLAP_F_B sub_actions into the count
            at_flat = Counter()
            for a in plan.actions:
                if a.action_type == "OVERLAP_F_B" and a.sub_actions:
                    at_flat["OVERLAP_F_B"] += 1
                    for s in a.sub_actions:
                        at_flat[f"{s.action_type}({s.comp_type})"] += 1
                else:
                    at_flat[a.action_type] += 1
            slot_kinds = Counter(s.kind for s in plan.data_slots.values())
            local = sum(1 for s in plan.data_slots.values() if s.is_local_transfer)
            p2p = sum(1 for s in plan.data_slots.values() if s.comm_primitive == "p2p_send")
            print(f"plan actions: {len(plan.actions)} types={dict(at)}", flush=True)
            print(f"  flattened: {dict(at_flat)}", flush=True)
            print(f"  data_slots: {len(plan.data_slots)} kinds={dict(slot_kinds)} "
                  f"local={local} p2p={p2p}", flush=True)
            # P2P comm actions: direct CommDetail field (no 2-hop lookup)
            for a in plan.actions:
                if a.action_type in ("SEND_F", "RECV_F", "SEND_B", "RECV_B") and a.comm is not None:
                    c = a.comm
                    print(f"    {a.action_type} s{a.stage} mb{a.mb_idx} seq={a.seq_idx} "
                          f"comm={c.primitive}/{c.role} {c.volume_bytes}B "
                          f"s{c.src_stage}->s{c.dst_stage} peer={c.peer_rank} "
                          f"op_id={c.comm_op_id} slot={c.slot_id}", flush=True)
            # UNSHARD/RESHARD action linkage
            for a in plan.actions:
                if a.action_type in ("UNSHARD", "RESHARD"):
                    op = plan.find_op_node(a.comm_op_id) if a.comm_op_id else None
                    opinfo = ""
                    if op is not None:
                        m = op.outputs[0] if op.outputs else None
                        opinfo = (f" -> OpNode#{op.op_id} raw={op.annotations.get('raw_op_type','')} "
                                  f"shape={list(m.shape) if m else []} comm_bytes={op.comm_bytes}")
                    print(f"    {a.action_type} s{a.stage} seq={a.seq_idx} "
                          f"template_ref={a.template_ref or '(none)'} "
                          f"comm_op_id={a.comm_op_id} is_noop={a.is_noop}{opinfo}", flush=True)
            # sample a few SEND_F / local activation slots
            for s in list(plan.data_slots.values())[:4]:
                tag = "LOCAL" if s.is_local_transfer else s.comm_primitive
                print(f"    {s.slot_id}: {s.kind} s{s.src_stage}→s{s.dst_stage} mb{s.mb_idx} "
                      f"{tag} bytes={s.volume_bytes} prod={s.producer_action_id[:10]} "
                      f"cons={s.consumer_action_ids[:1]}", flush=True)
        for tid, sg in wg.step_templates.items():
            print(f"  {tid}: step_type={sg.step_type} nodes={len(sg.nodes)}", flush=True)
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
