#!/usr/bin/env python3
"""Validate the P2 fusion pipeline on a real captured StepGraph:
1. run the 16-layer simulator (16 ranks) to capture L0 step graphs
2. synthesize a *stand-in* GE fusion profile (real profile comes from P1 /
   scripts/ge_fusion_capture.py on a real NPU) by grouping consecutive
   elementwise/shape aten primitives between anchor ops
3. apply_ge_fusion_profile + fusion_summary -> before/after compression
"""
import os
import sys
import torch.multiprocessing as mp

# GE fusion anchors (kept as own kernel); everything else is fusible
ANCHORS = {
    "matmul", "bmm", "flash_attention_fwd", "sdpa", "rms_norm", "layer_norm",
    "rope", "softmax", "adamw_step", "allreduce", "allgather", "reduce_scatter",
    "all_to_all", "p2p_send", "p2p_recv", "moe_token_permute", "moe_token_unpermute",
}
FUSIBLE = {"unknown", "view", "reshape", "transpose", "cat", "split", "gelu",
           "silu", "swiglu", "moe_re_routing", "einsum"}


def _synthesize_profile(step_graph, GEFusionProfile, FusedNode):
    """P1 stand-in: emit a plausible GE fusion profile by grouping runs of
    fusible aten primitives between anchor ops. NOT real GE fusion -- only
    exercises the P2 consume pipeline until a real NPU profile exists."""
    nodes = sorted(step_graph.nodes.values(), key=lambda n: n.seq_idx)
    fused = []
    run = []
    for n in nodes:
        if n.op_type in ANCHORS:
            if run:
                fused.append(_flush(run, fused, FusedNode)); run = []
            fused.append(FusedNode(node_id=len(fused), fused_op_type=n.op_type,
                                   original_op_seq_idxs=[n.seq_idx]))
        elif n.op_type in FUSIBLE:
            run.append(n.seq_idx)
        else:
            if run:
                fused.append(_flush(run, fused, FusedNode)); run = []
            fused.append(FusedNode(node_id=len(fused), fused_op_type=n.op_type,
                                   original_op_seq_idxs=[n.seq_idx]))
    if run:
        fused.append(_flush(run, fused, FusedNode))
    return GEFusionProfile(graph_name=step_graph.step_id, fused_nodes=fused)


def _flush(run, fused, FusedNode):
    return FusedNode(node_id=len(fused), fused_op_type="ElementwiseFusion",
                     original_op_seq_idxs=list(run))


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
    from torchtitan_npu.simulator.ir.ge_fusion import (
        GEFusionProfile, FusedNode, apply_ge_fusion_profile, fusion_summary,
    )
    orig = tm.SimulationTrainer.train

    def patched(self):
        orig(self)
        if rank != 0:
            return
        wg = self.workload_graph
        print(f"\n##### GE fusion profile validation (rank {rank}) #####", flush=True)
        for tid, sg in wg.step_templates.items():
            prof = _synthesize_profile(sg, GEFusionProfile, FusedNode)
            apply_ge_fusion_profile(sg, prof)
            s = fusion_summary(sg)
            print(f"  {tid} ({sg.step_type}): l0_nodes={s['l0_nodes']} -> "
                  f"regions={s['fused_regions']} (fused_multi={s['fused_multi_op_regions']}, "
                  f"unfused={s['unfused_singletons']}) compression={s['compression_ratio']:.1f}x "
                  f"eliminated_intermediates={s['eliminated_intermediates_bytes']}B "
                  f"fused_types={sorted(s['fused_op_types'])}", flush=True)

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
