#!/usr/bin/env python3
"""Validate the host-only GE-catalog fusion pass on a real captured StepGraph:
1. run the 16-layer simulator (16 ranks) to capture L0 step graphs
2. build_ge_fusion_profile: pure-host pass reproducing GE fusion behavior,
   targeting authentic GE op types (AddRmsNorm/AddLayerNorm/FusedMatMul/
   AdamApplyOneWithDecay/ElementwiseFusion)
3. apply_ge_fusion_profile + fusion_summary -> before/after compression
"""
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
    from torchtitan_npu.simulator.ir.ge_fusion import (
        build_ge_fusion_profile, apply_ge_fusion_profile, fusion_summary,
    )
    orig = tm.SimulationTrainer.train

    def patched(self):
        orig(self)
        if rank != 0:
            return
        wg = self.workload_graph
        print(f"\n##### host-only GE fusion pass (rank {rank}) #####", flush=True)
        for tid, sg in wg.step_templates.items():
            prof = build_ge_fusion_profile(sg)
            apply_ge_fusion_profile(sg, prof)
            s = fusion_summary(sg)
            types = Counter(r.fused_op_type for r in sg.fused_regions if not r.is_unfused)
            print(f"  {tid} ({sg.step_type}): l0_nodes={s['l0_nodes']} -> "
                  f"regions={s['fused_regions']} (fused={s['fused_multi_op_regions']}, "
                  f"unfused={s['unfused_singletons']}) compression={s['compression_ratio']:.1f}x "
                  f"elim={s['eliminated_intermediates_bytes']}B "
                  f"types={dict(types)}", flush=True)

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
