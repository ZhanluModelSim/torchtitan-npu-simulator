#!/usr/bin/env python3
"""Phase 1: offline GE fusion capture -- run on a REAL NPU machine to produce
the fusion profile that Phase 2 (torchtitan_npu.simulator.ir.ge_fusion) consumes.

WHY OFFLINE
-----------
GE's op fusion happens inside the aclgrph compile pipeline, whose
``build_initialize`` needs the NPU runtime/driver. The simulator runs under
meta-device (no NPU), so it cannot run GE fusion itself. Instead this script
captures GE's compiled (fused) graph on real hardware, and the simulator loads
the resulting profile offline (see validate_ge_fusion.py for the P2 side).

WHAT IT PRODUCES
----------------
1. ``<name>.air``   -- GE's native serialized fused graph (loadable via
   ``ge_fusion.load_fusion_profile_air``). Carries fused-node topology (types
   + data edges) but NOT the original-aten->fused-op attribution.
2. ``<name>.json``  -- the portable fusion profile consumed by
   ``ge_fusion.load_fusion_profile_json``. Each fused node lists its
   ``fused_op_type`` and the captured L0 ``OpNode.seq_idx`` values that merged
   into it (the original->fused mapping).

RUN ON A REAL NPU
-----------------
    NGPU=<n> python3 scripts/ge_fusion_capture.py --config deepseek_v4_pro_simulate_16_layers \
        --hf_assets_path ./tests/assets/tokenizer/deepseekv3_tokenizer \
        --output_dir ./ge_profiles

STATUS: scaffold. The .air topology capture is implemented (mirrors
build_sample_ge_air.py). The original->fused op mapping (``original_op_seq_idxs``
in the JSON) requires GE's debug dump config to record per-op fusion
attribution, which is machine/GE-version specific -- marked TODO below. Until a
real NPU run fills the mapping, the P2 side falls back to a synthesized profile
(see scripts/validate_ge_fusion.py).
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="deepseek_v4_pro_simulate_16_layers")
    ap.add_argument("--hf_assets_path", default="./tests/assets/tokenizer/deepseekv3_tokenizer")
    ap.add_argument("--output_dir", default="./ge_profiles")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. enable GE graph mode (jit_compile=False routes ops through GE) ---
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_compile_mode(jit_compile=False)
    print("[ge_capture] GE graph mode enabled (jit_compile=False)", flush=True)

    # --- 2. build the model forward graph and compile it via offline_compile ---
    # The real path: run one training step under jit_compile=False so torch_npu
    # lowers aten -> GE, then capture the compiled graph. On a NPU machine:
    #   import ge, ge.offline_compile as oc
    #   ge_graph = <the GE graph from torch_npu lowering>  # via torchair/GeModule
    #   oc.build_initialize()
    #   mb = oc.build_model(ge_graph)
    #   oc.save_model(os.path.join(out, f"{name}.om"), mb)
    #   ge_graph.save_to_air(os.path.join(out, f"{name}.air"))
    #   oc.build_finalize()
    raise NotImplementedError(
        "ge_fusion_capture must run on a real NPU machine. See the module "
        "docstring: build the GE graph via torch_npu lowering under "
        "jit_compile=False, then oc.build_model + save_to_air. The "
        "original->fused op mapping (original_op_seq_idxs) needs GE's debug "
        "dump config -- fill the TODO in the JSON emission below."
    )

    # --- 3. emit JSON profile (original->fused mapping) ---
    # TODO: walk the compiled graph + GE's fusion debug dump to record, per
    # fused node, the captured L0 OpNode.seq_idx values that merged into it.
    # Schema consumed by ge_fusion.load_fusion_profile_json:
    profile = {
        "graph_name": args.config,
        "fused_nodes": [
            # {"node_id": 0, "fused_op_type": "AddRmsNorm",
            #  "original_op_seq_idxs": [12, 13, 14], "input_fused_ids": [..],
            #  "output_bytes": <int>},
        ],
    }
    json_path = os.path.join(args.output_dir, f"{args.config}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"[ge_capture] wrote {json_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
