#!/usr/bin/env python3
"""Prove pure-host GE fusion: build the UNFUSED pattern Add(residual)+LayerNorm,
run ge.offline_compile.build_model host-only (soc_version, no NPU), with
DUMP_GE_GRAPH=1 so GE emits the per-stage graph dumps. If GE fuses Add+LayerNorm
into AddLayerNorm, a later dump stage will show an AddLayerNorm node replacing
the separate Add+LayerNorm -- proving real GE fusion runs without NPU hardware."""
import os
import glob
import ge.es as es
import ge.offline_compile as oc
from ge.graph import DataType as DT

es.load_all_plugins()

builder = es.GraphBuilder("fusetest")
x = builder.create_input(0, data_type=DT.DT_FLOAT, shape=[4, 8])
resid = builder.create_const_float([4, 8])
gamma = builder.create_const_float([8])
beta = builder.create_const_float([8])
add = es.math.Add(x, resid)                 # residual add (UNFUSED)
ln = es.nn.LayerNorm(add, gamma, beta, begin_norm_axis=-1, epsilon=1e-5)
ln_out = getattr(ln, "y", None) or getattr(ln, "x", None) or ln
builder.set_graph_output(ln_out, 0)
g = builder.build_and_reset()

print("BEFORE compile:", [(n.name, n.type) for n in g.get_all_nodes()], flush=True)
dumpdir = "/tmp/ge_fuse_dumps"
os.makedirs(dumpdir, exist_ok=True)
oc.build_initialize({"ge.socVersion": "Ascend910B"})
try:
    mb = oc.build_model(g, {})
    print("build_model OK", flush=True)
except Exception as ex:
    print("build_model FAIL:", ex, flush=True)
finally:
    oc.build_finalize()


def stage_ops(path):
    ops = []
    with open(path, encoding="utf-8") as f:
        cur = {}
        for line in f:
            line = line.strip()
            if line.startswith("name: "):
                cur["name"] = line.split('"', 2)[1]
            elif line.startswith("type: "):
                cur["type"] = line.split('"', 2)[1]
                if cur["type"] not in ("Data", "Const", "NetOutput"):
                    ops.append(cur)
                cur = {}
    return ops


stages = sorted(glob.glob(f"{dumpdir}/pid_*/*.txt"))
print(f"\n=== {len(stages)} dump stages ===", flush=True)
for p in stages:
    ops = stage_ops(p)
    types = [o["type"] for o in ops]
    fname = os.path.basename(p)
    marker = "  <-- FUSED" if "AddLayerNorm" in types else ""
    print(f"  {fname}: {types}{marker}", flush=True)
