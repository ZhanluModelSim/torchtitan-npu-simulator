#!/usr/bin/env python3
"""Build a realistic sample GE graph containing fused ops (AddRmsNorm,
FusedMatMul) + elementwise, then save_to_air. This .air is the sample fusion
profile artifact the P2 loader consumes (simulating what P1 offline-captures
on a real NPU). Also prints the node structure that the loader extracts."""
import os
import ge.es as es
from ge.graph import DataType as DT

es.load_all_plugins()

builder = es.GraphBuilder("sample_fused")
# Data input [4,8] float
x = builder.create_input(0, data_type=DT.DT_FLOAT, shape=[4, 8])
w1 = builder.create_const_float([8, 8])  # matmul weight
gamma = builder.create_const_float([8])   # rmsnorm weight
w2 = builder.create_const_float([8, 8])
bias = builder.create_const_float([8])

# MatMulV3(x, w1) -> matmul node
mm = es.nn.MatMulV3(x, w1)
# probe what AddRmsNorm returns
arn = es.nn.AddRmsNorm(x, mm, gamma, epsilon=1e-6)
print(f"AddRmsNorm returned: {type(arn).__name__}, dir={[a for a in dir(arn) if not a.startswith('_')][:15]}", flush=True)
# try common access patterns
norm_out = None
for attr in ("out", "y", "output", "result"):
    if hasattr(arn, attr):
        norm_out = getattr(arn, attr); print(f"  accessed via .{attr}", flush=True); break
if norm_out is None:
    # maybe tuple/list
    try:
        norm_out = arn[0]; print(f"  accessed via [0]", flush=True)
    except Exception:
        norm_out = arn; print(f"  using object directly", flush=True)
# FusedMatMul(norm_out, w2, bias=bias) -> fused matmul+bias
fm = es.nn.FusedMatMul(norm_out, w2, bias=bias)
builder.set_graph_output(fm, 0)
g = builder.build_and_reset()

nodes = g.get_all_nodes()
print(f"\n=== sample graph: {len(nodes)} nodes ===", flush=True)
for n in nodes:
    ins = []
    for i in range(n.get_inputs_size()):
        try:
            p = n.get_in_data_nodes_and_port_indexes(i)
            ins.append((p[0].name, p[1]))
        except RuntimeError:
            pass
    outs = []
    for i in range(n.get_outputs_size()):
        try:
            for p in n.get_out_data_nodes_and_port_indexes(i):
                outs.append((p[0].name, p[1]))
        except RuntimeError:
            pass
    print(f"  {n.name} type={n.type} in={ins} out={outs}", flush=True)

air = os.path.join(os.path.dirname(__file__), "..", "tests", "assets", "ge_fusion_sample.air")
air = os.path.abspath(air)
os.makedirs(os.path.dirname(air), exist_ok=True)
g.save_to_air(air)
print(f"\nsaved -> {air} ({os.path.getsize(air)} bytes)", flush=True)
