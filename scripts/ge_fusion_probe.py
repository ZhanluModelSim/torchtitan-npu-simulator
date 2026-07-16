#!/usr/bin/env python3
"""Probe GE graph build + dump/load roundtrip (no NPU needed) to verify the
simulator can serialize/load a pre-captured GE graph as a fusion profile."""
import os
import ge.es as es
from ge.graph import Graph, DataType as DT

es.load_all_plugins()

builder = es.GraphBuilder("probe")
x = builder.create_input(0, data_type=DT.DT_FLOAT, shape=[4, 4])
y = x.mul(builder.create_scalar_float(2.0)).add(builder.create_scalar_float(1.0))
builder.set_graph_output(y, 0)
g = builder.build_and_reset()
print(f"built graph: {len(g.get_all_nodes())} nodes", flush=True)
for n in g.get_all_nodes():
    print(f"  {n.name} type={n.type}", flush=True)

print("=== save_to_air / load_from_air roundtrip ===", flush=True)
g.save_to_air("/tmp/ge_air_test.air")
print(f"  air file: {os.path.getsize('/tmp/ge_air_test.air')} bytes", flush=True)
g2 = Graph("loaded")
g2.load_from_air("/tmp/ge_air_test.air")
nodes2 = g2.get_all_nodes()
print(f"  loaded graph: {len(nodes2)} nodes (roundtrip {'OK' if len(nodes2)==len(g.get_all_nodes()) else 'MISMATCH'})", flush=True)
for n in nodes2:
    print(f"    {n.name} type={n.type}", flush=True)
