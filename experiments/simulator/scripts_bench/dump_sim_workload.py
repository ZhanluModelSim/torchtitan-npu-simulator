"""Capture the simulator workload for the real-run-matched configs and dump
ATen-level operator aggregates (name, count, shape/dtype signatures) plus the
memory plan, in the same JSON shape as parse_real.py output."""

import json
import sys
from collections import Counter, defaultdict

from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate

out_prefix = sys.argv[1]

workload = simulate(
    SimulationConfig(
        config="qwen3_5_debugmodel",
        seq_len=512,
        local_batch_size=1,
        world_size=4,
        data_parallel_shard_degree=4,
        mixed_precision_param="bfloat16",
        logical_rank=0,
    )
)

summary = {}
step_counts = Counter()
for template_name, graph in workload.step_templates.items():
    for node in graph.nodes.values():
        raw = node.annotations["raw_op_type"]
        sig = (
            tuple(tuple(t.shape) for t in node.inputs),
            tuple(str(t.dtype).removeprefix("torch.") for t in node.inputs),
        )
        entry = summary.setdefault(raw, {"count": 0, "sig": Counter(), "phases": Counter()})
        entry["count"] += 1
        entry["sig"][sig] += 1
        entry["phases"][node.annotations.get("phase", "?")] += 1
        step_counts[template_name] += 1

result = {
    name: {
        "count": v["count"],
        "phases": dict(v["phases"]),
        "signatures": [
            {"shapes": str(s), "dtypes": str(d), "n": n} for (s, d), n in v["sig"].most_common(6)
        ],
    }
    for name, v in summary.items()
}
with open(f"{out_prefix}_summary.json", "w") as f:
    json.dump(result, f, indent=1)

memory = workload.iteration.schedule.annotations["memory_plan"]
with open(f"{out_prefix}_memory.json", "w") as f:
    json.dump(memory.to_summary_dict(), f, indent=1)

print("step templates:", {k: len(g.nodes) for k, g in workload.step_templates.items()})
print("distinct raw op types:", len(result))
print("memory plan keys:", sorted(memory.to_summary_dict().keys()))
