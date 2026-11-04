# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native Muon bucket prefetch preserves required waits and independent work."""

from collections import defaultdict
from contextlib import nullcontext

import torch
from torchtitan.distributed.flex_shard import _optimizer_reshard_runtime as reshard_runtime
from torchtitan_simulator.capture.dispatch_capture import get_active_capture
from torchtitan_simulator.compat.device import MetaDeviceModule
from torchtitan_simulator.config import SimulationConfig
from torchtitan_simulator.runner import simulate


def test_native_muon_alltoall_prefetch_keeps_overlap_and_buffer_reuse_dependencies(monkeypatch):
    original_device_module = torch.get_device_module
    # An earlier Muon session must not leave a cached device handle whose stream
    # context differs from the current capture's torch.meta stream context.
    simulate(SimulationConfig(optimizer_name="Muon", muon_ns_steps=1))
    assert torch.get_device_module is original_device_module
    runtime = reshard_runtime._BucketedRedistributionRuntime
    observations = []
    by_plan = {}
    event_owners = {}
    waits = []
    original_inbound = runtime._enqueue_storage_to_compute
    original_compute = runtime._compute_bucket
    original_outbound = runtime._enqueue_compute_to_storage
    original_redistributed = reshard_runtime._compute_redistributed
    original_wait = MetaDeviceModule.Stream.wait_event

    def scope(label):
        capture = get_active_capture()
        return capture.annotate_dispatch_events(label) if capture is not None else nullcontext()

    def inbound(plan, slot, context, **kwargs):
        if get_active_capture() is None:
            return original_inbound(plan, slot, context, **kwargs)
        index = len(observations)
        by_plan[id(plan)] = index
        observations.append({"slot": slot, "redistributed": [item.fqn for item in plan.redistributed_items]})
        with scope(f"muon_bucket_{index}:inbound"):
            work = original_inbound(plan, slot, context, **kwargs)
        event_owners[slot.compute_input_ready] = (index, "input_ready")
        event_owners[slot.compute_done] = (index, "compute_done")
        event_owners[slot.done] = (index, "done")
        return work

    def compute(work, slot, caller_stream, context, **kwargs):
        if get_active_capture() is None:
            return original_compute(work, slot, caller_stream, context, **kwargs)
        with scope(f"muon_bucket_{by_plan[id(work.plan)]}:compute"):
            return original_compute(work, slot, caller_stream, context, **kwargs)

    def outbound(work, context, **kwargs):
        if get_active_capture() is None:
            return original_outbound(work, context, **kwargs)
        with scope(f"muon_bucket_{by_plan[id(work.plan)]}:outbound"):
            return original_outbound(work, context, **kwargs)

    def redistributed(*args, **kwargs):
        with scope("muon_redistributed_compute"):
            return original_redistributed(*args, **kwargs)

    def wait_event(stream, event):
        capture = get_active_capture()
        frontier = event._frontier
        result = original_wait(stream, event)
        if capture is not None and event in event_owners:
            index, kind = event_owners[event]
            waits.append((index, kind, frontier, capture.stream_frontier(stream)))
        return result

    monkeypatch.setattr(runtime, "_enqueue_storage_to_compute", staticmethod(inbound))
    monkeypatch.setattr(runtime, "_compute_bucket", staticmethod(compute))
    monkeypatch.setattr(runtime, "_enqueue_compute_to_storage", staticmethod(outbound))
    monkeypatch.setattr(reshard_runtime, "_compute_redistributed", redistributed)
    monkeypatch.setattr(MetaDeviceModule.Stream, "wait_event", wait_event)

    workload = simulate(SimulationConfig(optimizer_name="Muon", world_size=8, expert_parallel_degree=2, seq_len=128))
    assert torch.get_device_module is original_device_module
    nodes = {node.op_id: node for template in workload.step_templates.values() for node in template.nodes.values()}
    scoped = defaultdict(list)
    for node in nodes.values():
        for label in node.annotations.get("framework_scopes", ()):
            scoped[label].append(node)

    def ancestors(op_id):
        seen, pending = set(), list(nodes[op_id].predecessors)
        while pending:
            parent = pending.pop()
            if parent not in seen:
                seen.add(parent)
                pending.extend(nodes[parent].predecessors)
        return seen

    matmuls = {"aten.mm.default", "aten.addmm.default", "aten.bmm.default", "aten.baddbmm.default"}
    bucket_nodes = []
    # Four decoder dense buckets, four experts buckets, and the mHC head.
    # With EFSDP=4 and EP=2 each expert matrix is row-sharded and needs redistribution.
    assert len(observations) == 9
    assert sum("inner_experts" in name for item in observations for name in item["redistributed"]) == 12
    for index in range(len(observations)):
        incoming = [n for n in scoped[f"muon_bucket_{index}:inbound"] if n.op_type == "all_to_all"]
        outgoing = [n for n in scoped[f"muon_bucket_{index}:outbound"] if n.op_type == "all_to_all"]
        assert len(incoming) == len(outgoing) == 1
        inbound_node, outbound_node = incoming[0], outgoing[0]
        compute_nodes = [n for n in scoped[f"muon_bucket_{index}:compute"] if n.annotations["raw_op_type"] in matmuls]
        redistributed_nodes = [
            n for n in compute_nodes if "muon_redistributed_compute" in n.annotations["framework_scopes"]
        ]
        for node in redistributed_nodes:
            assert inbound_node.op_id in ancestors(node.op_id), "Muon read an incomplete inbound all-to-all"
        if compute_nodes:
            assert compute_nodes[-1].op_id in ancestors(outbound_node.op_id)
            assert inbound_node.annotations["stream_id"] != compute_nodes[-1].annotations["stream_id"]
        assert inbound_node.op_id in ancestors(outbound_node.op_id)
        bucket_nodes.append((inbound_node, compute_nodes, outbound_node))

    for index, kind, frontier, target in waits:
        assert len(target) == 1
        wait = nodes[target[0]]
        assert wait.op_type == "stream_wait"
        assert set(frontier) <= set(wait.predecessors)
        incoming, compute_nodes, outgoing = bucket_nodes[index]
        if kind == "input_ready":
            assert incoming.op_id in ancestors(wait.op_id)
        elif kind == "compute_done":
            if compute_nodes:
                assert compute_nodes[-1].op_id in ancestors(wait.op_id)
            assert wait.op_id in ancestors(outgoing.op_id)
        else:
            assert kind == "done"
            assert outgoing.op_id in ancestors(wait.op_id)
    assert {(index, kind) for index, kind, _, _ in waits} == {
        (index, kind) for index in range(len(observations)) for kind in ("input_ready", "compute_done", "done")
    }

    for index in range(1, len(bucket_nodes)):
        incoming, current_compute, _ = bucket_nodes[index]
        _, previous_compute, previous_outgoing = bucket_nodes[index - 1]
        assert previous_compute
        # Launch order is not an execution dependency across the two native streams.
        assert previous_compute[-1].op_id not in ancestors(incoming.op_id)
        assert incoming.op_id not in ancestors(previous_compute[-1].op_id)
        if current_compute:
            assert previous_outgoing.op_id not in ancestors(current_compute[-1].op_id)
            assert current_compute[-1].op_id not in ancestors(previous_outgoing.op_id)

    last_slot_use = {}
    reused_slots = 0
    for index, observation in enumerate(observations):
        slot = id(observation["slot"])
        if slot in last_slot_use:
            old_index = last_slot_use[slot]
            assert bucket_nodes[old_index][2].op_id in ancestors(bucket_nodes[index][0].op_id)
            reused_slots += 1
        last_slot_use[slot] = index
    assert reused_slots > 0, "The native double buffer must be reused in this configuration"
