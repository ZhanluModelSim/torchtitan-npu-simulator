# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native collective identities and independent two-rank transport oracles."""

import multiprocessing
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.distributed import _functional_collectives as funcol
from torchtitan_simulator.capture.comm_events import collective_dependencies, observe_collectives
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.memory.estimator import estimate_static_memory


@pytest.mark.parametrize("primitive", ["allgather", "reduce_scatter", "all_to_all"])
def test_native_meta_collective_has_real_forward_backward_nodes_and_memory(tmp_path, primitive):
    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=2)
    try:
        x = torch.empty(4, 8, device="meta", requires_grad=True)
        capture = OpDispatchCapture()
        with observe_collectives() as recorder, capture:
            if primitive == "allgather":
                y = funcol.all_gather_single_autograd(x, 1, dist.group.WORLD)
                expected_shape = (4, 16)
            elif primitive == "reduce_scatter":
                y = funcol.reduce_scatter_single_autograd(x, "sum", 1, dist.group.WORLD)
                expected_shape = (4, 4)
            else:
                y = funcol.all_to_all_single_autograd(x, [1, 3], [2, 2], dist.group.WORLD)
                expected_shape = (4, 8)
            y.sum().backward()
        nodes = capture.build_nodes()
        assert y.shape == expected_shape and x.grad.shape == x.shape
        assert len(recorder.events) == 2
        for event in recorder.events:
            node = nodes[event.op_id]
            assert node.annotations["raw_op_type"].startswith("_c10d_functional.")
            assert event.comm_ranks == [[0, 1]] and event.world_size == 2
        forward = recorder.events[0]
        backward = recorder.events[1]
        assert forward.comm_primitive == primitive
        assert (
            backward.comm_primitive
            == {"allgather": "reduce_scatter", "reduce_scatter": "allgather", "all_to_all": "all_to_all"}[primitive]
        )
        assert nodes[forward.op_id].successors
        assert nodes[backward.op_id].predecessors
        plan = estimate_static_memory(capture.memory_events(), comm_events=recorder.events)
        assert plan.peak_active_bytes >= x.numel() * x.element_size()
        assert all(item.death_seq >= item.birth_seq for item in plan.tensor_lifetimes)
    finally:
        dist.destroy_process_group()


def _gloo_exchange(rank, store_path, queue):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        x = (torch.arange(3, dtype=torch.float64) + 10 * rank).requires_grad_()
        sends = [1, 2] if rank == 0 else [2, 1]
        receives = [1, 2] if rank == 0 else [2, 1]
        capture = OpDispatchCapture()
        with observe_collectives() as recorder, capture:
            y = funcol.all_to_all_single_autograd(x, receives, sends, dist.group.WORLD)
            values = y.tolist()
            (y.sum() * (rank + 1)).backward()
        nodes = capture.build_nodes()
        queue.put(
            (
                rank,
                values,
                x.grad.tolist(),
                [nodes[event.op_id].annotations["raw_op_type"] for event in recorder.events],
            )
        )
    finally:
        dist.destroy_process_group()


def test_native_all_to_all_gloo_preserves_uneven_rows_and_remote_gradients(tmp_path):
    # Fork keeps this pytest importlib module available to the CPU workers.
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    workers = [context.Process(target=_gloo_exchange, args=(rank, str(tmp_path / "group"), queue)) for rank in range(2)]
    try:
        for worker in workers:
            worker.start()
        results = [queue.get(timeout=45) for _ in workers]
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
        observed = {rank: (values, gradient, ops) for rank, values, gradient, ops in results}
        assert observed[0][:2] == ([0, 10, 11], [1, 2, 2])
        assert observed[1][:2] == ([1, 2, 12], [1, 1, 2])
        assert all(ops == ["_c10d_functional.all_to_all_single.default"] * 2 for _, _, ops in observed.values())
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            if worker.pid is not None:
                worker.join(timeout=5)
        queue.close()


def test_native_inplace_collective_descriptors_follow_data_and_preexisting_alias(tmp_path):
    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=2)
    try:
        original = torch.empty(4, device="meta")
        capture = OpDispatchCapture()
        with observe_collectives() as recorder, capture:
            value = original.sin()
            alias = value.view(2, 2)
            control = original.cos()
            with collective_dependencies(control):
                dist.all_reduce(value)
            alias.relu()
        nodes = capture.build_nodes()
        source = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.sin.default")
        consumer = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.relu.default")
        control_node = next(node for node in nodes.values() if node.annotations["raw_op_type"] == "aten.cos.default")
        (event,) = recorder.events
        assert event.src_exit_op == source.op_id
        assert event.dst_entry_op == consumer.op_id
        assert {source.op_id, control_node.op_id} <= set(nodes[event.op_id].predecessors)
        assert event.op_id in consumer.predecessors
        raw = next(item for item in capture.memory_events() if item.op_id == event.op_id)
        assert raw.inputs[0].storage_key == raw.outputs[0].storage_key
        assert raw.inputs[0].tensor_id == raw.outputs[0].tensor_id
    finally:
        dist.destroy_process_group()


def test_native_list_all_to_all_counts_every_chunk_and_keeps_output_storages(tmp_path):
    dist.init_process_group("fake", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=2)
    try:
        inputs = [torch.empty(1, 8, device="meta"), torch.empty(3, 8, device="meta")]
        outputs = [torch.empty(3, 8, device="meta"), torch.empty(1, 8, device="meta")]
        capture = OpDispatchCapture()
        with observe_collectives() as recorder, capture:
            dist.all_to_all(outputs, inputs)
            actual_outputs = {capture.tensor_id(tensor) for tensor in outputs}
            torch.cat(outputs)
        (event,) = recorder.events
        node = capture.build_nodes()[event.op_id]
        assert node.annotations["raw_op_type"] == "c10d.alltoall_.default"
        assert event.volume_bytes == node.comm_bytes == 128
        assert event.input_shapes == ((1, 8), (3, 8))
        assert event.output_shapes == ((3, 8), (1, 8))
        assert node.attrs["communication_input_shapes"] == [(1, 8), (3, 8)]
        raw = next(item for item in capture.memory_events() if item.op_id == event.op_id)
        assert {ref.tensor_id for ref in raw.outputs} == actual_outputs
        assert event.dst_entry_op
    finally:
        dist.destroy_process_group()
