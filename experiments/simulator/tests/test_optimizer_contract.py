# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torchtitan_simulator.capture.dispatch_capture import OpDispatchCapture
from torchtitan_simulator.capture.optimizer import capture_optimizer_param_groups
from torchtitan_simulator.runtime import meta_runtime


def test_native_fused_buckets_keep_actual_updates_and_mutations():
    first = torch.nn.Parameter(torch.empty(3, device="meta", dtype=torch.float32))
    second = torch.nn.Parameter(torch.empty(7, device="meta", dtype=torch.float64))
    unused = torch.nn.Parameter(torch.empty(11, device="meta"))
    capture = OpDispatchCapture()
    with meta_runtime():
        optimizer = torch.optim.AdamW(
            [first, second, unused], fused=True, lr=0.013, betas=(0.8, 0.97), eps=3e-7, weight_decay=0.021
        )
        first.grad = torch.empty_like(first)
        second.grad = torch.empty_like(second)
        with capture, capture_optimizer_param_groups(optimizer.step):
            optimizer.step()
            actual_storages = {
                capture.tensor_id(value): capture._storage_identity(value)[0]
                for parameter in (first, second)
                for value in (
                    parameter,
                    parameter.grad,
                    optimizer.state[parameter]["exp_avg"],
                    optimizer.state[parameter]["exp_avg_sq"],
                )
            }
    nodes = [
        node
        for node in capture.build_nodes().values()
        if node.annotations["raw_op_type"] == "aten._fused_adamw_.default"
    ]
    assert sorted(node.annotations["optimizer_logical_parameter_shapes"] for node in nodes) == [[(3,)], [(7,)]]
    assert all(node.annotations["optimizer_declared_group_sizes"] == [3] for node in nodes)
    assert all(node.annotations["optimizer_updated_parameter_count"] == 1 for node in nodes)
    assert all(node.outputs for node in nodes)
    assert unused not in optimizer.state
    assert optimizer.state[first]["exp_avg"].shape == (3,)
    assert optimizer.state[second]["exp_avg_sq"].dtype == torch.float64
    for node in nodes:
        assert {key: node.attrs[key] for key in ("lr", "beta1", "beta2", "eps", "weight_decay")} == {
            "lr": 0.013,
            "beta1": 0.8,
            "beta2": 0.97,
            "eps": 3e-7,
            "weight_decay": 0.021,
        }
        size = node.annotations["optimizer_logical_parameter_shapes"][0][0]
        parameter = first if size == 3 else second
        updated = [
            parameter,
            parameter.grad,
            optimizer.state[parameter]["exp_avg"],
            optimizer.state[parameter]["exp_avg_sq"],
        ]
        raw = next(event for event in capture.memory_events() if event.op_id == node.op_id)
        assert {ref.tensor_id for ref in raw.outputs} == {capture.tensor_id(value) for value in updated}
        assert all(ref.storage_key == actual_storages[ref.tensor_id] for ref in raw.outputs)
