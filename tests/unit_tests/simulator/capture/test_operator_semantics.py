# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan_npu.simulator.capture.dispatch_capture import OpDispatchCapture


def test_consecutive_chain_has_distinct_nodes_and_edges():
    capture = OpDispatchCapture()
    with capture:
        x = torch.ones(4, device="meta")
        for _ in range(5):
            x = x.relu()
    nodes = [node for node in capture.build_nodes().values() if node.annotations["raw_op_type"] == "aten.relu.default"]
    assert len(nodes) == 5
    assert len({node.op_id for node in nodes}) == 5
    assert all(previous.op_id in current.predecessors for previous, current in zip(nodes, nodes[1:]))
    assert all(node.op_id not in node.predecessors for node in nodes)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_schema_arguments_and_layout_are_retained(dtype):
    x = torch.ones(2, 3, dtype=dtype, device="meta")
    capture = OpDispatchCapture()
    with capture:
        torch.add(x, x, alpha=2)
        torch.sum(x, dim=1, keepdim=True)
        torch.sum(x, dim=0, keepdim=False)
        x.transpose(0, 1).clone()
    nodes = list(capture.build_nodes().values())
    add = next(node for node in nodes if node.annotations["raw_op_type"] == "aten.add.Tensor")
    reductions = [node for node in nodes if node.annotations["raw_op_type"] == "aten.sum.dim_IntList"]
    assert add.attrs["alpha"] == 2
    assert [(node.attrs["dim"], node.attrs["keepdim"]) for node in reductions] == [([1], True), ([0], False)]
    clone = next(node for node in nodes if node.annotations["raw_op_type"] == "aten.clone.default")
    assert clone.inputs[0].stride == (1, 3)
    assert clone.inputs[0].dtype == str(dtype).removeprefix("torch.")
