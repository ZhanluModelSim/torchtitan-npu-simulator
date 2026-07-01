# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.tensor_meta import TensorMeta


def test_tensor_meta_defaults_is_parameter_false():
    t = TensorMeta(name="x", shape=(2, 4), dtype="float32", device="meta")
    assert t.is_parameter is False


def test_tensor_meta_stores_all_fields():
    t = TensorMeta(name="w", shape=(8, 16), dtype="bfloat16", device="meta", is_parameter=True)
    assert t.name == "w"
    assert t.shape == (8, 16)
    assert t.dtype == "bfloat16"
    assert t.device == "meta"
    assert t.is_parameter is True


def test_op_node_construction_and_defaults():
    inp = TensorMeta(name="in_0", shape=(2, 4), dtype="float32", device="meta")
    out = TensorMeta(name="out_0", shape=(2, 4), dtype="float32", device="meta")
    node = OpNode(
        op_id="op_1",
        op_type="matmul",
        inputs=[inp],
        outputs=[out],
        attrs={},
        predecessors=[],
        successors=[],
    )
    assert node.flops == 0
    assert node.peak_mem == 0
    assert node.param_mem == 0
    assert node.comm_bytes == 0
    assert node.annotations == {}


def test_op_node_annotations_are_independent_between_instances():
    # dataclass default_factory must not share state across instances
    a = OpNode(op_id="a", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    b = OpNode(op_id="b", op_type="x", inputs=[], outputs=[], attrs={}, predecessors=[], successors=[])
    a.annotations["k"] = 1
    assert b.annotations == {}
