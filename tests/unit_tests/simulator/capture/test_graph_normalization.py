# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.capture.graph_normalization import fold_metadata_views
from torchtitan_npu.simulator.ir.op_node import OpNode


def _node(op_id: str, predecessors: list[str], *, metadata_view: bool = False) -> OpNode:
    return OpNode(
        op_id=op_id,
        op_type="transpose" if metadata_view else "matmul",
        inputs=[],
        outputs=[],
        attrs={},
        predecessors=predecessors,
        successors=[],
        annotations={"metadata_view": True} if metadata_view else {},
    )


def test_fold_metadata_views_removes_t_nodes_and_reconnects_dependencies():
    producer = _node("producer", [])
    transpose = _node("transpose", ["producer"], metadata_view=True)
    reshape = _node("reshape", ["transpose"], metadata_view=True)
    consumer = _node("consumer", ["reshape"])

    normalized = fold_metadata_views(
        {
            "producer": producer,
            "transpose": transpose,
            "reshape": reshape,
            "consumer": consumer,
        }
    )

    assert set(normalized) == {"producer", "consumer"}
    assert normalized["consumer"].predecessors == ["producer"]
    assert normalized["producer"].successors == ["consumer"]


def test_fold_metadata_views_preserves_external_dependencies():
    transpose = _node("transpose", ["external_producer"], metadata_view=True)
    consumer = _node("consumer", ["transpose"])

    normalized = fold_metadata_views({"transpose": transpose, "consumer": consumer})

    assert normalized["consumer"].predecessors == ["external_producer"]
    assert normalized["consumer"].successors == []
