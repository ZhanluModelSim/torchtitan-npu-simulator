# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.simulator.ir.op_node import OpNode
from torchtitan_npu.simulator.ir.step_graph import StepGraph


def _node(op_id: str, preds: list[str], succs: list[str]) -> OpNode:
    return OpNode(op_id=op_id, op_type="x", inputs=[], outputs=[], attrs={}, predecessors=preds, successors=succs)


def test_step_graph_computes_entry_and_exit_nodes():
    nodes = {
        "a": _node("a", [], ["b"]),
        "b": _node("b", ["a"], ["c"]),
        "c": _node("c", ["b"], []),
    }
    graph = StepGraph(step_id="s1", step_type="forward", nodes=nodes)
    assert graph.entry_nodes == ["a"]
    assert graph.exit_nodes == ["c"]
    assert graph.is_acyclic is True


def test_step_graph_detects_cycle():
    nodes = {
        "a": _node("a", ["b"], ["b"]),
        "b": _node("b", ["a"], ["a"]),
    }
    graph = StepGraph(step_id="s2", step_type="forward", nodes=nodes)
    assert graph.is_acyclic is False


def test_step_graph_empty_nodes_keeps_defaults():
    graph = StepGraph(step_id="s3", step_type="forward", nodes={})
    assert graph.entry_nodes == []
    assert graph.exit_nodes == []
    assert graph.is_acyclic is True


def test_step_graph_respects_explicit_entry_exit_override():
    nodes = {"a": _node("a", [], [])}
    graph = StepGraph(step_id="s4", step_type="forward", nodes=nodes, entry_nodes=["a"], exit_nodes=["a"])
    assert graph.entry_nodes == ["a"]
    assert graph.exit_nodes == ["a"]


def test_step_graph_diamond_dependency_is_acyclic():
    # a -> b, a -> c, b -> d, c -> d (classic diamond, must stay acyclic)
    nodes = {
        "a": _node("a", [], ["b", "c"]),
        "b": _node("b", ["a"], ["d"]),
        "c": _node("c", ["a"], ["d"]),
        "d": _node("d", ["b", "c"], []),
    }
    graph = StepGraph(step_id="s5", step_type="forward", nodes=nodes)
    assert graph.is_acyclic is True
    assert graph.entry_nodes == ["a"]
    assert graph.exit_nodes == ["d"]


def test_step_graph_external_predecessor_does_not_break_acyclic_check():
    # Regression test: a node whose predecessor lives in a DIFFERENT
    # StepGraph (e.g. this is a "backward" graph and "fwd_activation" is a
    # forward-phase op_id) must not be treated as creating a cycle, and
    # must still count as an entry node -- this exact bug was caught via
    # end-to-end integration testing: before the fix, every real
    # backward/optimizer StepGraph (whose nodes reference forward/backward
    # activations and gradients as external predecessors) was incorrectly
    # reported as `is_acyclic=False`.
    nodes = {
        "b1": _node("b1", ["fwd_activation_NOT_IN_THIS_DICT"], ["b2"]),
        "b2": _node("b2", ["b1"], []),
    }
    graph = StepGraph(step_id="s6", step_type="backward", nodes=nodes)
    assert graph.is_acyclic is True
    assert graph.entry_nodes == ["b1"]
    assert graph.exit_nodes == ["b2"]


def test_step_graph_multiple_external_predecessors_on_same_node():
    nodes = {
        "opt1": _node("opt1", ["grad_from_backward_1", "grad_from_backward_2"], []),
    }
    graph = StepGraph(step_id="s7", step_type="optimizer", nodes=nodes)
    assert graph.is_acyclic is True
    assert graph.entry_nodes == ["opt1"]
