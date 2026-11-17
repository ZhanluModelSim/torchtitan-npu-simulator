# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch.fx import Graph, GraphModule, Interpreter
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.graph_trainer.registry import MEMORY_POLICY_REGISTRY
from torchtitan.experiments.graph_trainer.selective_activation_remat import selective_activation_remat_pass

import torchtitan_npu.models.deepseek_v4.memory_policy  # noqa: F401


@pytest.fixture(scope="module")
def swiglu_op():
    # Model only the NPU operator boundary; policy and remat are real.
    # Older supported PyTorch versions do not implement Library.__enter__.
    lib = torch.library.Library("w13_policy_test", "DEF")
    lib.define("swiglu_group(Tensor x) -> Tensor")

    def forward(x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    lib.impl("swiglu_group", forward, "CompositeImplicitAutograd")
    yield torch.ops.w13_policy_test.swiglu_group.default


def _build_graph(swiglu_op, *, separate_projections=False, intervening_op=False):
    graph = Graph()
    x = graph.placeholder("x")
    w13 = graph.placeholder("w13")
    w2 = graph.placeholder("w2")
    nodes = {}
    backward_inputs = []

    def add(target, args, fqn, name):
        node = graph.call_function(target, args)
        node.name = name
        node.meta["custom"] = {"module_fqn": fqn}
        return node

    for layer in (0, 1, 4):
        outputs = []
        for kind, suffix in (("shared", "shared_experts"), ("routed", "routed_experts.inner_experts")):
            fqn = f"layers.{layer}.moe.{suffix}"
            key = f"{kind}{layer}"
            if separate_projections:
                gate = add(torch.ops.aten.mm.default, (x, w2), fqn, key + "_gate")
                up = add(torch.ops.aten.mm.default, (x, w2), fqn, key + "_up")
                gu = add(torch.ops.aten.cat.default, ([gate, up], -1), fqn, key + "_gu")
            else:
                gu = add(torch.ops.aten.mm.default, (x, w13), fqn, key + "_gu")
            view = add(torch.ops.aten.view.default, (gu, [2, 6]), fqn, key + "_view")
            swiglu_input = view
            if intervening_op:
                swiglu_input = add(torch.ops.aten.add.Tensor, (view, 0), fqn, key + "_bridge")
            h = add(swiglu_op, (swiglu_input,), fqn, key + "_h")
            out = add(torch.ops.aten.mm.default, (h, w2), fqn, key + "_w2")
            nodes[key] = (gu, swiglu_input, h, out)
            backward_inputs.extend((swiglu_input, h, out))
            outputs.append(out)
        gate = add(torch.ops.aten.mm.default, (x, w2), f"layers.{layer}.moe.router.gate", f"gate{layer}")
        merge = add(torch.ops.aten.add.Tensor, tuple(outputs), f"layers.{layer}.moe", f"merge{layer}")
        nodes[f"base{layer}"] = (gate, merge)

    backward = []
    for index, inp in enumerate(backward_inputs):
        node = add(torch.ops.aten.sum.default, (inp,), inp.meta["custom"]["module_fqn"], f"bwd{index}")
        node.meta["autograd_backward"] = True
        backward.append(node)
    graph.output(tuple(backward))
    gm = GraphModule({}, graph)

    class PopulateValues(Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            node.meta["val"] = value
            return value

    inputs = (torch.randn(2, 3), torch.randn(3, 6), torch.randn(3, 3))
    PopulateValues(gm).run(*inputs)
    return gm, nodes, inputs


def _build_default_swiglu_graph():
    graph = Graph()
    x = graph.placeholder("x")
    w13 = graph.placeholder("w13")
    w2 = graph.placeholder("w2")
    nodes = {}
    backward_inputs = []

    def add(target, args, fqn, name):
        node = graph.call_function(target, args)
        node.name = name
        node.meta["custom"] = {"module_fqn": fqn}
        return node

    for layer in (0, 1):
        outputs = []

        shared_root = f"layers.{layer}.moe.shared_experts"
        shared_w1 = add(torch.ops.aten.mm.default, (x, w2), shared_root + ".w1", f"shared{layer}_w1")
        shared_silu = add(torch.ops.aten.silu.default, (shared_w1,), shared_root, f"shared{layer}_silu")
        shared_w3 = add(torch.ops.aten.mm.default, (x, w2), shared_root + ".w3", f"shared{layer}_w3")
        shared_h = add(torch.ops.aten.mul.Tensor, (shared_silu, shared_w3), shared_root, f"shared{layer}_h")
        shared_w2 = add(torch.ops.aten.mm.default, (shared_h, w2), shared_root + ".w2", f"shared{layer}_w2")
        nodes[f"shared{layer}"] = (shared_w1, shared_w3, shared_h, shared_w2)
        backward_inputs.extend((shared_w1, shared_w3, shared_h, shared_w2))
        outputs.append(shared_w2)

        routed_root = f"layers.{layer}.moe.routed_experts.inner_experts"
        routed_w1 = add(torch.ops.aten.mm.default, (x, w2), routed_root, f"routed{layer}_w1")
        routed_silu = add(torch.ops.aten.silu.default, (routed_w1,), routed_root, f"routed{layer}_silu")
        routed_w3 = add(torch.ops.aten.mm.default, (x, w2), routed_root, f"routed{layer}_w3")
        routed_h = add(torch.ops.aten.mul.Tensor, (routed_silu, routed_w3), routed_root, f"routed{layer}_h")
        routed_w2 = add(torch.ops.aten.mm.default, (routed_h, w2), routed_root, f"routed{layer}_w2")
        nodes[f"routed{layer}"] = (routed_w1, routed_w3, routed_h, routed_w2)
        backward_inputs.extend((routed_w1, routed_w3, routed_h, routed_w2))
        outputs.append(routed_w2)

        gate = add(torch.ops.aten.mm.default, (x, w2), f"layers.{layer}.moe.router.gate", f"gate{layer}")
        merge = add(torch.ops.aten.add.Tensor, tuple(outputs), f"layers.{layer}.moe", f"merge{layer}")
        nodes[f"base{layer}"] = (gate, merge)

    backward = []
    for index, inp in enumerate(backward_inputs):
        node = add(torch.ops.aten.sum.default, (inp,), inp.meta["custom"]["module_fqn"], f"default_bwd{index}")
        node.meta["autograd_backward"] = True
        backward.append(node)
    graph.output(tuple(backward))
    gm = GraphModule({}, graph)

    class PopulateValues(Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            node.meta["val"] = value
            return value

    inputs = (torch.randn(2, 3), torch.randn(3, 6), torch.randn(3, 3))
    PopulateValues(gm).run(*inputs)
    return gm, nodes, inputs


@pytest.mark.parametrize("separate_projections", [False, True])
@pytest.mark.parametrize("intervening_op", [False, True])
def test_fused_mhc_moe_save_policy_preserves_values_and_removes_selected_gemm_replay(
    swiglu_op, separate_projections, intervening_op
):
    gm, nodes, inputs = _build_graph(
        swiglu_op, separate_projections=separate_projections, intervening_op=intervening_op
    )
    expected = gm(*inputs)
    differentiable = tuple(value.detach().requires_grad_() for value in inputs)
    expected_grads = torch.autograd.grad(sum(gm(*differentiable)), differentiable, allow_unused=True)
    config = SimpleNamespace(
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        )
    )

    MEMORY_POLICY_REGISTRY["dsv4-mhc-moe-save"](gm, config=config)

    for layer in (0, 1, 4):
        router_gate, moe_merge = nodes[f"base{layer}"]
        assert router_gate.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert moe_merge.meta["recompute"] == CheckpointPolicy.MUST_SAVE
        for kind in ("shared", "routed"):
            gu, swiglu_input, h, w2 = nodes[f"{kind}{layer}"]
            assert gu.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
            assert swiglu_input.meta["recompute"] == CheckpointPolicy.MUST_SAVE
            assert h.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
            assert w2.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE

    selective_activation_remat_pass(gm)
    names = {node.name for node in gm.graph.nodes}
    for layer in (0, 1, 4):
        assert f"shared{layer}_gu_recomputed" not in names
        if separate_projections:
            assert f"shared{layer}_gate_recomputed" not in names
            assert f"shared{layer}_up_recomputed" not in names
            assert f"routed{layer}_gate_recomputed" not in names
            assert f"routed{layer}_up_recomputed" not in names
        assert f"routed{layer}_gu_recomputed" not in names
    gm.graph.lint()
    torch.testing.assert_close(gm(*inputs), expected, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(sum(gm(*differentiable)), differentiable, allow_unused=True)
    torch.testing.assert_close(actual_grads, expected_grads)


def test_mhc_moe_save_policy_without_swiglu_group_saves_w1_w3_outputs():
    gm, nodes, _ = _build_default_swiglu_graph()
    config = SimpleNamespace(
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        )
    )

    MEMORY_POLICY_REGISTRY["dsv4-mhc-moe-save"](gm, config=config)

    for layer in (0, 1):
        router_gate, moe_merge = nodes[f"base{layer}"]
        assert router_gate.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert moe_merge.meta["recompute"] == CheckpointPolicy.MUST_SAVE
        for kind in ("shared", "routed"):
            w1, w3, h, w2 = nodes[f"{kind}{layer}"]
            assert w1.meta["recompute"] == CheckpointPolicy.MUST_SAVE
            assert w3.meta["recompute"] == CheckpointPolicy.MUST_SAVE
            assert h.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
            assert w2.meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE


def test_fused_mhc_moe_save_policy_rejects_missing_activation_match(swiglu_op):
    gm, _, _ = _build_graph(swiglu_op)
    for node in gm.graph.nodes:
        if node.target == swiglu_op:
            node.target = torch.ops.aten.relu.default
    config = SimpleNamespace(
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        )
    )
    with pytest.raises(ValueError, match="Cannot identify"):
        MEMORY_POLICY_REGISTRY["dsv4-mhc-moe-save"](gm, config=config)
