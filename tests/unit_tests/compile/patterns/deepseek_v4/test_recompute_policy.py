# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.fx as fx
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.graph_trainer.memory_policy import (
    tag_with_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.registry import MEMORY_POLICY_REGISTRY
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)

import torchtitan_npu.models.deepseek_v4.config_registry as _dsv4_config_registry  # noqa: F401


def _node(
    graph: fx.Graph,
    name: str,
    target: object,
    args: tuple[object, ...],
    fqn: str,
    *,
    backward: bool = False,
) -> fx.Node:
    node = graph.call_function(target, args=args)
    node.name = name
    node.meta["custom"] = {"module_fqn": fqn}
    if backward:
        node.meta["autograd_backward"] = True
    return node


def _build_graph() -> tuple[fx.GraphModule, dict[str, fx.Node]]:
    graph = fx.Graph()
    nodes: dict[str, fx.Node] = {}

    x = graph.placeholder("x")
    weight = graph.placeholder("weight")
    grad = graph.placeholder("grad")

    previous = x

    for layer_id in range(8):
        layer_input = _node(
            graph,
            f"layer{layer_id}_input",
            torch.ops.aten.add.Tensor,
            (previous, x),
            f"layers.{layer_id}.block.input",
        )

        if layer_id in (0, 4):
            pre = _node(
                graph,
                f"pre{layer_id}",
                torch.ops.aten.add.Tensor,
                (layer_input, x),
                f"layers.{layer_id}.mhc.hc_pre",
            )
            wo_b = _node(
                graph,
                f"wo_b{layer_id}",
                torch.ops.aten.matmul.default,
                (pre, weight),
                f"layers.{layer_id}.attention.wo_b",
            )
            moe = _node(
                graph,
                f"moe{layer_id}",
                torch.ops.aten.add.Tensor,
                (wo_b, wo_b),
                f"layers.{layer_id}.moe",
            )
            output = _node(
                graph,
                f"post{layer_id}",
                torch.ops.aten.relu.default,
                (moe,),
                f"layers.{layer_id}.mhc.hc_post",
            )

            nodes.update(
                {
                    f"pre{layer_id}": pre,
                    f"wo_b{layer_id}": wo_b,
                    f"moe{layer_id}": moe,
                    f"post{layer_id}": output,
                }
            )
        else:
            output = _node(
                graph,
                f"out{layer_id}",
                torch.ops.aten.relu.default,
                (layer_input,),
                f"layers.{layer_id}.block.output",
            )

        nodes[f"out{layer_id}"] = output
        previous = output

    nodes["bwd_output"] = _node(
        graph,
        "bwd_output",
        torch.ops.aten.mul.Tensor,
        (previous, grad),
        "layers.7.backward",
        backward=True,
    )

    for layer_id in (0, 4):
        nodes[f"bwd_pre{layer_id}"] = _node(
            graph,
            f"bwd_pre{layer_id}",
            torch.ops.aten.mul.Tensor,
            (nodes[f"pre{layer_id}"], grad),
            f"layers.{layer_id}.mhc.hc_pre.backward",
            backward=True,
        )
        nodes[f"bwd_post{layer_id}"] = _node(
            graph,
            f"bwd_post{layer_id}",
            torch.ops.aten.mul.Tensor,
            (nodes[f"post{layer_id}"], grad),
            f"layers.{layer_id}.mhc.hc_post.backward",
            backward=True,
        )

    graph.output(
        (
            nodes["bwd_output"],
            nodes["bwd_pre0"],
            nodes["bwd_post0"],
            nodes["bwd_pre4"],
            nodes["bwd_post4"],
        )
    )

    return fx.GraphModule(torch.nn.Module(), graph), nodes


def test_dsv4_mhc_cross_layer_remat() -> None:
    gm, nodes = _build_graph()

    config = SimpleNamespace(
        compile=SimpleNamespace(memory_policy="dsv4-mhc"),
        parallelism=SimpleNamespace(
            fsdp_reshard_after_forward="always",
            pipeline_parallel_degree=1,
        ),
    )

    assert "dsv4-mhc" in MEMORY_POLICY_REGISTRY
    gm = tag_with_memory_policy_pass(gm, config=config)

    # MHC 节点策略
    for layer_id in (0, 4):
        assert nodes[f"wo_b{layer_id}"].meta["recompute"] == CheckpointPolicy.MUST_SAVE
        assert nodes[f"moe{layer_id}"].meta["recompute"] == CheckpointPolicy.MUST_SAVE
        assert nodes[f"pre{layer_id}"].meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE
        assert nodes[f"post{layer_id}"].meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE

    # layer3 -> layer4 是跨四层边界
    assert nodes["out3"].meta["recompute"] == CheckpointPolicy.MUST_SAVE
    assert nodes["out0"].meta["recompute"] == CheckpointPolicy.MUST_RECOMPUTE

    gm = selective_activation_remat_pass(gm)
    names = {node.name for node in gm.graph.nodes}

    # pre/post 应该重计算
    for layer_id in (0, 4):
        assert f"pre{layer_id}_recomputed" in names
        assert f"post{layer_id}_recomputed" in names

    # wo_b、moe 和跨层边界不应该重计算
    for name in ("wo_b0", "wo_b4", "moe0", "moe4", "out3"):
        assert f"{name}_recomputed" not in names

    # backward 必须使用重计算副本
    for layer_id in (0, 4):
        assert f"pre{layer_id}_recomputed" in {node.name for node in nodes[f"bwd_pre{layer_id}"].all_input_nodes}
        assert f"post{layer_id}_recomputed" in {node.name for node in nodes[f"bwd_post{layer_id}"].all_input_nodes}

    # post 重计算直接依赖已保存的 moe 输出
    post4_recomputed = next(node for node in gm.graph.nodes if node.name == "post4_recomputed")
    assert nodes["moe4"] in post4_recomputed.all_input_nodes
    assert "moe4_recomputed" not in {node.name for node in post4_recomputed.all_input_nodes}
