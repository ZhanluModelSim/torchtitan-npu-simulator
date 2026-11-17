# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for EP chunk materialization of functionalized accumulations."""

from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.graph_trainer.common_utils import _MODULE_FQN
from torchtitan.experiments.graph_trainer.ep_chunk_pass import apply_chunk_pass

# Tests exercise the patched EP chunk implementation directly.
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
    ep_forward_accumulation,
)

ep_forward_accumulation.apply()


def test_mutation_functionalization_pipeline_enables_accumulation_patch(monkeypatch):
    from torchtitan_npu.patches.torchtitan.experiments.graph_trainer import (
        functionalize_recompute_mutations,
    )

    apply = Mock()
    monkeypatch.setattr(ep_forward_accumulation, "apply", apply)
    monkeypatch.setattr(
        functionalize_recompute_mutations,
        "construct_default_graph_passes",
        Mock(return_value=[]),
    )

    functionalize_recompute_mutations._mutation_functionalization_pipeline(
        object(),
        object(),
    )

    apply.assert_called_once_with()


def _symbolic_batch_fake_mode(batch: int = 4):
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    fake_mode = torch._subclasses.FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=shape_env,
    )
    with fake_mode:
        sym_batch = shape_env.create_unbacked_symint()
        torch._dynamo.override_optimization_hint(sym_batch, batch)
    return fake_mode, sym_batch


def _functionalized_buffer_update_graph() -> torch.fx.GraphModule:
    graph = torch.fx.Graph()
    buffer = graph.placeholder("buffer")
    x = graph.placeholder("x")
    output = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    delta = graph.call_function(torch.ops.aten.sum.dim_IntList, args=(x, [0]))
    updated = graph.call_function(torch.ops.aten.add.Tensor, args=(buffer, delta))
    graph.call_function(torch.ops.aten.copy_.default, args=(buffer, updated))
    graph.output(output)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    fake_mode, sym_batch = _symbolic_batch_fake_mode()
    with fake_mode:
        x_val = torch.empty(sym_batch, 3)
        buffer_val = torch.empty(3)

    buffer.meta["val"] = buffer_val
    x.meta["val"] = x_val
    output.meta["val"] = x_val
    for node in (output, delta, updated):
        node.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}
        node.meta["recompute"] = CheckpointPolicy.PREFER_RECOMPUTE
    delta.meta["val"] = buffer_val
    updated.meta["val"] = buffer_val
    return gm


def test_chunk_materializes_functionalized_buffer_accumulation_once():
    gm = _functionalized_buffer_update_graph()
    apply_chunk_pass(
        gm,
        mode="batch",
        module_patterns=["layers.*.moe"],
        num_static_inputs=1,
    )
    gm.graph.lint()

    materializations = [
        node
        for node in gm.graph.nodes
        if node.meta.get("chunked_materialization_proof") == "forward_functionalized_buffer_accumulation"
    ]
    assert len(materializations) == 1

    x = torch.arange(12.0).reshape(4, 3)
    initial = torch.tensor([10.0, 20.0, 30.0])
    buffer = initial.clone()
    output = gm(buffer, x)

    torch.testing.assert_close(output, torch.relu(x))
    torch.testing.assert_close(buffer, initial + x.sum(dim=0))


def test_chunk_still_rejects_unproven_forward_live_out():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    relu = graph.call_function(torch.ops.aten.relu.default, args=(x,))
    reduced = graph.call_function(torch.ops.aten.amax.default, args=(relu, [0], False))
    graph.output((relu, reduced))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)

    fake_mode, sym_batch = _symbolic_batch_fake_mode()
    with fake_mode:
        x_val = torch.empty(sym_batch, 3)
        reduced_val = torch.empty(3)
    x.meta["val"] = x_val
    relu.meta["val"] = x_val
    reduced.meta["val"] = reduced_val
    for node in (relu, reduced):
        node.meta["custom"] = {_MODULE_FQN: "layers.0.moe"}

    with pytest.raises(ValueError, match="forward accumulation proof"):
        apply_chunk_pass(gm, mode="batch", module_patterns=["layers.*.moe"])
