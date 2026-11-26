# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Joint-graph Host Engram gradients and sparse updates against eager training."""

import copy

import pytest
import torch
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced

from tests.unit_tests.models.deepseek_v4_1.test_engram import _build_engram
from torchtitan_npu.extensions.torchtitan.experiments.graph_trainer.whole_graph_benchmark import _preserve_model_state

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("with_image", [False, True], ids=["text", "image"])
@pytest.mark.parametrize(
    "hidden_grad,freeze_scale",
    [(True, False), (True, True), (False, False)],
    ids=["hidden-anchor", "hidden-anchor-frozen-gate", "gate-fallback"],
)
def test_joint_graph_sparse_updates_match_eager(with_image, hidden_grad, freeze_scale):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        eager = _build_engram()
        if freeze_scale:
            eager.gate.q_weight.requires_grad_(False)
        graph = copy.deepcopy(eager)
        eager_optim = torch.optim.SparseAdam([eager.table.weight], lr=0.05)
        graph_optim = torch.optim.SparseAdam([graph.table.weight], lr=0.05)
        graph.table.enable_graph_training()
        tokens, positions = torch.tensor([4, 4, 5, 4]), torch.arange(4)
        image_mask = torch.tensor([False, True, False, False]) if with_image else None
        hidden = torch.randn(4, 4, 8, requires_grad=hidden_grad)

        def joint(x, input_ids, pos):
            output = graph(x, input_ids, pos, image_mask=image_mask)
            parameters = [p for p in graph.parameters() if p.requires_grad]
            targets = [x, *parameters] if hidden_grad else parameters
            return output, torch.autograd.grad(output.square().mean(), targets)

        traced = minimal_fx_tracer(joint, module=graph)(hidden, tokens, positions)
        assert graph.table.pending_sparse_grad() is None
        assert any(
            node.target == torch.ops.torchtitan_npu.host_engram_lookup_backward.default
            for node in traced.gm.graph.nodes
        )
        traced.gm.graph.eliminate_dead_code()
        traced.gm.recompile()
        replay = run_traced(traced, module=graph)
        for step in range(3):
            x = (hidden.detach() + step / 8).requires_grad_(hidden_grad)
            reference = eager(x, tokens, positions, image_mask=image_mask)
            reference.square().mean().backward()
            output, gradients = replay(x, tokens, positions)
            torch.testing.assert_close(output, reference, atol=0, rtol=0)
            expected_gradients = [p.grad for p in eager.gate.parameters() if p.requires_grad]
            if hidden_grad:
                expected_gradients.insert(0, x.grad)
            for actual, expected in zip(gradients, expected_gradients, strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                graph.table.pending_sparse_grad().to_dense(),
                eager.table.pending_sparse_grad().to_dense(),
                atol=0,
                rtol=0,
            )
            assert graph.table.weight.grad is None
            for module, optimizer in ((eager, eager_optim), (graph, graph_optim)):
                module.table.prepare_sparse_optimizer_step()
                optimizer.step()
                module.table.clear_sparse_gradient()
                module.zero_grad(set_to_none=True)
            torch.testing.assert_close(graph.table.weight, eager.table.weight, atol=0, rtol=0)


def test_graph_lookup_reuses_hidden_anchor_without_reading_gate_weight_twice(monkeypatch):
    from torchtitan_npu.models.deepseek_v4_1.engram import core

    engram = _build_engram()
    engram.table.enable_graph_training()
    q_weight = engram.gate.q_weight
    original_local_tensor = core._local_tensor
    reads = []

    def counted_local_tensor(value):
        if value is q_weight:
            reads.append(value)
        return original_local_tensor(value)

    monkeypatch.setattr(core, "_local_tensor", counted_local_tensor)
    hidden = torch.randn(4, 4, 8, requires_grad=True)
    engram(hidden, torch.tensor([4, 4, 5, 4]), torch.arange(4)).square().mean().backward()
    assert len(reads) == 1
    assert engram.table.pending_sparse_grad() is not None


def test_calibration_restores_existing_pending_sparse_gradient():
    engram = _build_engram()
    engram.table.accumulate_sparse_gradient(torch.tensor([1]), torch.ones(1, engram.table.embedding_dim))
    pending = engram.table.pending_sparse_grad()
    with pytest.raises(RuntimeError, match="candidate failed"), _preserve_model_state(engram):
        engram.table.accumulate_sparse_gradient(torch.tensor([2]), torch.ones(1, engram.table.embedding_dim))
        raise RuntimeError("candidate failed")
    assert engram.table.pending_sparse_grad() is pending
