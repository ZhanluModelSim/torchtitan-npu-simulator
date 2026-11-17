# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CV-chunk symbolic token permutation and dispatcher isolation."""

import torch
import torch_npu
from torch._dynamo.decorators import mark_unbacked
from torchtitan.config.override import apply_overrides
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced

from torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_token_dispatcher import BatchChunkTokenDispatcher
from torchtitan_npu.models.deepseek_v4.config_registry import graph_trainer_deepseek_v4_debugmodel
from torchtitan_npu.override.common.token_dispatcher import AscAllToAllTokenDispatcher
from torchtitan_npu.patches.torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher


def test_symbolic_dispatch_preserves_gradients_without_changing_ordinary_dispatch(monkeypatch):
    # Emulate only the native NPU kernel boundary; config selection and tracing are real.
    backward_calls = []

    def fake_permute(tokens, indices):
        if torch.overrides.has_torch_function((tokens, indices)):
            return torch.overrides.handle_torch_function(fake_permute, (tokens, indices), tokens, indices)
        order = indices.flatten().argsort(stable=True)
        return tokens.repeat_interleave(indices.shape[1], dim=0)[order], order.argsort().to(torch.int32)

    def fake_backward(grad, sorted_indices, num_tokens, dtype, topk):
        backward_calls.append((num_tokens, dtype, topk))
        return grad[sorted_indices.long()].reshape(num_tokens, topk, grad.shape[-1]).sum(1).to(dtype)

    monkeypatch.setattr(torch_npu, "npu_moe_token_permute", fake_permute)
    monkeypatch.setattr(torch_npu, "npu_moe_token_permute_grad_v2", fake_backward)
    for experimental in (False, True, False):
        config = graph_trainer_deepseek_v4_debugmodel()
        config.override.imports = [
            "torchtitan_npu.extensions.experiment.cv_parallel.batch_chunk_token_dispatcher.asc_dispatcher"
            if experimental
            else "torchtitan_npu.override.common.token_dispatcher.asc"
        ]
        apply_overrides(config.override, config)
        dispatcher = next(config.model_spec.model.traverse(AllToAllTokenDispatcher.Config))[1].build()
        assert type(dispatcher) is (BatchChunkTokenDispatcher if experimental else AscAllToAllTokenDispatcher)
        tokens = (torch.arange(6, dtype=torch.float64).reshape(2, 3) / 8 - 0.25).requires_grad_()
        scores = ((torch.arange(6, dtype=torch.float64).reshape(2, 3) + 1) / 16).requires_grad_()
        ids = torch.tensor([[3, 0, 1], [2, 1, 0]])
        counts = torch.tensor([2, 2, 1, 1])
        # Hand-enumerated expert-major token and score rows for these assignments.
        expected_routed = tokens[[0, 1, 0, 1, 1, 0]] * scores.flatten()[[1, 5, 2, 4, 3, 0], None]
        expected = (expected_routed, *torch.autograd.grad(expected_routed.square().sum(), (tokens, scores)))

        def step(x, probabilities, assignments, expert_counts, *, selected_dispatcher=dispatcher):
            routed, _, metadata = selected_dispatcher.dispatch(x, probabilities, assignments, expert_counts)
            weighted = routed * metadata.routed_scores_R[:, None]
            return weighted, *torch.autograd.grad(weighted.square().sum(), (x, probabilities))

        before = len(backward_calls)
        if experimental:
            for tensor in (tokens, scores, ids):
                mark_unbacked(tensor, 0, min=2, shape_id="batch_chunk_override")
            traced = minimal_fx_tracer(step)(tokens, scores, ids, counts)
            actual = run_traced(traced)(tokens, scores, ids, counts)
            fake_tokens = next(n.meta["val"] for n in traced.gm.graph.nodes if n.op == "placeholder")
            assert isinstance(fake_tokens.shape[0], torch.SymInt)
            assert sorted(backward_calls[before:]) == [(2, torch.float64, 3), (6, torch.float64, 1)]
        else:
            actual = step(tokens, scores, ids, counts)
            assert len(backward_calls) == before

        for result, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, target, rtol=0, atol=0)
