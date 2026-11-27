# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantized V4 selector wiring with CPU stand-ins for QLI and SLIKG kernels."""

from types import SimpleNamespace

import pytest
import torch
from torchao.quantization.qat import QATStep
from torchao.quantization.quant_api import quantize_
from torchao_npu.configs.module_swap_configs.deepseek_v4.quant_lightning_indexer import (
    DeepSeekV4QuantLightningIndexerConfig,
)
from torchao_npu.quantized_modules.deepseek_v4 import lightning_indexer


@pytest.mark.parametrize("batch", [1, 2])
def test_module_swap_routes_teacher_to_indexer_inputs(monkeypatch, batch):
    tokens, heads, dim, topk = 8, 2, 4, 4
    query = torch.zeros(batch * tokens, heads, dim, dtype=torch.bfloat16, requires_grad=True)
    key = torch.zeros(batch * 2, dim, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.zeros(batch * tokens, heads, dtype=torch.bfloat16, requires_grad=True)
    forward_record = torch.tensor([7], dtype=torch.int32)
    backward_record = torch.tensor([9], dtype=torch.int32)
    cu_q = torch.arange(batch + 1, dtype=torch.int32) * tokens
    cu_k = torch.arange(batch + 1, dtype=torch.int32) * 2
    used_q = torch.full((batch,), tokens, dtype=torch.int32)
    used_k = torch.full((batch,), 2, dtype=torch.int32)
    residual = torch.zeros(batch, dtype=torch.int32)
    metadata = SimpleNamespace(forward=forward_record, backward=backward_record)
    seen = {}

    def input_adapter(module, q, k, w, md):
        # The host adapter establishes packed TND operands before the autograd node.
        assert md is metadata
        return lightning_indexer.QuantizedLightningIndexerInputs(
            query=q,
            key=k.unsqueeze(1).contiguous(),
            weights=w.float(),
            topk=topk,
            metadata=md.forward,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            cmp_residual_k=residual,
            backward_metadata=md.backward,
            seqused_q=used_q,
            seqused_k=used_k,
        )

    def quant_li(q, k, w, query_config, key_config, **kwargs):
        seen["forward"] = kwargs
        seen["operands"] = q, k, w
        indices = torch.tensor([0, -1, 1, -1], dtype=torch.int32).repeat(batch * tokens, 1).unsqueeze(1)
        return indices, torch.tensor([[-3.0, 6.0, 9.0, 12.0]]).expand_as(indices)

    def slikg(**kwargs):
        seen["backward"] = kwargs
        return (
            torch.full_like(kwargs["q"], 2),
            torch.full_like(kwargs["k"], 3),
            torch.full_like(kwargs["w"], 4),
            None,
        )

    monkeypatch.setattr(lightning_indexer, "quant_lightning_indexer", quant_li)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", slikg, raising=False)
    selector = torch.nn.Identity()
    original_forward = selector.forward
    config = DeepSeekV4QuantLightningIndexerConfig(input_adapter=input_adapter)

    quantize_(selector, config, filter_fn=lambda module, fqn: module is selector)
    indices, scores = selector(query, key, weights, attention_metadata=metadata)

    assert selector.forward.__func__ is lightning_indexer.quantized_lightning_indexer_forward
    assert indices.shape == (batch * tokens, topk) and indices.dtype == torch.int32
    assert not indices.requires_grad
    assert scores.shape == indices.shape and scores.dtype == torch.float32
    assert scores.grad_fn is not None
    torch.testing.assert_close(indices[0], torch.tensor([1, 0, -1, -1], dtype=torch.int32))
    torch.testing.assert_close(scores[0], torch.tensor([9.0, -3.0, -torch.inf, -torch.inf]))
    assert seen["forward"]["metadata"] is forward_record
    assert seen["forward"]["quant_mode"] == 5
    assert seen["forward"]["topk"] == topk
    for operands, expected_shape in zip(
        seen["operands"], [(batch * tokens, heads, dim), (batch * 2, 1, dim), (batch * tokens, heads)], strict=True
    ):
        assert operands.shape == expected_shape

    # Attention already masked padding and scaled the teacher by the objective coefficient and denominator.
    teacher = torch.tensor([8.0, 4.0, 0.0, 0.0]).expand(batch * tokens, topk)
    scores.backward(teacher)

    backward = seen["backward"]
    assert backward["metadata"] is backward_record
    torch.testing.assert_close(backward["sparse_indices"], indices.reshape(batch * tokens, 1, topk))
    expected_teacher = teacher.reshape(batch * tokens, 1, topk)
    torch.testing.assert_close(backward["attn_softmax_l1_norm"], expected_teacher, rtol=0, atol=0)
    for name, expected in [
        ("cu_seqlens_q", cu_q),
        ("cu_seqlens_k", cu_k),
        ("seqused_q", used_q),
        ("seqused_k", used_k),
        ("cmp_residual_k", residual),
    ]:
        assert backward[name] is expected
    assert backward["layout_q"] == "TND" and backward["layout_k"] == "TND"
    assert backward["mask_mode"] == 3 and backward["cmp_ratio"] == 4
    torch.testing.assert_close(query.grad, torch.full_like(query, 2), rtol=0, atol=0)
    torch.testing.assert_close(key.grad, torch.full_like(key, 3), rtol=0, atol=0)
    torch.testing.assert_close(weights.grad, torch.full_like(weights, 4), rtol=0, atol=0)
    quantize_(
        selector, DeepSeekV4QuantLightningIndexerConfig(step=QATStep.CONVERT), filter_fn=lambda m, fqn: m is selector
    )
    assert selector.forward == original_forward


def test_module_swap_rejects_missing_teacher_gradient(monkeypatch):
    query = torch.ones(2, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    key = torch.ones(1, 4, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.ones(2, 1, dtype=torch.bfloat16, requires_grad=True)
    record = torch.tensor([7], dtype=torch.int32)

    def input_adapter(module, q, k, w, metadata):
        return lightning_indexer.QuantizedLightningIndexerInputs(
            query=q,
            key=k.unsqueeze(1),
            weights=w.float(),
            topk=1,
            metadata=metadata,
            cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32),
            cmp_residual_k=torch.zeros(1, dtype=torch.int32),
            backward_metadata=metadata,
        )

    def quant_li(q, *args, **kwargs):
        return torch.zeros(q.size(0), 1, 1, dtype=torch.int32), torch.ones(q.size(0), 1, 1)

    def slikg(**kwargs):
        pytest.fail("SLIKG must not run without a teacher gradient")

    class AttentionWithoutTeacher(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, scores):
            return q.clone()

        @staticmethod
        def backward(ctx, grad_output):
            # Keep the carrier edge in the graph, but omit the attention's teacher.
            return grad_output, None

    monkeypatch.setattr(lightning_indexer, "quant_lightning_indexer", quant_li)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_lightning_indexer_kl_loss_grad", slikg, raising=False)
    selector = torch.nn.Identity()
    quantize_(
        selector,
        DeepSeekV4QuantLightningIndexerConfig(input_adapter=input_adapter),
        filter_fn=lambda module, fqn: module is selector,
    )
    _, scores = selector(query, key, weights, attention_metadata=record)

    with pytest.raises(RuntimeError, match="missing the teacher gradient from sparse attention"):
        AttentionWithoutTeacher.apply(query, scores).sum().backward()
