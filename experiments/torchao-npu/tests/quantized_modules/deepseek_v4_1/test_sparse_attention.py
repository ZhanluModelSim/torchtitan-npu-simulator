# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torchao.quantization.qat import QATStep
from torchao_npu.configs.module_swap_configs.deepseek_v4_1.quant_sparse_attention import (
    DeepSeekV41QuantSparseAttentionConfig,
    _quant_v41_sparse_attention_transform,
)
from torchao_npu.quantized_modules.deepseek_v4_1 import lightning_indexer, sparse_attention


def _indexer_metadata():
    """The selector's group on the entry: the sparse attention core never reads it."""
    return SimpleNamespace(
        fwd=torch.tensor([7], dtype=torch.int32),
        bwd=torch.tensor([5], dtype=torch.int32),
    )


def _metadata(*, tokens=4, compressed_tokens=2, ratio=2, quant=False):
    """The attention metadata one layer is handed: this geometry's frames, and its AI_CPU metadata.

    The SMLA pair is what the quantized core reads; the selector's group rides the same entry when
    the MXFP4 selector is installed, and a stack that keeps its selector in BF16 carries none.
    """
    if compressed_tokens is None:
        cmp_k = None
    else:
        cmp_k = SimpleNamespace(
            cu_seqlens=torch.tensor([0, compressed_tokens], dtype=torch.int32),
            residual=None if ratio == 1 else torch.zeros(1, dtype=torch.int32),
            seqused=torch.tensor([compressed_tokens], dtype=torch.int32),
        )
    return SimpleNamespace(
        q=SimpleNamespace(
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
            seqused=torch.tensor([tokens], dtype=torch.int32),
        ),
        swa_k=SimpleNamespace(
            cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32),
            seqused=torch.tensor([tokens], dtype=torch.int32),
        ),
        cmp_k=cmp_k,
        aicpu_metadata=SimpleNamespace(
            sparse_attn=SimpleNamespace(
                fwd=torch.tensor([3], dtype=torch.int32),
                bwd=torch.tensor([4], dtype=torch.int32),
            ),
            lightning_indexer=_indexer_metadata() if quant else None,
        ),
    )


def _fake_mla_apply(captured):
    def apply(q, swa_k, cmp_k, indices, sinks, attention_metadata, scale, ratio, window, scores, wants_teacher):
        captured.update(
            q=q,
            swa_k=swa_k,
            cmp_k=cmp_k,
            indices=indices,
            sinks=sinks,
            attention_metadata=attention_metadata,
            scale=scale,
            ratio=ratio,
            window=window,
            scores=scores,
            wants_teacher=wants_teacher,
        )
        lse = torch.zeros((1, q.shape[0], q.shape[1]), dtype=q.dtype, device=q.device)
        return q, lse

    return staticmethod(apply)


@pytest.mark.parametrize(
    ("ratio", "with_shared", "cmp_tokens"),
    [(0, False, None), (1, True, 4), (2, True, 2)],
    ids=["window-only", "full-resolution-shared-kv", "compressed-shared-kv"],
)
def test_sparse_mla_quantizes_only_swa_and_preserves_main_kv_in_both_passes(
    monkeypatch, ratio, with_shared, cmp_tokens
):
    """Exercise the custom autograd function around mocked NPU kernel boundaries."""
    device = "cpu"
    q = torch.ones((4, 2, 4), device=device, dtype=torch.bfloat16, requires_grad=True)
    swa_k = torch.ones((4, 1, 4), device=device, dtype=torch.bfloat16, requires_grad=True)
    cmp_k = (
        torch.ones((cmp_tokens, 1, 4), device=device, dtype=torch.bfloat16, requires_grad=True) if with_shared else None
    )
    indices = (
        torch.tensor([[[0, -1]], [[1, 0]], [[0, 1]], [[1, -1]]], device=device, dtype=torch.int32)
        if with_shared
        else None
    )
    scores = torch.empty((4, 1, 2), device=device, dtype=torch.float32, requires_grad=True) if with_shared else None
    sinks = torch.ones((2,), device=device, dtype=torch.float32, requires_grad=True)
    attention_metadata = _metadata(tokens=4, compressed_tokens=cmp_tokens, ratio=ratio)
    attention_metadata.q.seqused = torch.tensor([2], dtype=torch.int32)
    attention_metadata.swa_k.seqused = torch.tensor([3], dtype=torch.int32)
    if attention_metadata.cmp_k is not None:
        attention_metadata.cmp_k.seqused = torch.tensor([min(cmp_tokens, 1)], dtype=torch.int32)
    expected_residual = attention_metadata.cmp_k.residual if with_shared else None
    expected_lengths = {
        "seqused_q": attention_metadata.q.seqused,
        "seqused_ori_kv": attention_metadata.swa_k.seqused,
        "seqused_cmp_kv": attention_metadata.cmp_k.seqused if with_shared else None,
    }
    quantize_calls = []
    forward_kwargs = {}
    lse = torch.zeros((1, q.shape[0], q.shape[1]), device=device, dtype=q.dtype)
    teacher = torch.arange(8, device=device, dtype=torch.float32).reshape(4, 1, 2) if with_shared else torch.empty(0)

    def fake_quantize(kv, quant_group_size, quant_mode):
        quantize_calls.append((kv, quant_group_size, quant_mode))
        return kv + 1

    def sparse_flash_mla(q_input, **kwargs):
        forward_kwargs.update(kwargs)
        return q_input + 1, lse

    def sparse_flash_mla_grad(q_input, grad_output, output, saved_lse, **kwargs):
        assert q_input is q
        assert output.shape == q.shape
        assert saved_lse is lse
        torch.testing.assert_close(grad_output, torch.ones_like(q))
        assert kwargs["ori_kv"] is forward_kwargs["ori_kv"]
        assert kwargs["ori_kv"] is not swa_k
        torch.testing.assert_close(kwargs["ori_kv"], torch.full_like(swa_k, 2))
        assert kwargs["cmp_kv"] is cmp_k
        assert kwargs["cmp_sparse_indices"] is indices
        assert kwargs["sinks"] is sinks
        # Both passes read the metadata the override built for these frames, so the quantized core
        # launches no AI_CPU operator and cannot disagree with the unquantized one.
        assert kwargs["metadata"] is attention_metadata.aicpu_metadata.sparse_attn.bwd
        assert kwargs["cmp_residual_kv"] is expected_residual
        assert kwargs["softmax_scale"] == 0.5
        assert kwargs["cmp_ratio"] == max(ratio, 1)
        for name, expected in expected_lengths.items():
            assert kwargs[name] is expected
        return (
            torch.full_like(q, 2),
            torch.full_like(swa_k, 3),
            torch.full_like(cmp_k, 4) if cmp_k is not None else None,
            torch.full_like(sinks, 5),
            None,
            teacher,
        )

    monkeypatch.setattr(sparse_attention, "fake_quantize_mx_bf16", fake_quantize)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla", sparse_flash_mla)
    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", sparse_flash_mla_grad)

    output, returned_lse = sparse_attention._SparseMLA.apply(
        q,
        swa_k,
        cmp_k,
        indices,
        sinks,
        attention_metadata,
        0.5,
        ratio,
        2,
        scores,
        with_shared,
    )

    assert returned_lse is lse
    assert not returned_lse.requires_grad
    assert len(quantize_calls) == 1
    assert quantize_calls[0][0] is swa_k
    assert quantize_calls[0][1:] == (32, "mxfp8_bf16")
    torch.testing.assert_close(forward_kwargs["ori_kv"], swa_k + 1)
    assert forward_kwargs["cmp_kv"] is cmp_k
    assert forward_kwargs["metadata"] is attention_metadata.aicpu_metadata.sparse_attn.fwd
    assert forward_kwargs["cmp_sparse_indices"] is indices
    assert forward_kwargs["sinks"] is sinks
    assert forward_kwargs["cmp_residual_kv"] is expected_residual

    output.sum().backward()

    for options in (forward_kwargs,):
        for name, expected in expected_lengths.items():
            assert options[name] is expected
    torch.testing.assert_close(q.grad, torch.full_like(q, 2))
    torch.testing.assert_close(swa_k.grad, torch.full_like(swa_k, 3))
    torch.testing.assert_close(sinks.grad, torch.full_like(sinks, 5))
    if with_shared:
        torch.testing.assert_close(cmp_k.grad, torch.full_like(cmp_k, 4))
        torch.testing.assert_close(scores.grad, teacher.masked_fill(indices < 0, 0.0))


def test_forward_ratio_zero_passes_window_only_inputs_to_mla(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))

    device = "cpu"
    q = torch.randn((1, 4, 2, 512), device=device, dtype=torch.bfloat16)
    swa_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seq_q=torch.tensor([0, 4], device=device, dtype=torch.int32),
        doc_ids_BL=torch.zeros(4, device=device, dtype=torch.int32),
    )
    module = sparse_attention.QuantV41SparseAttention(
        window_size=3,
        softmax_scale=0.125,
        compress_ratio=0,
    )

    output = module(
        q,
        swa_k,
        torch.zeros(2, device=device, dtype=torch.float32),
        metadata,
        cmp_k=None,
        topk_indices=None,
    )

    assert output.shape == q.shape
    assert captured["cmp_k"] is None
    assert captured["indices"] is None
    assert captured["attention_metadata"] is metadata
    assert captured["ratio"] == 0
    assert captured["window"] == 3
    assert captured["scores"] is None
    assert captured["wants_teacher"] is False


def test_forward_ratio_one_uses_full_resolution_shared_kv_without_residual(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))

    device = "cpu"
    q = torch.randn((1, 4, 2, 512), device=device, dtype=torch.bfloat16)
    swa_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    cmp_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seq_q=torch.tensor([0, 4], device=device, dtype=torch.int32),
        doc_ids_BL=torch.zeros(4, device=device, dtype=torch.int32),
    )
    module = sparse_attention.QuantV41SparseAttention(
        window_size=3,
        softmax_scale=0.125,
        compress_ratio=1,
    )
    topk_scores = torch.empty((1, 4, 2), device=device, dtype=torch.float32, requires_grad=True)

    output = module(
        q,
        swa_k,
        torch.zeros(2, device=device, dtype=torch.float32),
        metadata,
        cmp_k=cmp_k,
        topk_indices=torch.zeros((1, 4, 2), device=device, dtype=torch.int64),
        topk_scores=topk_scores,
    )

    assert output.shape == q.shape
    assert captured["attention_metadata"] is metadata
    assert captured["cmp_k"].shape == (4, 1, 512)
    assert captured["scores"].shape == (4, 1, 2)
    assert captured["wants_teacher"] is True


def test_forward_ratio_two_passes_the_selection_through_in_tnd_layout(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))

    device = "cpu"
    q = torch.randn((1, 4, 2, 512), device=device, dtype=torch.bfloat16)
    swa_k = torch.randn((1, 4, 512), device=device, dtype=torch.bfloat16)
    cmp_k = torch.randn((1, 2, 512), device=device, dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        cu_seq_q=torch.tensor([0, 2, 4], device=device, dtype=torch.int32),
        doc_ids_BL=torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32),
    )
    topk_indices = torch.tensor(
        [[[0, 1], [1, 0], [1, 0], [0, 1]]],
        device=device,
        dtype=torch.int64,
    )
    module = sparse_attention.QuantV41SparseAttention(
        window_size=5,
        softmax_scale=0.25,
        compress_ratio=2,
    )
    topk_scores = torch.empty((1, 4, 2), device=device, dtype=torch.float32, requires_grad=True)

    output = module(
        q,
        swa_k,
        torch.zeros(2, device=device, dtype=torch.float32),
        metadata,
        cmp_k=cmp_k,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
    )

    assert output.shape == q.shape
    # ``[B, L, 1, K]`` -> ``[T, 1, K]``, values untouched: the selector speaks
    # document-local coordinates already and its order is the teacher's slot order.
    assert captured["indices"].shape == (4, 1, 2)
    assert captured["indices"].reshape(1, 4, 2).tolist() == topk_indices.tolist()
    assert captured["cmp_k"].shape == (2, 1, 512)
    assert captured["ratio"] == 2
    assert captured["scores"].shape == (4, 1, 2)
    assert captured["wants_teacher"] is True


def test_forward_eval_keeps_sparse_quantization_without_teacher_carrier(monkeypatch):
    captured = {}
    monkeypatch.setattr(sparse_attention._SparseMLA, "apply", _fake_mla_apply(captured))
    module = sparse_attention.QuantV41SparseAttention(window_size=3, softmax_scale=0.125, compress_ratio=1)
    module.eval()
    q = torch.zeros((1, 4, 2, 4), dtype=torch.bfloat16)
    cmp_k = torch.zeros((1, 4, 4), dtype=torch.bfloat16)

    output = module(
        q,
        torch.zeros((1, 4, 4), dtype=torch.bfloat16),
        torch.zeros(2, dtype=torch.float32),
        _metadata(tokens=4, compressed_tokens=4),
        cmp_k=cmp_k,
        topk_indices=torch.zeros((1, 4, 2), dtype=torch.int32),
    )

    assert output.shape == q.shape
    assert captured["scores"] is None
    assert captured["wants_teacher"] is False


def test_forward_training_requires_teacher_carrier_for_compressed_kv():
    module = sparse_attention.QuantV41SparseAttention(window_size=3, softmax_scale=0.125, compress_ratio=1)

    with pytest.raises(ValueError, match="requires topk_scores"):
        module(
            torch.zeros((1, 4, 2, 4), dtype=torch.bfloat16),
            torch.zeros((1, 4, 4), dtype=torch.bfloat16),
            torch.zeros(2, dtype=torch.float32),
            _metadata(tokens=4, compressed_tokens=4),
            cmp_k=torch.zeros((1, 4, 4), dtype=torch.bfloat16),
            topk_indices=torch.zeros((1, 4, 2), dtype=torch.int32),
        )


def test_module_swap_forwards_teacher_carrier_and_restores_original(monkeypatch):
    class SparseAttentionHost(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return "original", args, kwargs

    host = SparseAttentionHost()
    original_forward = host.forward
    captured = {}

    def quantized_forward(self, q, swa_k, attn_sink, attention_metadata, **kwargs):
        captured.update(self=self, args=(q, swa_k, attn_sink, attention_metadata), kwargs=kwargs)
        return "quantized"

    monkeypatch.setattr(sparse_attention.QuantV41SparseAttention, "forward", quantized_forward)
    _quant_v41_sparse_attention_transform(host, DeepSeekV41QuantSparseAttentionConfig())
    operands = tuple(object() for _ in range(4))
    cmp_k, indices, scores = object(), object(), object()

    assert host(*operands, cmp_k=cmp_k, topk_indices=indices, topk_scores=scores, ignored=True) == "quantized"
    assert captured["self"] is host
    assert captured["args"] == operands
    assert captured["kwargs"] == {"cmp_k": cmp_k, "topk_indices": indices, "topk_scores": scores}

    _quant_v41_sparse_attention_transform(host, DeepSeekV41QuantSparseAttentionConfig(step=QATStep.CONVERT))
    result = host(*operands, topk_scores=scores)

    assert result[0] == "original"
    assert host._torchao_npu_original_forward == original_forward


def test_quantized_sparse_teacher_reaches_quantized_lightning_indexer_slikg(monkeypatch):
    tokens, topk = 4, 2
    attention_metadata = _metadata(tokens=tokens, compressed_tokens=2, quant=True)
    idx_q = torch.zeros((tokens, 2, 4), dtype=torch.bfloat16, requires_grad=True)
    idx_k = torch.zeros((2, 1, 4), dtype=torch.bfloat16, requires_grad=True)
    idx_w = torch.zeros((tokens, 2), dtype=torch.float32, requires_grad=True)
    selected = torch.tensor([[[1, -1]], [[1, 0]], [[1, 0]], [[0, -1]]], dtype=torch.int32)
    teacher = torch.arange(tokens * topk, dtype=torch.float32).reshape(tokens, 1, topk)
    seen = {}

    monkeypatch.setattr(lightning_indexer, "_pack_mxfp4", lambda value: (value, value.new_zeros(1)))
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "ds41",
        SimpleNamespace(
            quant_lightning_indexer=lambda *args, **kwargs: (selected, torch.empty(0), torch.empty(0), torch.empty(0)),
        ),
        raising=False,
    )

    def slikg(**kwargs):
        seen["teacher"] = kwargs["attn_softmax_l1_norm"]
        return torch.full_like(idx_q, 2), torch.full_like(idx_k, 3), torch.full_like(idx_w, 4), torch.empty(0)

    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad",
        slikg,
        raising=False,
    )
    monkeypatch.setattr(sparse_attention, "fake_quantize_mx_bf16", lambda value, **kwargs: value)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_flash_mla",
        lambda q, **kwargs: (q + 1, torch.zeros((1, tokens, q.shape[1]), dtype=q.dtype)),
    )

    def sparse_flash_mla_grad(q, grad_output, output, lse, **kwargs):
        return (
            torch.ones_like(q),
            torch.ones_like(kwargs["ori_kv"]),
            torch.ones_like(kwargs["cmp_kv"]),
            torch.ones_like(kwargs["sinks"]),
            None,
            teacher,
        )

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "sparse_flash_mla_grad", sparse_flash_mla_grad)

    indices, scores, _ = lightning_indexer._QuantV41LightningIndexerTND.apply(
        idx_q,
        idx_k,
        idx_w,
        topk,
        2,
        attention_metadata,
        False,
        None,
        torch.tensor(16.0),
        # This layer runs no candidate pool: a pool-free selector is the QLI branch.
        -1,
        -1,
    )
    output, _ = sparse_attention._SparseMLA.apply(
        torch.zeros((tokens, 2, 4), dtype=torch.bfloat16, requires_grad=True),
        torch.zeros((tokens, 1, 4), dtype=torch.bfloat16, requires_grad=True),
        torch.zeros((2, 1, 4), dtype=torch.bfloat16, requires_grad=True),
        indices,
        torch.zeros(2, dtype=torch.float32, requires_grad=True),
        attention_metadata,
        0.5,
        2,
        3,
        scores,
        True,
    )

    output.sum().backward()

    expected_teacher = teacher.masked_fill(indices < 0, 0.0) / 16.0
    torch.testing.assert_close(seen["teacher"], expected_teacher)
    torch.testing.assert_close(idx_q.grad, torch.full_like(idx_q, 2))
    torch.testing.assert_close(idx_k.grad, torch.full_like(idx_k, 3))
    torch.testing.assert_close(idx_w.grad, torch.full_like(idx_w, 4))
