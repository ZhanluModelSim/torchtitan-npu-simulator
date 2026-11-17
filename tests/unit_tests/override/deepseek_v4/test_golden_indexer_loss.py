# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Packed Golden attention and indexer gradients against a dense CPU oracle."""

import importlib
from dataclasses import dataclass

import pytest
import torch
from torchtitan.config import Configurable, OverrideConfig, apply_overrides
from torchtitan.models.common.attention import VarlenMetadata

from torchtitan_npu.models.deepseek_v4 import _debugmodel
from torchtitan_npu.models.deepseek_v4.attention import CompressedSparseInnerAttention
from torchtitan_npu.models.deepseek_v4.compressor import select_compressed_slots
from torchtitan_npu.models.deepseek_v4.metadata import build_compressed_varlen_metadata
from torchtitan_npu.models.deepseek_v4.token_dispatcher import CPTokenDispatcher
from torchtitan_npu.override.deepseek_v4.sparse_attn import golden as golden_factory
from torchtitan_npu.override.deepseek_v4.sparse_attn.golden import GoldenCompressedSparseInnerAttention

pytestmark = pytest.mark.cpu


@dataclass(kw_only=True)
class _RootConfig(Configurable.Config):
    attention: CompressedSparseInnerAttention.Config


@pytest.fixture
def packed_metadata():
    # A short document has no compressed keys; the others have ragged tails.
    boundaries = torch.tensor([0, 3, 12, 24], dtype=torch.int32)
    varlen = VarlenMetadata(cu_seq_q=boundaries, cu_seq_k=boundaries.clone(), max_q=12, max_k=12)
    return build_compressed_varlen_metadata(varlen, (1, 4))


def _doc_relative_positions(metadata):
    """The positions the model passes to the indexer: each document counts from zero."""
    bounds = metadata.varlen.cu_seq_q.tolist()
    return torch.cat([torch.arange(end - start) for start, end in zip(bounds[:-1], bounds[1:])]).unsqueeze(0)


def _golden_core(monkeypatch, coefficient):
    registry = importlib.import_module("torchtitan.config.override")
    monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
    root = _RootConfig(attention=CompressedSparseInnerAttention.Config(
        window_size=3, compress_ratio=4, softmax_scale=0.5, index_topk=3,
    ))
    apply_overrides(OverrideConfig(imports=[
        (f"{golden_factory.__module__}.{golden_factory.__name__}", {"indexer_loss_coeff": coefficient}),
    ]), root)
    core = root.attention.build()
    assert isinstance(core, GoldenCompressedSparseInnerAttention)
    core.init_states(buffer_device=torch.device("cpu"))
    return core


def _inputs():
    generator = torch.Generator().manual_seed(12)
    shapes = ((1, 24, 2, 4), (1, 24, 4), (1, 6, 4), (1, 24, 2, 3), (1, 6, 3), (1, 24, 2), (2,))
    return [torch.randn(shape, generator=generator).requires_grad_() for shape in shapes]


def _dense_oracle(inputs, sparse, metadata):
    q, window_k, cmp_k, idx_q, idx_k, idx_w, sink = inputs
    outputs = []
    loss = q.new_zeros(())
    cmp_bounds = metadata.plans[4].cu_seqlens_cmp_k.tolist()
    bounds = metadata.varlen.cu_seq_q.tolist()
    for document, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        c_start, c_end = cmp_bounds[document:document + 2]
        for row in range(start, end):
            chosen = sparse[0, row]
            chosen = chosen[chosen >= 0]
            window = window_k[0, max(start, row - 2):row + 1]
            keys = torch.cat((window, cmp_k[0, c_start + chosen]))
            logits = q[0, row] @ keys.T * 0.5
            probabilities = torch.cat((logits, sink[:, None]), dim=-1).softmax(-1)
            outputs.append(probabilities[:, :-1] @ keys)
            if chosen.numel():
                teacher = probabilities[:, window.shape[0]:-1].detach().sum(0)
                scores = (torch.relu(idx_q[0, row] @ idx_k[0, c_start + chosen].T) * idx_w[0, row, :, None]).sum(0)
                target = teacher / teacher.sum()
                loss = loss + (teacher * (target.log() - scores.log_softmax(-1))).sum()
    return torch.stack(outputs).unsqueeze(0), loss * (0.5 / 24)


@pytest.mark.parametrize("coefficient,chunk", [(1.0, 2), (0.3, 64)], ids=["chunked-unit", "unchunked-scaled"])
def test_golden_auxiliary_loss_and_gradients_match_dense_oracle(monkeypatch, packed_metadata, coefficient, chunk):
    implementation = importlib.import_module(GoldenCompressedSparseInnerAttention.__module__)
    monkeypatch.setattr(implementation, "_ATTN_CHUNK", chunk)
    core = _golden_core(monkeypatch, coefficient)
    inputs = _inputs()
    expected_inputs = [value.detach().clone().requires_grad_() for value in inputs]
    q, window_k, cmp_k, idx_q, idx_k, idx_w, sink = inputs
    sparse = select_compressed_slots(
        idx_q, idx_k, idx_w, packed_metadata.varlen.cu_seq_q, packed_metadata.plans[4].cu_seqlens_cmp_k, 3
    )
    expected_output, expected_aux = _dense_oracle(expected_inputs, sparse, packed_metadata)

    actual = core(q, window_k, cmp_k, idx_q, idx_k, idx_w, attn_sink=sink,
                  sparse_indices=sparse, attention_masks=packed_metadata)
    upstream = torch.linspace(-1, 1, actual.numel()).reshape_as(actual)
    (actual * upstream).sum().backward()
    ((expected_output * upstream).sum() + expected_aux * coefficient).backward()

    torch.testing.assert_close(actual, expected_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(core._indexer_loss_acc, expected_aux.detach(), rtol=1e-5, atol=1e-6)
    for actual_input, expected_input in zip(inputs, expected_inputs):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(idx_q.grad) > 0
    assert torch.equal(idx_q.grad[:, :3], torch.zeros_like(idx_q.grad[:, :3]))
    assert "_indexer_loss_acc" not in core.state_dict()


@pytest.mark.parametrize("disabled", ["coefficient", "eval"], ids=["zero-coefficient", "evaluation"])
def test_golden_auxiliary_loss_preserves_forward_and_respects_disable(monkeypatch, packed_metadata, disabled):
    enabled = _golden_core(monkeypatch, 1.0)
    inactive = _golden_core(monkeypatch, 0.0 if disabled == "coefficient" else 1.0)
    if disabled == "eval":
        inactive.eval()
    inputs = _inputs()
    q, window_k, cmp_k, idx_q, idx_k, idx_w, sink = inputs
    sparse = select_compressed_slots(
        idx_q, idx_k, idx_w, packed_metadata.varlen.cu_seq_q, packed_metadata.plans[4].cu_seqlens_cmp_k, 3
    )
    kwargs = dict(attn_sink=sink, sparse_indices=sparse, attention_masks=packed_metadata)

    output = inactive(*inputs[:-1], **kwargs)
    reference = enabled(*inputs[:-1], **kwargs)
    output.sum().backward()

    assert torch.equal(output, reference)
    assert idx_q.grad is None and idx_k.grad is None and idx_w.grad is None
    assert inactive._indexer_loss_acc.item() == 0


def test_golden_all_padding_selection_has_zero_indexer_gradients(monkeypatch, packed_metadata):
    core = _golden_core(monkeypatch, 1.0)
    inputs = _inputs()
    sparse = torch.full((1, 24, 3), -1, dtype=torch.long)

    output = core(*inputs[:-1], attn_sink=inputs[-1], sparse_indices=sparse, attention_masks=packed_metadata)
    (output.sum() * 0).backward()

    assert core._indexer_loss_acc.item() == 0
    for value in inputs[3:6]:
        assert torch.equal(value.grad, torch.zeros_like(value))


def test_golden_auxiliary_backward_reaches_all_indexer_parameters(monkeypatch, packed_metadata):
    config = _debugmodel().layers[2].attention.indexer
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        indexer = config.build()
        indexer.init_states(buffer_device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(23)
    x = torch.randn(1, 24, config.weights_proj.in_features, generator=generator)
    qr = torch.randn(1, 24, config.wq_b.in_features, generator=generator)
    core = _golden_core(monkeypatch, 1.0)
    q, window_k, cmp_k, _, _, _, sink = _inputs()
    idx_q, keys, idx_w = indexer(x, qr, positions=_doc_relative_positions(packed_metadata),
                                attention_masks=packed_metadata)
    idx_k = CPTokenDispatcher.Config().build().select(keys, packed_metadata.plans[4])
    sparse = select_compressed_slots(
        idx_q, idx_k, idx_w, packed_metadata.varlen.cu_seq_q, packed_metadata.plans[4].cu_seqlens_cmp_k, 3
    )

    output = core(q, window_k, cmp_k, idx_q, idx_k, idx_w, attn_sink=sink,
                  sparse_indices=sparse, attention_masks=packed_metadata)
    (output.sum() * 0).backward()

    for name, parameter in indexer.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    for value in (q, window_k, cmp_k, sink):
        assert torch.equal(value.grad, torch.zeros_like(value))
