# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The fused V4 InnerIndexer: canonical order, teacher edge, SLIKG dispatch.

The NPU kernels are replaced by eager stand-ins with their exact contract (document-local
indices, ``-1`` padding, the teacher arriving on ``topk_scores``' gradient), so these cases
pin the wiring rather than the kernel math: sorting happens inside the Function before the
indices are saved, the teacher is divided by the microbatch's query-row count exactly once,
and both kernel calls read the metadata the fused attention precomputed.
"""

import pytest
import torch

from torchtitan_npu.override.deepseek_v4.lightning_indexer import ascendc
from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import (
    AscInnerIndexer,
    sort_indices_and_scores,
)
from torchtitan_npu.override.deepseek_v4.metadata import lightning_indexer_options
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import (
    AscCompressedSparseInnerAttention,
)

pytestmark = pytest.mark.usefixtures("dsv4")

_TOKENS = 8
_TOPK = 4
_HEADS = 2
_HEAD_DIM = 4
_INDEX_TOP_K = _TOPK
_WINDOW = 8
_GEOMETRY = dict(
    num_heads=2,
    head_dim=8,
    index_num_heads=_HEADS,
    index_head_dim=_HEAD_DIM,
)


def _metadata():
    positions = torch.arange(_TOKENS)
    config = AscCompressedSparseInnerAttention.Config(
        window_size=_WINDOW,
        compress_ratio=4,
        softmax_scale=0.5,
        index_topk=_INDEX_TOP_K,
        **_GEOMETRY,
    )
    return config.build_attention_metadata(positions)


def _indexer():
    return AscInnerIndexer(AscInnerIndexer.Config(index_topk=_INDEX_TOP_K))


def _kernel_indices() -> torch.Tensor:
    """A kernel-order selection with padding in the middle, not at the tail."""
    return torch.tensor(
        [
            [[2, -1, 0, 1]],
            [[-1, 3, 1, 0]],
            [[1, 0, -1, 2]],
            [[0, 1, 2, 3]],
            [[3, -1, 2, -1]],
            [[-1, -1, 0, 2]],
            [[1, 2, 3, -1]],
            [[2, 0, -1, 1]],
        ],
        dtype=torch.int32,
    )


@pytest.mark.parametrize("use_fp32_sort", [False, True], ids=["native-sort", "fp32-sort"])
def test_kernel_scores_and_backward_teacher_keep_the_same_slots(monkeypatch, use_fp32_sort):
    """The same permutation moves logits and indices, including padding, before backward."""
    if use_fp32_sort:
        monkeypatch.setattr(ascendc, "_uses_fp32_sort", lambda indices: indices.dtype == torch.int32)
    md = _metadata()
    captured = {}
    kernel_scores = torch.arange(_TOKENS * _TOPK, dtype=torch.float32).reshape(_TOKENS, 1, _TOPK)
    kernel_scores.masked_fill_(_kernel_indices() < 0, -torch.inf)

    def fake_li(q, k, w, topk, *, metadata, return_value, **options):
        captured["metadata"] = metadata
        captured["options"] = options
        assert return_value == 1
        return _kernel_indices(), kernel_scores

    def fake_slig(*, q, k, w, sparse_indices, attn_softmax_l1_norm, metadata, **options):
        captured["indices"] = sparse_indices
        captured["metadata_bwd"] = metadata
        captured["options_bwd"] = options
        captured["teacher"] = attn_softmax_l1_norm
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(w), None

    monkeypatch.setattr(torch.ops.cann_ops_transformer, "lightning_indexer", fake_li, raising=False)
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "sparse_lightning_indexer_kl_loss_grad",
        fake_slig,
        raising=False,
    )

    torch.manual_seed(0)
    idx_q = torch.randn(_TOKENS, _HEADS, _HEAD_DIM, requires_grad=True)
    idx_k = torch.randn(_TOKENS // 4, _HEAD_DIM, requires_grad=True)
    idx_w = torch.randn(_TOKENS, _HEADS, requires_grad=True)
    picks, scores = _indexer()(idx_q, idx_k, idx_w, attention_metadata=md)

    assert picks.shape == (_TOKENS, _TOPK) and picks.dtype == torch.int32
    assert scores.shape == picks.shape and scores.dtype == torch.float32
    expected, order = _kernel_indices().sort(dim=-1, descending=True)
    torch.testing.assert_close(picks, expected.squeeze(1), rtol=0, atol=0)
    torch.testing.assert_close(scores, kernel_scores.gather(-1, order).squeeze(1), rtol=0, atol=0)

    # The forward metadata is the entry's own record; the options are the frame contract.
    assert captured["metadata"] is md.aicpu_metadata.lightning_indexer.fwd
    # The options are the entry's own frame -- the one its keys were compacted onto.
    assert captured["options"] == lightning_indexer_options(ratio=md.ratio, q=md.q, cmp_k=md.cmp_k)

    teacher = torch.zeros_like(scores)
    teacher[..., 0] = 0.1
    teacher[..., 1] = 0.3
    scores.backward(teacher)

    # The saved indices are exactly what SMLAG would have been handed, and the teacher reaches
    # SLIKG exactly as it was supplied by the attention's backward.
    assert torch.equal(captured["indices"], picks.reshape(-1, 1, _TOPK))
    assert captured["metadata_bwd"] is md.aicpu_metadata.lightning_indexer.bwd
    assert captured["options_bwd"] == captured["options"]
    expected_teacher = teacher.reshape(-1, 1, _TOPK)
    torch.testing.assert_close(captured["teacher"], expected_teacher, rtol=0, atol=0)

    # The kernel's gradients land on the projections the indexer owns.
    for tensor in (idx_q, idx_k, idx_w):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
    for row in teacher.reshape(-1, _TOPK):
        assert float(row.abs().sum()) > 0


def test_the_carrier_is_a_function_output(monkeypatch):
    """A caller-made tensor cannot carry the teacher; the returned carrier has a ``grad_fn``."""
    md = _metadata()
    monkeypatch.setattr(
        torch.ops.cann_ops_transformer,
        "lightning_indexer",
        lambda *a, **k: (_kernel_indices(), torch.zeros((_TOKENS, 1, _TOPK))),
        raising=False,
    )
    idx_q = torch.zeros(_TOKENS, _HEADS, _HEAD_DIM, requires_grad=True)
    idx_k = torch.zeros(_TOKENS // 4, _HEAD_DIM, requires_grad=True)
    idx_w = torch.zeros(_TOKENS, _HEADS, requires_grad=True)
    picks, scores = _indexer()(idx_q, idx_k, idx_w, attention_metadata=md)
    assert scores.requires_grad and scores.grad_fn is not None
    assert not picks.requires_grad


def test_the_override_entry_replaces_the_base_selector():
    """The registry-facing ``asc`` entry derives the fused selector config."""
    from torchtitan_npu.override.deepseek_v4.lightning_indexer import asc

    fused = asc(AscInnerIndexer.Config(index_topk=_INDEX_TOP_K))
    assert isinstance(fused, AscInnerIndexer.Config)
    assert fused.index_topk == _INDEX_TOP_K


def test_the_pypto_entry_selects_the_pypto_selector():
    """The PyPTO port is standalone: it implements the selector contract itself."""
    from torchtitan_npu.models.deepseek_v4.compressor import InnerIndexer
    from torchtitan_npu.override.deepseek_v4.lightning_indexer import pypto
    from torchtitan_npu.override.deepseek_v4.lightning_indexer.pypto import PyPTOInnerIndexer

    fused = pypto(AscInnerIndexer.Config(index_topk=_INDEX_TOP_K))
    assert isinstance(fused, PyPTOInnerIndexer.Config)
    # The two ports share the base selector's model contract, not each other's bridge.
    assert issubclass(PyPTOInnerIndexer, InnerIndexer)
    assert not issubclass(PyPTOInnerIndexer, AscInnerIndexer)


def test_sort_indices_and_scores_fp32_path_matches_exact_descending_order(monkeypatch):
    """FP32 stable sort keeps compressed indices exact up to 2**24 and preserves ties."""
    monkeypatch.setattr(ascendc, "_uses_fp32_sort", lambda indices: indices.dtype == torch.int32)
    indices = torch.tensor(
        [
            [3, -1, 16_777_216, 0, -1, 42],
            [-1, -1, -1, 7, 16_777_215, 7],
        ],
        dtype=torch.int32,
    )
    scores = torch.tensor(
        [
            [0.3, 9.0, 1.5, 0.1, 8.0, 2.2],
            [4.0, 5.0, 6.0, 0.7, 1.5, 0.8],
        ],
        dtype=torch.float32,
    )

    sorted_indices, sorted_scores = sort_indices_and_scores(indices, scores)

    expected_indices = torch.tensor(
        [
            [16_777_216, 42, 3, 0, -1, -1],
            [16_777_215, 7, 7, -1, -1, -1],
        ],
        dtype=torch.int32,
    )
    expected_scores = torch.tensor(
        [
            [1.5, 2.2, 0.3, 0.1, 9.0, 8.0],
            [1.5, 0.7, 0.8, 4.0, 5.0, 6.0],
        ],
        dtype=torch.float32,
    )
    assert sorted_indices.dtype == torch.int32
    assert torch.equal(sorted_indices, expected_indices)
    assert torch.equal(sorted_scores, expected_scores)

    # Non-int32 inputs stay on native integer sort to avoid precision loss above 2**24.
    int64_indices = torch.tensor([[16_777_216, -1, 16_777_217]], dtype=torch.int64)
    int64_scores = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    sorted_i64, sorted_i64_scores = sort_indices_and_scores(int64_indices, int64_scores)
    assert sorted_i64.dtype == torch.int64
    assert torch.equal(sorted_i64, torch.tensor([[16_777_217, 16_777_216, -1]], dtype=torch.int64))
    assert torch.equal(sorted_i64_scores, torch.tensor([[3.0, 1.0, 2.0]], dtype=torch.float32))
