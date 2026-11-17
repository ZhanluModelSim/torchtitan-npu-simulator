# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts of the DeepSeek-V3.2 reference sparse core.

The reference core is two attn-gym calls: ``lightning_indexer`` picks the slots and
``gather_attn`` attends to exactly those slots.  Every expected value below is derived from
the document layout and the raw projections -- never from the core under test -- so the
tests pin the numbering (document-local, ``-1`` padding, causal within a document), the
softmax's support (the selection alone: no window, no sink, no dense mask), and that the
indexer's ``(H * D) ** -0.5`` is handed to attn-gym exactly once.
"""

from types import SimpleNamespace

import pytest
import torch
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.attention import (
    VarlenAttention,
    VarlenMetadata,
    create_varlen_metadata_for_document,
)
from torchtitan.models.common.rope import CosSinRoPE

from torchtitan_npu.models.deepseek_v3_2 import model_registry
from torchtitan_npu.models.deepseek_v3_2.model import DeepSeekV32Model, Indexer, SparseInnerAttention
from torchtitan_npu.patches.torchtitan.distributed.varlen_cp import CPVarlenMetadata
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss

# Two documents in one packed row, the second one longer than ``topk``.
_POSITIONS = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])
_TOKENS = _POSITIONS.numel()
_DOC_STARTS = [0, 3]
_QUERY_HEADS = 2
_KV_LORA_RANK = 4
_ROPE_HEAD_DIM = 2
_INDEX_HEADS = 2
_INDEX_HEAD_DIM = 4
_INDEX_TOPK = 3
_SCALE = 0.5


@pytest.fixture(autouse=True)
def _clean_aux_loss_state():
    """The indexer-loss registry and the step denominator are process state."""
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def _masks():
    return create_varlen_metadata_for_document(_POSITIONS)


def _document_of(row: int) -> int:
    """The document a packed row belongs to, from the layout alone."""
    return sum(start <= row for start in _DOC_STARTS) - 1


def _dense_scores(idx_q_BLNH, idx_k_BL1H, idx_w_BLN) -> torch.Tensor:
    """The indexer's own score over every pair: ``sum_h w * relu(q . k)``, no mask.

    ``idx_w`` already carries the indexer's ``(H * D) ** -0.5``, and a positive factor is
    the same for every candidate, so the ranking here is the ranking the operator selects
    on however attn-gym spells the scaling.
    """
    dots = torch.einsum("blhd,bsd->blhs", idx_q_BLNH, idx_k_BL1H)
    return (dots.relu() * idx_w_BLN.unsqueeze(-1)).sum(dim=2)


def _selection_margin(idx_q_BLNH, idx_k_BL1H, idx_w_BLN) -> float:
    """The smallest gap between a row's last pick and its best rejected slot.

    ``relu`` makes every non-positive candidate score exactly zero, so a row whose reach
    includes two such candidates has a real tie and no oracle can say which one the
    operator must return.  The inputs below keep the gap wide enough to compare sets.
    """
    scores = _dense_scores(idx_q_BLNH, idx_k_BL1H, idx_w_BLN)[0]
    margin = float("inf")
    for row in range(_TOKENS):
        start = _DOC_STARTS[_document_of(row)]
        ranked = sorted((float(scores[row, slot]) for slot in range(start, row + 1)), reverse=True)
        if len(ranked) > _INDEX_TOPK:
            margin = min(margin, ranked[_INDEX_TOPK - 1] - ranked[_INDEX_TOPK])
    return margin


def _expected_selection(idx_q_BLNH, idx_k_BL1H, idx_w_BLN) -> list[list[int]]:
    """Top-k document-local slots per row, or the independent oracle for the operator.

    A slot is selectable when it is in the query's own document and at or before the
    query, which is exactly the packed causal contract; unreachable slots stay ``-1``.
    """
    scores = _dense_scores(idx_q_BLNH, idx_k_BL1H, idx_w_BLN)[0]
    selection = []
    for row in range(_TOKENS):
        document = _document_of(row)
        start = _DOC_STARTS[document]
        end = _DOC_STARTS[document + 1] if document + 1 < len(_DOC_STARTS) else _TOKENS
        candidates = list(range(start, min(end, row + 1)))
        ranked = sorted(candidates, key=lambda slot: (-float(scores[row, slot]), slot))
        picked = ranked[:_INDEX_TOPK]
        selection.append([slot - start for slot in picked] + [-1] * (_INDEX_TOPK - len(picked)))
    return selection


def _inputs(seed: int):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, _TOKENS)
    return {
        "q_nope_BLNH": torch.randn(*shape, _QUERY_HEADS, _KV_LORA_RANK, generator=generator),
        "k_nope_BL1H": torch.randn(*shape, 1, _KV_LORA_RANK, generator=generator),
        "q_rope_BLNH": torch.randn(*shape, _QUERY_HEADS, _ROPE_HEAD_DIM, generator=generator),
        "k_rope_BL1H": torch.randn(*shape, 1, _ROPE_HEAD_DIM, generator=generator),
        "idx_q_BLNH": torch.randn(*shape, _INDEX_HEADS, _INDEX_HEAD_DIM, generator=generator),
        # The index keys are one head, so the tensor is the row itself (no head axis).
        "idx_k_BL1H": torch.randn(*shape, _INDEX_HEAD_DIM, generator=generator),
        "idx_w_BLN": torch.randn(*shape, _INDEX_HEADS, generator=generator),
    }


def _core(seed: int = 0) -> SparseInnerAttention:
    torch.manual_seed(seed)
    return SparseInnerAttention(SparseInnerAttention.Config(index_topk=_INDEX_TOPK)).eval()


def _indexer_config() -> Indexer.Config:
    return Indexer.Config(
        dim=8,
        q_lora_rank=6,
        index_n_heads=_INDEX_HEADS,
        index_head_dim=_INDEX_HEAD_DIM,
        rope_head_dim=_ROPE_HEAD_DIM,
        index_topk=_INDEX_TOPK,
        wq_b=Linear.Config(in_features=6, out_features=_INDEX_HEADS * _INDEX_HEAD_DIM),
        wk=Linear.Config(in_features=8, out_features=_INDEX_HEAD_DIM),
        k_norm=LayerNorm.Config(normalized_shape=_INDEX_HEAD_DIM),
        weights_proj=Linear.Config(in_features=8, out_features=_INDEX_HEADS),
        rope=CosSinRoPE.Config(dim=_ROPE_HEAD_DIM, max_seq_len=_TOKENS, theta=10000.0),
    )


def test_the_selection_carries_the_scores_of_the_slots_it_picked():
    """``select`` returns both halves: attn-gym's indices and the indexer's logits at them.

    The scores keep the folded ``idx_w`` convention -- ``(H * D) ** -0.5`` is already in it,
    because the fused kernel reads the weights that way -- while the selection call hands
    attn-gym the unfolded weights, since its indexer applies that factor to its own score.
    A uniform positive factor ranks the same slots, so only the scores can pin which
    convention each half uses.
    """
    inputs = _inputs(seed=10)
    selection, scores = Indexer.select(
        inputs["idx_q_BLNH"],
        inputs["idx_k_BL1H"],
        inputs["idx_w_BLN"],
        _masks(),
        _INDEX_TOPK,
    )

    expected = _dense_scores(inputs["idx_q_BLNH"], inputs["idx_k_BL1H"], inputs["idx_w_BLN"])
    valid = selection[0] >= 0
    document_starts = torch.tensor([_DOC_STARTS[_document_of(row)] for row in range(_TOKENS)]).unsqueeze(-1)
    picked_rows = (selection[0].long() + document_starts).clamp_min(0)
    expected_scores = expected[0].gather(-1, picked_rows)

    assert scores.shape == selection.shape == (1, _TOKENS, _INDEX_TOPK)
    torch.testing.assert_close(scores[0][valid], expected_scores[valid], rtol=1e-5, atol=1e-6)
    # An unused slot is marked in the scores, not merely numbered ``-1``.
    assert torch.isneginf(scores[0][~valid]).all()


def test_the_scores_stay_fp32_under_a_low_precision_input():
    """The score pass upcasts, as the removed dense scores and attn-gym's own indexer do.

    The selection call keeps the module's dtype -- attn-gym validates the weights against the
    queries -- but a bf16 score pass costs about 1.1 max relative error on the student's logits
    on 910C, and the indexer's gradient follows it, so the pass that trains the indexer is
    spelled in fp32.
    """
    inputs = {name: value.to(torch.bfloat16) for name, value in _inputs(seed=10).items()}
    selection, scores = Indexer.select(
        inputs["idx_q_BLNH"],
        inputs["idx_k_BL1H"],
        inputs["idx_w_BLN"],
        _masks(),
        _INDEX_TOPK,
    )

    assert scores.dtype == torch.float32
    # The same logits from the same (bf16) operands, accumulated by hand in fp32.
    q_THD = inputs["idx_q_BLNH"].reshape(_TOKENS, _INDEX_HEADS, _INDEX_HEAD_DIM).float()
    keys_TD = inputs["idx_k_BL1H"].reshape(_TOKENS, _INDEX_HEAD_DIM).float()
    weights_TH = inputs["idx_w_BLN"].reshape(_TOKENS, _INDEX_HEADS).float()
    valid = selection[0] >= 0
    document_starts = torch.tensor([_DOC_STARTS[_document_of(row)] for row in range(_TOKENS)]).unsqueeze(-1)
    picked_rows = (selection[0].long() + document_starts).clamp_min(0)
    dots_THK = torch.einsum("thd,tkd->thk", q_THD, keys_TD[picked_rows])
    expected_TH = (dots_THK.relu() * weights_TH.unsqueeze(-1)).sum(dim=1)

    torch.testing.assert_close(scores[0][valid], expected_TH[valid], rtol=1e-5, atol=1e-5)


def test_the_selection_is_document_local_causal_and_padded_with_minus_one():
    """``lightning_indexer`` returns the slots ``gather_attn`` reads: local, causal, ``-1``."""
    inputs = _inputs(seed=10)
    assert _selection_margin(inputs["idx_q_BLNH"], inputs["idx_k_BL1H"], inputs["idx_w_BLN"]) > 1e-3

    selection, _ = Indexer.select(
        inputs["idx_q_BLNH"],
        inputs["idx_k_BL1H"],
        inputs["idx_w_BLN"],
        _masks(),
        _INDEX_TOPK,
    )

    assert selection.shape == (1, _TOKENS, _INDEX_TOPK)
    assert selection.dtype == torch.int32
    expected = _expected_selection(inputs["idx_q_BLNH"], inputs["idx_k_BL1H"], inputs["idx_w_BLN"])
    for row, (actual, want) in enumerate(zip(selection[0].tolist(), expected, strict=True)):
        # Order is attn-gym's to choose; the set of slots and the padding are not.
        assert sorted(slot for slot in actual if slot >= 0) == sorted(slot for slot in want if slot >= 0), row
        assert actual.count(-1) == want.count(-1), row
        # Every pick stays inside its own document's pool and at or before the query.
        document_start = _DOC_STARTS[_document_of(row)]
        assert all(0 <= slot <= row - document_start for slot in actual if slot >= 0), row
    # A row reaches only its own prefix, so a short document cannot fill the selection:
    # the first row of the first document can reach one slot, its last row all three.
    assert selection[0, 0].tolist().count(-1) == _INDEX_TOPK - 1
    assert selection[0, 2].tolist().count(-1) == 0


def test_the_core_equals_a_hand_masked_sparse_softmax():
    """One softmax over exactly the selected slots, with ``k_nope`` as MLA's absorbed value."""
    inputs = _inputs(seed=10)
    assert _selection_margin(inputs["idx_q_BLNH"], inputs["idx_k_BL1H"], inputs["idx_w_BLN"]) > 1e-3
    core = _core()

    output_BLNH = core(attention_masks=_masks(), scale=_SCALE, **inputs)

    k_BLSD = torch.cat([inputs["k_nope_BL1H"][0, :, 0], inputs["k_rope_BL1H"][0, :, 0]], dim=-1)
    q_BLHD = torch.cat([inputs["q_nope_BLNH"][0], inputs["q_rope_BLNH"][0]], dim=-1)
    expected = torch.zeros_like(output_BLNH)
    for row, slots in enumerate(_expected_selection(inputs["idx_q_BLNH"], inputs["idx_k_BL1H"], inputs["idx_w_BLN"])):
        picked = [_DOC_STARTS[_document_of(row)] + slot for slot in slots if slot >= 0]
        logits = (q_BLHD[row] @ k_BLSD[picked].transpose(-1, -2)).mul(_SCALE)
        probabilities = torch.softmax(logits, dim=-1)
        expected[0, row] = probabilities @ inputs["k_nope_BL1H"][0, picked, 0]

    assert output_BLNH.shape == (1, _TOKENS, _QUERY_HEADS, _KV_LORA_RANK)
    torch.testing.assert_close(output_BLNH, expected, rtol=1e-5, atol=1e-6)


def test_a_document_never_reads_another_documents_keys():
    """Rewriting one document's keys leaves the other document's rows untouched."""
    inputs = _inputs(seed=1)
    core = _core()
    before = core(attention_masks=_masks(), scale=_SCALE, **inputs)

    perturbed = dict(inputs)
    perturbed["k_nope_BL1H"] = inputs["k_nope_BL1H"].clone()
    perturbed["k_nope_BL1H"][:, : _DOC_STARTS[1]] += 100.0
    after = core(attention_masks=_masks(), scale=_SCALE, **perturbed)

    assert not torch.allclose(before[:, : _DOC_STARTS[1]], after[:, : _DOC_STARTS[1]])
    assert torch.equal(before[:, _DOC_STARTS[1] :], after[:, _DOC_STARTS[1] :])


def test_the_indexer_loss_reads_the_selected_rows_of_every_document():
    """The teacher gathers the picked rows of the packed sequence, not of one document.

    A selection is numbered inside its own document, while the teacher reads the keys' rows,
    so a loss handed the document-local indices would score the second document's queries
    against the first document's keys -- a wrong teacher for every document but the first.
    """
    inputs = _inputs(seed=10)
    denominator = 8.0
    AuxLoss.set_step_denominator(torch.tensor(denominator))
    core = _core()
    core.train()

    core(attention_masks=_masks(), scale=_SCALE, **inputs)

    k_BLSD = torch.cat([inputs["k_nope_BL1H"][0, :, 0], inputs["k_rope_BL1H"][0, :, 0]], dim=-1)
    q_BLHD = torch.cat([inputs["q_nope_BLNH"][0], inputs["q_rope_BLNH"][0]], dim=-1)
    kl = 0.0
    for row, slots in enumerate(_expected_selection(inputs["idx_q_BLNH"], inputs["idx_k_BL1H"], inputs["idx_w_BLN"])):
        picked = [_DOC_STARTS[_document_of(row)] + slot for slot in slots if slot >= 0]
        teacher_logits = (q_BLHD[row] @ k_BLSD[picked].transpose(-1, -2)).mul(_SCALE)
        teacher = torch.softmax(teacher_logits, dim=-1).mean(dim=0)
        student_logits = (
            (inputs["idx_q_BLNH"][0, row] @ inputs["idx_k_BL1H"][0, picked].transpose(-1, -2)).relu()
            * inputs["idx_w_BLN"][0, row].unsqueeze(-1)
        ).sum(dim=0)
        student = torch.softmax(student_logits, dim=-1)
        kl += float((teacher * (teacher.clamp_min(1e-30).log() - student.log())).sum())

    assert float(core.indexer_loss.instance_acc) == pytest.approx(kl / denominator, rel=1e-5)


def test_the_reference_core_rejects_context_parallel_metadata():
    """A CP rank holds a document's middle, which attn-gym's prefix-aligned causal cannot express."""
    inputs = _inputs(seed=0)
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    cp_metadata = CPVarlenMetadata(
        cu_seq_q=cu,
        cu_seq_k=torch.tensor([0, 3, 5], dtype=torch.int32),
        max_q=3,
        max_k=3,
        k_global_gather_indices=torch.arange(8),
    )

    try:
        _core()(attention_masks=cp_metadata, scale=_SCALE, **inputs)
    except ValueError as error:
        assert "context-parallel" in str(error)
        assert "override.deepseek_v3_2.sparse_attn.asc" in str(error)
    else:
        raise AssertionError("the reference core must refuse context-parallel metadata")


def test_the_shipped_layers_dispatch_to_a_varlen_mask():
    """The mask seam reads the inner attention's config, so the model must declare a varlen one.

    A flex-typed core would make the model build a ``BlockMask``, and the reference forward --
    which reads the metadata's packed offsets -- would never see it.  The owner below is the
    stand-in the dispatch reads from (``Decoder.get_attention_masks`` touches only
    ``config.first_attention``), so this runs without materializing a model.
    """
    layer_cfg = model_registry("debugmodel").model.layers[0]
    owner = SimpleNamespace(config=SimpleNamespace(first_attention=layer_cfg.attention))

    masks = DeepSeekV32Model.get_attention_masks(owner, _POSITIONS)

    assert isinstance(layer_cfg.attention.inner_attention, SparseInnerAttention.Config)
    assert issubclass(SparseInnerAttention.Config, VarlenAttention.Config)
    assert isinstance(masks, VarlenMetadata)
    assert masks.cu_seq_q.tolist() == [0, 3, 8]
