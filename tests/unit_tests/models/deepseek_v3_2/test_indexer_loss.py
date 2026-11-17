# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU contracts of DeepSeek-V3.2's sparse indexer loss.

The layer builds the loss from the real model config, the loss is an `AuxLoss`, and the reference
(non-fused) forward hands `inject` the sum over this microbatch's tokens -- the quantity the
framework divides by the step's global valid-token count.  The loss reads two tensors of the same
shape: the attention's logits and the indexer's, both already gathered at the selection (`B, Lq, K`
rather than the dense `B, Lq, Lkv` grid), so these cases pin the reduction and the guards and not
how the caller produced either.  The fused NPU kernel path that replaces the reference forward on
device is covered by the device-side integration tests, not here.
"""

import pytest
import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import create_varlen_metadata_for_document

from torchtitan_npu.models.deepseek_v3_2 import model_registry
from torchtitan_npu.models.deepseek_v3_2.model import SparseIndexerLoss
from torchtitan_npu.patches.torchtitan.models.common.aux_loss import AuxLoss


def _masks(num_tokens: int):
    """One document covering the whole row, so a document-local slot is its own row.

    Positions that reset to zero start a document, hence the increasing positions here: the
    selection's numbering and the keys' rows then coincide, which is what these cases compare.
    """
    return create_varlen_metadata_for_document(torch.arange(num_tokens).unsqueeze(0))


_COEFF = 2.0
_DENOMINATOR = 8.0
_NUM_HEADS = 4
_TOPK = 2


@pytest.fixture(autouse=True)
def _clean_aux_loss_state():
    """The registry and the step denominator are process state; restore them around every test."""
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None
    yield
    AuxLoss._group_counts.clear()
    AuxLoss.group_acc.clear()
    AuxLoss._step_denominator = None


def _indexer_inputs(seed: int):
    """The reference forward's operands: one query token over two keys, one index head.

    The selection is both keys in order, so nothing is masked out and the KL stays well defined.
    """
    generator = torch.Generator().manual_seed(seed)
    q_BLNH = torch.randn(1, 1, 1, _NUM_HEADS, generator=generator, requires_grad=True)
    k_BL1H = torch.randn(1, _TOPK, 1, _NUM_HEADS, generator=generator, requires_grad=True)
    topk_indices = torch.arange(_TOPK).view(1, 1, _TOPK)
    student_logits = torch.randn(1, 1, _TOPK, generator=generator)
    return q_BLNH, k_BL1H, topk_indices, student_logits


def _kl_row(q_BLNH, k_BL1H, topk_indices, student_logits, row: int) -> torch.Tensor:
    """One query row's KL through ``F.kl_div``: an independent implementation."""
    picked_KH = k_BL1H[0, :, 0, :][topk_indices[0, row]]
    p_K = F.softmax(q_BLNH[0, row, 0] @ picked_KH.T, dim=-1)
    q_pred_K = F.softmax(student_logits[0, row], dim=-1)
    return F.kl_div(q_pred_K.log(), p_K, reduction="sum")


def _kl_sum(q_BLNH, k_BL1H, topk_indices, student_logits) -> torch.Tensor:
    """The same KL summed over every query row, which is the reduction the caller must pass on."""
    return sum(_kl_row(q_BLNH, k_BL1H, topk_indices, student_logits, row) for row in range(q_BLNH.size(1)))


def test_the_layer_builds_an_aux_loss_from_the_model_config():
    """The real registry path: a DeepSeek-V3.2 layer's indexer loss is the backported AuxLoss."""
    layer = model_registry("debugmodel").model.layers[0]

    loss_config = layer.attention.inner_attention.indexer_loss
    loss = loss_config.build()

    assert isinstance(loss_config, SparseIndexerLoss.Config)
    assert isinstance(loss, AuxLoss)
    # Per-token-additive: ranks contribute distinct data, so the metric reduces over dp+cp.
    assert loss.reduce_mesh == "loss"
    assert loss.coeff == 1.0


def test_the_reference_loss_injects_the_token_sum_over_the_step_denominator():
    """Hand-derived: the register holds ``sum(KL)/denominator``, with no coefficient in it."""
    AuxLoss.set_step_denominator(torch.tensor(_DENOMINATOR))
    loss = SparseIndexerLoss(SparseIndexerLoss.Config(coeff=_COEFF))
    q_BLNH, k_BL1H, topk_indices, student_logits = _indexer_inputs(seed=0)
    carrier = torch.zeros(1)

    out = loss(q_BLNH, k_BL1H, 1.0, topk_indices, student_logits, _masks(1), carrier=carrier)

    expected = float(_kl_sum(q_BLNH.detach(), k_BL1H.detach(), topk_indices, student_logits)) / _DENOMINATOR
    assert torch.equal(out, carrier)
    assert float(loss.instance_acc) == pytest.approx(expected, rel=1e-4, abs=1e-7)


def test_the_injected_indexer_gradient_scales_inversely_with_the_step_denominator():
    """Doubling the step's token count halves the indexer gradient: both factors live in ``inject``."""
    grads = []
    for denominator in (4.0, _DENOMINATOR):
        AuxLoss.set_step_denominator(torch.tensor(denominator))
        loss = SparseIndexerLoss(SparseIndexerLoss.Config(coeff=_COEFF))
        q_BLNH, k_BL1H, topk_indices, student_logits = _indexer_inputs(seed=1)
        carrier = torch.zeros(1, requires_grad=True)

        loss(q_BLNH, k_BL1H, 1.0, topk_indices, student_logits, _masks(1), carrier=carrier).sum().backward()

        assert torch.equal(carrier.grad, torch.ones_like(carrier))
        grads.append(k_BL1H.grad.clone())

    torch.testing.assert_close(grads[1] * 2.0, grads[0])


def test_the_reference_loss_sums_over_every_query_row():
    """The register is the row *sum*: the operator's loss is a sum over query rows, and the
    framework divides it by the step's token count.  With a single row the sum and the mean
    coincide, so only a multi-row micro-batch can hold the two apart."""
    AuxLoss.set_step_denominator(torch.tensor(_DENOMINATOR))
    loss = SparseIndexerLoss(SparseIndexerLoss.Config(coeff=_COEFF))
    seq_len = 3
    generator = torch.Generator().manual_seed(2)
    q_BLNH = torch.randn(1, seq_len, 1, _NUM_HEADS, generator=generator, requires_grad=True)
    k_BL1H = torch.randn(1, _TOPK, 1, _NUM_HEADS, generator=generator, requires_grad=True)
    topk_indices = torch.arange(_TOPK).view(1, 1, _TOPK).expand(1, seq_len, _TOPK).contiguous()
    student_logits = torch.randn(1, seq_len, _TOPK, generator=generator)
    carrier = torch.zeros(1)

    loss(q_BLNH, k_BL1H, 1.0, topk_indices, student_logits, _masks(seq_len), carrier=carrier)

    expected = (
        sum(
            float(_kl_row(q_BLNH.detach(), k_BL1H.detach(), topk_indices, student_logits, row))
            for row in range(seq_len)
        )
        / _DENOMINATOR
    )
    assert float(loss.instance_acc) == pytest.approx(expected, rel=1e-4, abs=1e-7)


def test_the_gradient_is_the_analytic_y_minus_p_even_when_the_student_saturates():
    """The indexer gradient is ``Z * Y - mass`` on every slot, saturated ones included.

    The operator emits that gradient unconditionally, while an epsilon inside the log-probabilities
    (``log(y + eps)``) makes autograd collapse to ~0 for a slot with ``y << eps``.  This case drives
    the student into exactly that regime -- its second slot lands below 1e-8 -- and compares the
    gradient with the analytic expression, which the epsilon spelling misses by orders of magnitude.
    """
    AuxLoss.set_step_denominator(torch.tensor(_DENOMINATOR))
    loss = SparseIndexerLoss(SparseIndexerLoss.Config(coeff=_COEFF))
    generator = torch.Generator().manual_seed(3)
    q_BLNH = torch.randn(1, 1, 1, _NUM_HEADS, generator=generator)
    k_BL1H = torch.randn(1, _TOPK, 1, _NUM_HEADS, generator=generator)
    topk_indices = torch.arange(_TOPK).view(1, 1, _TOPK)
    student_logits = torch.tensor([[[0.0, -40.0]]], requires_grad=True)
    carrier = torch.zeros(1)
    scale = 0.5

    # The teacher's logits come from the keys gathered per query row, exactly as the attention
    # sees them; with a single index head the head average is a no-op.
    gathered_KH = k_BL1H.float().squeeze(2)[0][topk_indices[0, 0]]
    teacher_logits_K = (q_BLNH.float()[0, 0, 0] @ gathered_KH.T).mul(scale)
    mass_K = torch.softmax(teacher_logits_K, dim=-1)
    student_K = torch.softmax(student_logits[0, 0], dim=-1)
    assert float(student_K[1]) < 1e-8, "the case must sit in the saturated regime"

    loss(q_BLNH, k_BL1H, scale, topk_indices, student_logits, _masks(1), carrier=carrier).sum().backward()

    expected = (student_K * mass_K.sum() - mass_K) * (_COEFF / _DENOMINATOR)
    torch.testing.assert_close(student_logits.grad[0, 0], expected, rtol=1e-5, atol=1e-9)


def test_slots_without_a_selection_contribute_nothing_and_never_produce_nan():
    """``-1`` slots are outside both normalizations, and a row of them contributes nothing at all.

    The operator processes exactly the causal prefix of the selection and never inspects the
    padding; masking with a real ``-inf`` here would give ``0 * -inf`` and poison the backward.
    """
    AuxLoss.set_step_denominator(torch.tensor(_DENOMINATOR))
    loss = SparseIndexerLoss(SparseIndexerLoss.Config(coeff=_COEFF))
    generator = torch.Generator().manual_seed(4)
    q_BLNH = torch.randn(1, 1, 1, _NUM_HEADS, generator=generator)
    k_BL1H = torch.randn(1, _TOPK, 1, _NUM_HEADS, generator=generator)
    student_logits = torch.randn(1, 1, _TOPK, generator=generator, requires_grad=True)
    carrier = torch.zeros(1)
    scale = 0.5

    # One usable slot: the teacher mass concentrates on it, so both distributions are one-hot and
    # the KL is zero.
    one_slot = torch.tensor([[[0, -1]]])
    loss(q_BLNH, k_BL1H, scale, one_slot, student_logits, _masks(1), carrier=carrier).sum().backward()

    assert float(loss.instance_acc) == pytest.approx(0.0)
    assert torch.isfinite(student_logits.grad).all()

    # No usable slot at all: the row is skipped, not NaN'd.
    loss.instance_acc.zero_()
    student_logits.grad = None
    empty_row = torch.tensor([[[-1, -1]]])
    loss(q_BLNH, k_BL1H, scale, empty_row, student_logits, _masks(1), carrier=carrier).sum().backward()

    assert float(loss.instance_acc) == 0.0
    assert torch.isfinite(student_logits.grad).all()
    assert torch.equal(student_logits.grad, torch.zeros_like(student_logits.grad))
