# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import dataclasses
from importlib import import_module

import pytest
import torch

from torchtitan_npu.models.deepseek_v3_2 import model_registry as deepseek_v3_2_model_registry
from torchtitan_npu.models.deepseek_v3_2.config_registry import (
    deepseek_v3_2_debugmodel,
)
from torchtitan_npu.models.deepseek_v4.config_registry import (
    deepseek_v4_debugmodel,
)
from torchtitan_npu.models.deepseek_v4.mtp import (
    MTPChunkedLossWrapper,
    MTPModelOutput,
    prepare_mtp_batch,
)

_community_loss = import_module("torchtitan.components.loss")
_community_mtp = import_module("torchtitan.models.deepseek_v3.mtp")


def test_dsv4_mtp_batch_precedes_cp_sharding():
    tokens = torch.arange(8).unsqueeze(0)
    labels = tokens + 1
    positions = torch.arange(8).unsqueeze(0)

    batch = prepare_mtp_batch(tokens, labels, positions, num_mtp_layers=1)

    # One head-tail CP rank owns disjoint prefix and suffix pieces. Selecting
    # from the globally shifted tensor preserves the future token at index 1;
    # rolling the already-local tensor would incorrectly select token 6.
    local_indices = torch.tensor([0, 1, 6, 7])
    prepared_local = batch.input_tokens[:, local_indices, 0]
    incorrectly_rolled_local = _community_mtp.roll_mtp_sequence(
        tokens[:, local_indices],
        shift=1,
        fill_value=0,
    )

    torch.testing.assert_close(prepared_local, torch.tensor([[1, 2, 7, 0]]))
    assert prepared_local[0, 1] != incorrectly_rolled_local[0, 1]


def test_dsv32_default_config_disables_mtp():
    config = deepseek_v3_2_debugmodel()

    assert not config.model_spec.model.mtp_layers
    assert isinstance(config.loss, _community_loss.ChunkedLossWrapper.Config)


def test_dsv3_mtp_uses_community_implementation():
    assert _community_mtp.MTPDecoder.forward.__module__ == "torchtitan.models.deepseek_v3.mtp"
    assert _community_mtp.MTPLoss.__call__.__module__ == "torchtitan.models.deepseek_v3.mtp"


def test_dsv32_mtp_keeps_community_cp_restriction():
    config = deepseek_v3_2_debugmodel()
    config.model_spec = deepseek_v3_2_model_registry(
        "debugmodel",
        num_mtp_layers=1,
    )
    config.training = dataclasses.replace(
        config.training,
        global_batch_size=8,
    )
    config.parallelism = dataclasses.replace(
        config.parallelism,
        context_parallel_degree=2,
    )

    with pytest.raises(NotImplementedError, match="context parallelism"):
        config.model_spec.model.update_from_config(config=config)


def test_dsv4_mtp_accepts_cp_configuration():
    config = deepseek_v4_debugmodel()
    config.training = dataclasses.replace(
        config.training,
        global_batch_size=8,
    )
    config.parallelism = dataclasses.replace(
        config.parallelism,
        context_parallel_degree=2,
    )

    config.model_spec.model.update_from_config(config=config)


def test_prepare_mtp_batch_respects_packed_sequence_boundaries():
    tokens = torch.arange(10, 18).unsqueeze(0)
    labels = tokens + 100
    positions = torch.tensor([[0, 1, 2, 0, 1, 0, 1, 2]])

    batch = prepare_mtp_batch(tokens, labels, positions, num_mtp_layers=2)

    expected_tokens = torch.tensor(
        [
            [11, 12],
            [12, 0],
            [0, 0],
            [14, 0],
            [0, 0],
            [16, 17],
            [17, 0],
            [0, 0],
        ]
    ).unsqueeze(0)
    expected_valid = expected_tokens.ne(0)
    expected_labels = torch.where(
        expected_valid,
        expected_tokens + 100,
        _community_loss.IGNORE_INDEX,
    )

    torch.testing.assert_close(batch.input_tokens, expected_tokens)
    torch.testing.assert_close(batch.input_valid_mask, expected_valid)
    torch.testing.assert_close(batch.labels, expected_labels)


def test_prepared_mtp_labels_match_community_roll():
    batch_size, seq_len = 1, 8
    labels = torch.arange(seq_len).unsqueeze(0)
    positions = torch.arange(seq_len).unsqueeze(0)
    tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

    mtp_batch = prepare_mtp_batch(tokens, labels, positions, num_mtp_layers=2)
    expected = torch.stack(
        [
            _community_mtp.roll_mtp_sequence(
                labels,
                shift=depth,
                positions=positions,
                fill_value=_community_loss.IGNORE_INDEX,
            )
            for depth in (1, 2)
        ],
        dim=-1,
    )

    torch.testing.assert_close(mtp_batch.labels, expected, rtol=0, atol=0)


@dataclasses.dataclass(frozen=True)
class _MTPChunkedLossCase:
    labels: torch.Tensor
    positions: torch.Tensor
    tokens: torch.Tensor
    hidden: list[torch.Tensor]
    full_hidden: list[torch.Tensor]
    lm_head: torch.nn.Linear
    full_lm_head: torch.nn.Linear
    global_valid_tokens: torch.Tensor
    vocab_size: int


def _build_mtp_chunked_loss_case() -> _MTPChunkedLossCase:
    batch_size, seq_len, dim, vocab_size = 1, 8, 4, 16
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        labels = torch.randint(vocab_size, (batch_size, seq_len))
        tokens = torch.randint(vocab_size, (batch_size, seq_len))
        hidden = [torch.randn(batch_size, seq_len, dim, requires_grad=True) for _ in range(3)]
        lm_head = torch.nn.Linear(dim, vocab_size, bias=False)

    return _MTPChunkedLossCase(
        labels=labels,
        positions=torch.arange(seq_len).unsqueeze(0),
        tokens=tokens,
        hidden=hidden,
        full_hidden=[item.detach().clone().requires_grad_() for item in hidden],
        lm_head=lm_head,
        full_lm_head=copy.deepcopy(lm_head),
        global_valid_tokens=torch.tensor(labels.numel()),
        vocab_size=vocab_size,
    )


def _run_full_mtp_loss(case: _MTPChunkedLossCase) -> torch.Tensor:
    loss_fn = _community_mtp.MTPLoss(
        _community_mtp.MTPLoss.Config(
            mtp_scale=0.3,
            global_vocab_size=case.vocab_size,
        )
    )
    loss, _ = loss_fn(
        [case.full_lm_head(item) for item in case.full_hidden],
        case.labels,
        case.global_valid_tokens,
        positions=case.positions,
    )
    loss.backward()
    return loss


def _run_chunked_mtp_loss(case: _MTPChunkedLossCase) -> torch.Tensor:
    loss_fn = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            num_chunks=2,
            mtp_scale=0.3,
            loss_fn=_community_loss.CrossEntropyLoss.Config(
                global_vocab_size=case.vocab_size,
            ),
        )
    )
    loss_fn.set_lm_head(case.lm_head)
    loss, _ = loss_fn(
        MTPModelOutput(
            case.hidden,
            mtp_labels=prepare_mtp_batch(
                case.tokens,
                case.labels,
                case.positions,
                num_mtp_layers=2,
            ).labels,
        ),
        case.labels,
        case.global_valid_tokens,
    )
    loss.backward()
    return loss


def _assert_chunked_gradients_match(case: _MTPChunkedLossCase) -> None:
    torch.testing.assert_close(
        case.lm_head.weight.grad,
        case.full_lm_head.weight.grad,
        rtol=1e-5,
        atol=1e-6,
    )
    for actual, expected in zip(case.hidden, case.full_hidden, strict=True):
        torch.testing.assert_close(
            actual.grad,
            expected.grad,
            rtol=1e-5,
            atol=1e-6,
        )


def test_mtp_chunked_loss_matches_full_loss_and_gradients():
    case = _build_mtp_chunked_loss_case()

    full_loss = _run_full_mtp_loss(case)
    chunked_loss = _run_chunked_mtp_loss(case)

    torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-6)
    _assert_chunked_gradients_match(case)
