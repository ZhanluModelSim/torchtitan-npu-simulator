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

from torchtitan_npu.models.deepseek_v4.mtp import (
    MTPChunkedLossWrapper,
)
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_debugmodel

_community_loss = import_module("torchtitan.components.loss")
_community_mtp = import_module("torchtitan.models.deepseek_v3.mtp")


def test_dsv4_mtp_rejects_cp_configuration():
    config = deepseek_v4_debugmodel(num_mtp_layers=1)
    with torch.device("meta"):
        model = config.model.build()
    parallelism = dataclasses.replace(config.parallelism, context_parallel_degree=2)
    with pytest.raises(NotImplementedError, match="TP=CP=PP=1"):
        model.parallelize(parallelism=parallelism)


@dataclasses.dataclass(frozen=True)
class _MTPChunkedLossCase:
    labels: torch.Tensor
    positions: torch.Tensor
    hidden: list[torch.Tensor]
    full_hidden: list[torch.Tensor]
    lm_head: torch.nn.Linear
    full_lm_head: torch.nn.Linear
    global_loss_token_counts: torch.Tensor
    vocab_size: int


def _build_mtp_chunked_loss_case() -> _MTPChunkedLossCase:
    seq_len, dim, vocab_size = 8, 4, 16
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        labels = torch.randint(vocab_size, (seq_len,))
        hidden = [torch.randn(seq_len, dim, requires_grad=True) for _ in range(3)]
        lm_head = torch.nn.Linear(dim, vocab_size, bias=False)

    return _MTPChunkedLossCase(
        labels=labels,
        positions=torch.arange(seq_len),
        hidden=hidden,
        full_hidden=[item.detach().clone().requires_grad_() for item in hidden],
        lm_head=lm_head,
        full_lm_head=copy.deepcopy(lm_head),
        global_loss_token_counts=torch.tensor([seq_len, seq_len - 1, seq_len - 2]),
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
        tuple(case.full_lm_head(item) for item in case.full_hidden),
        (
            case.labels,
            *(
                _community_mtp.roll_mtp_sequence(case.labels, shift=depth, positions=case.positions, fill_value=-100)
                for depth in range(1, len(case.full_hidden))
            ),
        ),
        case.global_loss_token_counts,
    )
    loss.backward()
    return loss


def _run_chunked_mtp_loss(case: _MTPChunkedLossCase) -> torch.Tensor:
    loss_fn = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            num_chunks=2,
            mtp_scale=0.3,
            loss_fn=_community_loss.CrossEntropyLoss.Config(global_vocab_size=case.vocab_size),
        )
    )
    loss_fn.set_lm_head(case.lm_head)
    labels = tuple(
        torch.cat((case.labels[depth:], case.labels.new_full((depth,), -100))) for depth in range(len(case.hidden))
    )
    loss, _ = loss_fn(tuple(case.hidden), labels, case.global_loss_token_counts)
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
