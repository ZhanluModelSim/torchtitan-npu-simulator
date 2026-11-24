# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial
from types import SimpleNamespace

import spmd_types as spmd
import torch
from spmd_types.checker import typecheck
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss

from torchtitan_npu.models.deepseek_v4.model import DeepSeekV4Model
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper


def test_forward_routes_untyped_routing_counts_under_spmd():
    # Compute leaves are isolated: this checks real V4 denominator routing,
    # without the upstream DSA PartitionSpec or any distributed communication.
    received = []

    def main_layer(hidden, *args, aux_loss_denominator, **kwargs):
        received.append(aux_loss_denominator)
        return hidden

    def mtp_layer(embed, hidden, *args, aux_loss_denominator, **kwargs):
        received.append(aux_loss_denominator)
        return hidden, hidden.sum(1)

    main_layer.attention = mtp_layer.attention = SimpleNamespace(inner_attention=None)
    model = SimpleNamespace(
        layers={"0": main_layer, "1": main_layer},
        mtp_layers=(mtp_layer, mtp_layer),
        tok_embeddings=lambda x: x.float().unsqueeze(-1),
        hc_mult=2,
        hc_head=lambda x: x.sum(1),
        norm=None,
        lm_head=None,
        _skip_lm_head=True,
        mtp_isolate_gradients=False,
    )
    model._forward_mtp_layers = partial(DeepSeekV4Model._forward_mtp_layers, model)
    tokens, masks = (torch.arange(4),) * 3, (torch.ones(4, dtype=torch.bool),) * 2
    counts = torch.tensor([4, 3, 2])
    with spmd.set_current_mesh({"ep": spmd.MeshAxis.of(2, 1)}):
        for tensor in (*tokens, *masks):
            spmd.assert_type(tensor, {"ep": spmd.R})
        with typecheck():
            output = DeepSeekV4Model.forward(
                model,
                tokens,
                mtp_input_valid_masks=masks,
                attention_metadata={type(None): object()},
                aux_loss_denominators=counts,
            )
    assert [item.item() for item in received] == [4, 4, 3, 2]
    assert len(output) == 3
    for item in output:
        torch.testing.assert_close(item, torch.tensor([[0.0], [2.0], [4.0], [6.0]]))


def test_chunked_loss_splits_untyped_loss_counts_under_spmd(monkeypatch):
    # Isolate chunk projection/backward from upstream layout checks; keep the
    # real MTP wrapper, its scaled loss, and cross entropy consumer.
    received = []

    def branch_loss(self, pred, labels, count, **kwargs):
        received.append(count)
        with spmd.no_typecheck():
            loss, metrics = self.loss_fn(pred, labels, count)
        spmd.assert_type(loss, {"ep": spmd.R})
        return loss, metrics

    monkeypatch.setattr(ChunkedLossWrapper, "__call__", branch_loss)
    labels = (torch.tensor([0, 1, 0, 1]),) * 3
    pred = (torch.zeros(4, 2),) * 3
    counts = torch.tensor([4, 3, 2])
    wrapper = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=2),
            mtp_scale=0.3,
        )
    )
    with spmd.set_current_mesh({"ep": spmd.MeshAxis.of(2, 1)}), typecheck():
        loss, _ = wrapper(pred, labels, counts)
    assert [item.item() for item in received] == [4, 3, 2]
    expected = torch.log(torch.tensor(2.0)) * (1 + 0.15 * (4 / 3 + 4 / 2))
    torch.testing.assert_close(loss, expected)
