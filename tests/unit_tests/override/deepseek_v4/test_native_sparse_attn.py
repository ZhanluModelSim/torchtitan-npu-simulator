# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native V4 config selection and CPU-visible SMLA adapter contracts."""

import copy
from itertools import pairwise

import pytest
import torch
from torchtitan.config import OverrideConfig, apply_overrides, derive
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan.models.deepseek_v4.attention import DeepSeekV4Attention as UpstreamDeepSeekV4Attention
from torchtitan.models.deepseek_v4.compressor import Indexer, SparseIndexerLoss

from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.attention import DeepSeekV4Attention
from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc
from torchtitan_npu.override.deepseek_v4.sparse_attn.native import (
    AscCompressedSparseAttention,
    AscHeavilyCompressedAttention,
    AscSlidingWindowAttention,
)

pytestmark = pytest.mark.usefixtures("dsv4")


def _model_config():
    return _make_v4_config(
        dim=16,
        n_layers=3,
        vocab_size=32,
        n_heads=2,
        head_dim=8,
        rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=4,
        n_groups=1,
        compress_ratios=(1, 4, 128),
        window_size=4,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        moe_inter_dim=8,
        num_experts=2,
        num_shared_experts=1,
        top_k=1,
        n_hash_layers=0,
        route_norm=False,
        route_scale=1.0,
        load_balance_coeff=1e-3,
        hc_mult=2,
        sinkhorn_iters=2,
        hc_eps=1e-6,
        max_seq_len=256,
        compress_rope_theta=10000.0,
        original_seq_len=256,
    )


def _select(config, *, selector_backend="native"):
    apply_overrides(
        OverrideConfig(
            imports=[
                (
                    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
                    {
                        "num_heads": 2,
                        "head_dim": 8,
                        "index_num_heads": 2,
                        "index_head_dim": 8,
                        "selector_backend": selector_backend,
                    },
                )
            ]
        ),
        config,
    )


def test_native_sparse_override_keeps_outer_attention_indexer_and_state_keys():
    config = _model_config()
    original = copy.deepcopy(config)

    _select(config, selector_backend="asc")

    expected_inner = (AscSlidingWindowAttention, AscCompressedSparseAttention, AscHeavilyCompressedAttention)
    with torch.random.fork_rng(devices=[]):
        for before, after, inner_type in zip(original.layers, config.layers, expected_inner, strict=True):
            assert type(after.attention) is DeepSeekV4Attention.Config
            assert type(after.attention.inner_attention) is inner_type.Config
            if after.attention.indexer is not None:
                assert type(after.attention.indexer) is Indexer.Config
            reference = before.attention.build()
            selected = after.attention.build()
            assert type(selected) is DeepSeekV4Attention
            assert type(selected.inner_attention) is inner_type
            actual = {name: (tuple(value.shape), value.dtype) for name, value in selected.state_dict().items()}
            expected = {name: (tuple(value.shape), value.dtype) for name, value in reference.state_dict().items()}
            assert actual == expected
            assert "attn_sink.weight" in actual
            if after.attention.compress_ratio == 128:
                assert "compressor_128.ape" in actual
        assert type(config.layers[1].attention.inner_attention.build().selector).__name__ == "AscLightningIndexer"


def _dense_kernel_oracle(
    q,
    swa_k,
    attn_sink,
    *,
    attention_metadata,
    softmax_scale,
    compress_ratio,
    window_size,
    cmp_k=None,
    topk_indices=None,
    teacher_carrier=None,
):
    """Enumerate each query's window/selected keys; stand in only for CANN."""
    assert teacher_carrier is None
    offsets = attention_metadata.cu_seq_q.tolist()
    compressed_start = 0
    outputs, lses = [], []
    for start, end in pairwise(offsets):
        for token in range(start, end):
            keys = [swa_k[max(start, token - window_size + 1) : token + 1]]
            if cmp_k is not None:
                if topk_indices is None:
                    chosen = torch.arange((token - start + 1) // compress_ratio)
                else:
                    chosen = topk_indices[token]
                    chosen = chosen[chosen >= 0].long()
                keys.append(cmp_k[compressed_start + chosen])
            values = torch.cat(keys)
            logits = q[token] @ values.T * softmax_scale
            logits = torch.cat((logits, attn_sink[:, None]), dim=-1)
            probabilities = logits.softmax(-1)[:, :-1]
            outputs.append(probabilities @ values)
            lses.append(logits.logsumexp(-1))
        if compress_ratio > 1:
            compressed_start += (end - start) // compress_ratio
    return torch.stack(outputs), torch.stack(lses).unsqueeze(0).detach()


def test_native_reserved_empty_segments_are_removed_only_from_kernel_frames(monkeypatch):
    config = _model_config()
    original = copy.deepcopy(config.layers[1].attention.inner_attention)
    _select(config)
    selected_config = config.layers[1].attention.inner_attention
    positions = torch.arange(16)
    metadata = selected_config.build_attention_metadata(positions, max_num_documents=2, max_context_length=16)
    reference_metadata = original.build_attention_metadata(positions, max_num_documents=2, max_context_length=16)
    torch.testing.assert_close(metadata.cu_seq_q, torch.tensor([0, 16, 16], dtype=torch.int32))
    torch.testing.assert_close(metadata.q.cu_seqlens, torch.tensor([0, 16], dtype=torch.int32))
    torch.testing.assert_close(metadata.swa_k.cu_seqlens, torch.tensor([0, 16], dtype=torch.int32))
    torch.testing.assert_close(metadata.cmp_k.cu_seqlens, torch.tensor([0, 4], dtype=torch.int32))
    assert metadata.max_k == reference_metadata.max_k == 16
    monkeypatch.setattr(ascendc, "run_sparse_flash_mla", _dense_kernel_oracle)
    generator = torch.Generator().manual_seed(43)
    arguments = (
        torch.randn(16, 2, 8, generator=generator),
        torch.randn(16, 8, generator=generator),
        torch.randn(4, 8, generator=generator),
        torch.randn(16, 2, 8, generator=generator),
        torch.randn(4, 8, generator=generator),
        torch.randn(16, 2, generator=generator),
        torch.tensor([-0.3, 0.8]),
    )

    actual = selected_config.build()(*arguments, attention_metadata=metadata)
    expected = original.build()(*arguments, attention_metadata=reference_metadata)

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize(
    ("layer_index", "auxiliary"),
    [(0, False), (1, False), (2, False), (1, True)],
    ids=["swa", "csa", "hca", "csa-native-aux"],
)
def test_native_attention_forward_and_gradients_match_upstream_with_dense_kernel(monkeypatch, layer_index, auxiliary):
    # AuxLoss tracks instance counts globally during construction.
    monkeypatch.setattr(AuxLoss, "_group_counts", copy.deepcopy(AuxLoss._group_counts))
    monkeypatch.setattr(AuxLoss, "group_acc", copy.deepcopy(AuxLoss.group_acc))
    config = _model_config()
    if auxiliary:
        config.layers[1].attention.inner_attention.aux_loss = SparseIndexerLoss.Config(
            coeff=0.17, softmax_scale=8**-0.5, num_heads=2
        )
    original = copy.deepcopy(config.layers[layer_index].attention)
    _select(config)
    selected_config = config.layers[layer_index].attention
    monkeypatch.setattr(ascendc, "run_sparse_flash_mla", _dense_kernel_oracle)
    generator = torch.Generator().manual_seed(71)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        reference = derive(original, UpstreamDeepSeekV4Attention.Config).build()
        reference.init_states()
        selected = selected_config.build()
        selected.load_state_dict(reference.state_dict())
        selected.init_states()
        selected.load_state_dict(reference.state_dict())
    segment_length = 128 if layer_index == 2 else 16
    positions = torch.arange(segment_length).repeat(2)
    padding = torch.zeros(positions.numel(), dtype=torch.bool)
    padding[-2:] = True
    metadata = selected_config.inner_attention.build_attention_metadata(positions, padding_mask=padding)
    native_metadata = original.inner_attention.build_attention_metadata(positions, padding_mask=padding)
    torch.testing.assert_close(
        metadata.cu_seq_q,
        torch.tensor([0, segment_length, 2 * segment_length], dtype=torch.int32),
    )
    assert metadata.max_k == segment_length
    values = torch.randn(positions.numel(), 16, generator=generator) * 0.3
    reference_input = values.clone().requires_grad_()
    selected_input = values.clone().requires_grad_()
    denominator = torch.tensor(positions.numel() - 2, dtype=torch.float32)

    expected = reference(
        reference_input, native_metadata, positions, aux_loss_denominator=denominator, padding_mask=padding
    )
    actual = selected(selected_input, metadata, positions, aux_loss_denominator=denominator, padding_mask=padding)
    upstream = torch.randn(expected.shape, generator=generator)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(selected_input.grad, reference_input.grad, rtol=2e-4, atol=2e-5)
    expected_parameters = dict(reference.named_parameters())
    for name, parameter in selected.named_parameters():
        expected_grad = expected_parameters[name].grad
        if expected_grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=2e-4, atol=2e-5, msg=name)
    if auxiliary:
        assert selected.indexer.wq_b.weight.grad is not None
        assert selected.indexer.wq_b.weight.grad.norm() > 0
