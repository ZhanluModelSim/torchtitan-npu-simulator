# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""V4 inner-indexer selection and SMLA contracts."""

import copy
import inspect
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.functional_tensor import dispatch_functionalize
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.config import OverrideConfig, apply_overrides, derive
from torchtitan.experiments.graph_trainer.make_fx_tracer import minimal_fx_tracer, run_traced
from torchtitan.models.deepseek_v4 import attention as reference_attention
from torchtitan.models.deepseek_v4 import compressor as reference_compressor
from torchtitan.models.deepseek_v4.compressor import compressed_cu_seqlens

from torchtitan_npu.models.deepseek_v4.attention import Attention
from torchtitan_npu.models.deepseek_v4.compressor import Indexer, InnerIndexer
from torchtitan_npu.models.deepseek_v4.flavors import _make_dsv4_config as _make_v4_config
from torchtitan_npu.models.deepseek_v4.model import GraphTrainerDeepSeekV4Model
from torchtitan_npu.override.deepseek_v4.lightning_indexer.ascendc import AscInnerIndexer
from torchtitan_npu.override.deepseek_v4.sparse_attn import ascendc
from torchtitan_npu.patches.torchtitan.experiments.graph_trainer.functionalize_recompute_mutations import (
    functionalize_recompute_mutations_pass,
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


def _select(config, *, fused_indexer=False):
    apply_overrides(
        OverrideConfig(
            imports=[
                *(["torchtitan_npu.override.deepseek_v4.lightning_indexer.asc"] if fused_indexer else []),
                (
                    "torchtitan_npu.override.deepseek_v4.sparse_attn.asc",
                    {
                        "num_heads": 2,
                        "head_dim": 8,
                        "index_num_heads": 2,
                        "index_head_dim": 8,
                    },
                ),
            ]
        ),
        config,
    )


@pytest.mark.parametrize("fused_indexer", [False, True], ids=["reference-selection", "fused-selection"])
def test_sparse_overrides_keep_outer_attention_indexer_and_state_keys(fused_indexer):
    config = _model_config()
    original = copy.deepcopy(config)

    _select(config, fused_indexer=fused_indexer)

    with torch.random.fork_rng(devices=[]):
        for before, after in zip(original.layers, config.layers, strict=True):
            assert type(after.attention) is Attention.Config
            if after.attention.indexer is not None:
                assert type(after.attention.indexer) is Indexer.Config
            reference = before.attention.build()
            selected = after.attention.build()
            assert type(selected) is Attention
            assert isinstance(selected.inner_attention, ascendc.AscCompressedSparseInnerAttention)
            actual = {name: (tuple(value.shape), value.dtype) for name, value in selected.state_dict().items()}
            expected = {name: (tuple(value.shape), value.dtype) for name, value in reference.state_dict().items()}
            assert actual == expected
            assert "attn_sink.weight" in actual
            if after.attention.compress_ratio == 128:
                assert "compressor_128.ape" in actual
            if after.attention.indexer is not None:
                assert type(selected.indexer.inner_indexer) is (AscInnerIndexer if fused_indexer else InnerIndexer)


def test_fused_attention_rejects_tp_mesh():
    config = _model_config()
    _select(config)
    inner_attention = config.layers[1].attention.inner_attention.build()
    mesh = SimpleNamespace(get_optional_mesh={"tp": object()}.get)

    with pytest.raises(NotImplementedError, match="TP=1"):
        inner_attention._parallelize(mesh)


def _dense_kernel_oracle(self, q, swa_k, attn_sink, *, attention_metadata, cmp_k, topk_indices, topk_scores, **_):
    """Enumerate each query's window and selected keys as an independent reference."""
    assert topk_scores is None
    window_size, compress_ratio = self.window_size, self.compress_ratio
    offsets = attention_metadata.documents.cu_seq_q.tolist()
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
            logits = q[token] @ values.T * self.softmax_scale
            logits = torch.cat((logits, attn_sink[:, None]), dim=-1)
            probabilities = logits.softmax(-1)[:, :-1]
            outputs.append(probabilities @ values)
            lses.append(logits.logsumexp(-1))
        if compress_ratio > 1:
            compressed_start += (end - start) // compress_ratio
    return torch.stack(outputs), torch.stack(lses).unsqueeze(0).detach()


def test_reserved_empty_segments_are_removed_only_from_kernel_frames(monkeypatch):
    config = _model_config()
    original = derive(config.layers[1].attention.inner_attention, reference_attention.CompressedSparseAttention.Config)
    _select(config)
    selected_config = config.layers[1].attention.inner_attention
    positions = torch.arange(16)
    metadata = selected_config.build_attention_metadata(positions, max_num_documents=2, max_context_length=16)
    reference_metadata = original.build_attention_metadata(positions, max_num_documents=2, max_context_length=16)
    torch.testing.assert_close(metadata.documents.cu_seq_q, torch.tensor([0, 16, 16], dtype=torch.int32))
    torch.testing.assert_close(metadata.q.cu_seqlens, torch.tensor([0, 16], dtype=torch.int32))
    torch.testing.assert_close(metadata.swa_k.cu_seqlens, torch.tensor([0, 16], dtype=torch.int32))
    torch.testing.assert_close(metadata.cmp_k.cu_seqlens, torch.tensor([0, 4], dtype=torch.int32))
    assert metadata.documents.max_k == reference_metadata.max_k == 16
    monkeypatch.setattr(ascendc.AscSparseMLAMixin, "forward", _dense_kernel_oracle)
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

    attention = config.layers[1].attention.build().eval()
    indices, scores = attention.indexer.inner_indexer(*arguments[3:6], attention_metadata=metadata)
    actual = attention.inner_attention(
        arguments[0],
        arguments[1],
        arguments[-1],
        cmp_k=arguments[2],
        topk_indices=indices,
        topk_scores=scores,
        attention_metadata=metadata,
        aux_loss_denominator=torch.tensor(16),
    )
    expected = original.build().eval()(
        *arguments, attention_metadata=reference_metadata, aux_loss_denominator=torch.tensor(16)
    )

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)


def test_graph_trainer_keeps_the_padded_document_boundaries_for_the_compressor():
    config = _model_config()
    _select(config)
    model = derive(config, GraphTrainerDeepSeekV4Model.Config).build()
    positions = torch.cat((torch.arange(128), torch.arange(256)))
    metadata = model._get_attention_metadata(positions, max_num_documents=4, max_context_length=256)
    for entry in metadata.values():
        torch.testing.assert_close(entry.documents.cu_seq_q, torch.tensor([0, 128, 384, 384, 384], dtype=torch.int32))
        torch.testing.assert_close(entry.q.cu_seqlens, torch.tensor([0, 128, 384], dtype=torch.int32))


@pytest.mark.skipif(
    "propagate_input_mutations" not in inspect.signature(dispatch_functionalize).parameters,
    reason="functionalization backport requires propagate_input_mutations",
)
def test_graph_trainer_compressed_offsets_replay_with_a_different_document_count():
    config = _model_config()
    _select(config)
    model = derive(config, GraphTrainerDeepSeekV4Model.Config).build()

    def compressed_offsets(offsets):
        compressed = compressed_cu_seqlens(offsets, ratio=4)
        return torch.empty_like(compressed).copy_(compressed)

    key = config.layers[1].attention.inner_attention.attention_metadata_key
    first = model._get_attention_metadata(torch.arange(256).repeat(3))[key]
    traced = minimal_fx_tracer(compressed_offsets)(first.documents.cu_seq_q)
    for node in traced.gm.graph.nodes:
        if node.op == "call_function":
            node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE
    functional = functionalize_recompute_mutations_pass(traced.gm, traced.example_inputs)
    assert functional is not traced.gm
    traced.gm = functional
    replay = run_traced(traced)

    for doc_len, num_docs in [(384, 2), (128, 6)]:
        metadata = model._get_attention_metadata(torch.arange(doc_len).repeat(num_docs))[key]
        boundaries = torch.arange(num_docs + 1, dtype=torch.int32) * doc_len
        torch.testing.assert_close(metadata.q.cu_seqlens, boundaries)
        expected = (boundaries // 4).new_full((193,), 192)
        expected[: num_docs + 1] = boundaries // 4
        torch.testing.assert_close(replay(metadata.documents.cu_seq_q), expected)


@pytest.mark.parametrize("layer_index", [0, 1, 2], ids=["swa", "csa", "hca"])
def test_attention_forward_and_gradients_match_upstream_with_dense_kernel(monkeypatch, layer_index):
    config = _model_config()
    config.layers[1].attention.inner_attention.aux_loss = None
    original = copy.deepcopy(config.layers[layer_index].attention)
    if original.indexer is not None:
        original.indexer = derive(original.indexer, reference_compressor.Indexer.Config)
        original.inner_attention = derive(
            original.inner_attention, reference_attention.CompressedSparseAttention.Config
        )
    _select(config)
    selected_config = config.layers[layer_index].attention
    monkeypatch.setattr(ascendc.AscSparseMLAMixin, "forward", _dense_kernel_oracle)
    generator = torch.Generator().manual_seed(71)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        reference = derive(original, reference_attention.DeepSeekV4Attention.Config).build()
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
        metadata.documents.cu_seq_q,
        torch.tensor([0, segment_length, 2 * segment_length], dtype=torch.int32),
    )
    assert metadata.documents.max_k == segment_length
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
