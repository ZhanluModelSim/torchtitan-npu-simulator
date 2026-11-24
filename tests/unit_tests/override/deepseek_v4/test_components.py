# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Composition of native V4 component overrides through the trainer config."""

import copy

import torch
from torchtitan.config import apply_overrides
from torchtitan.models.common import ComplexRoPE, RMSNorm
from torchtitan.models.deepseek_v4.compressor import Compressor

from torchtitan_npu.models.deepseek_v4.attention import Attention
from torchtitan_npu.models.deepseek_v4.compressor import Indexer, InnerIndexer
from torchtitan_npu.override.common.rms_norm import AscRMSNorm
from torchtitan_npu.override.common.rope import AscComplexRoPE
from torchtitan_npu.override.deepseek_v4.compressor.ascendc import AscNativeCompressor
from torchtitan_npu.override.deepseek_v4.sparse_attn.ascendc import AscCompressedSparseInnerAttention
from torchtitan_npu_recipes.tests.models.deepseek_v4 import deepseek_v4_flash_43layers_16experts_eager_bf16


def test_component_overrides_keep_native_outer_modules_and_checkpoint_schema():
    config = deepseek_v4_flash_43layers_16experts_eager_bf16(num_mtp_layers=1)
    original = copy.deepcopy(config.model)
    attentions = [node for _, node, _, _ in config.model.traverse(Attention.Config)]
    indexers = [node for _, node, _, _ in config.model.traverse(Indexer.Config)]
    attention_ids, indexer_ids = [id(node) for node in attentions], [id(node) for node in indexers]
    config.override.imports = [
        "torchtitan_npu.override.deepseek_v4.common.rms_norm",
        "torchtitan_npu.override.deepseek_v4.common.rope",
        "torchtitan_npu.override.deepseek_v4.compressor.asc",
        ("torchtitan_npu.override.deepseek_v4.sparse_attn.asc", {"num_heads": 16, "index_num_heads": 8}),
    ]

    apply_overrides(config.override, config)

    assert [id(node) for _, node, _, _ in config.model.traverse(Attention.Config)] == attention_ids
    assert [id(node) for _, node, _, _ in config.model.traverse(Indexer.Config)] == indexer_ids
    assert all(type(node) is Attention.Config for node in attentions)
    assert all(isinstance(node.inner_attention, AscCompressedSparseInnerAttention.Config) for node in attentions)
    assert all(type(node.inner_indexer) is InnerIndexer.Config for node in indexers)
    assert all(type(node) is AscNativeCompressor.Config for _, node, _, _ in config.model.traverse(Compressor.Config))
    assert all(type(node) is AscRMSNorm.Config for _, node, _, _ in config.model.traverse(RMSNorm.Config))
    assert all(type(node) is AscComplexRoPE.Config for _, node, _, _ in config.model.traverse(ComplexRoPE.Config))
    assert all(type(node.q_head_norm) is AscRMSNorm.Config for node in attentions)
    assert all(not node.q_head_norm.elementwise_affine for node in attentions)
    with torch.device("meta"):
        before = original.build().state_dict()
        after = config.model.build().state_dict()
    assert {name: (tensor.shape, tensor.dtype) for name, tensor in after.items()} == {
        name: (tensor.shape, tensor.dtype) for name, tensor in before.items()
    }


def test_query_norm_preserves_reference_bf16_rounding_and_gradients():
    config = deepseek_v4_flash_43layers_16experts_eager_bf16().model.layers[0].attention.q_head_norm
    norm = config.build()
    generator = torch.Generator().manual_seed(42)
    reference = torch.randn(7, 16, 512, generator=generator, dtype=torch.bfloat16, requires_grad=True)
    actual_input = reference.detach().clone().requires_grad_()
    cotangent = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    expected = reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + config.eps)
    actual = norm(actual_input)
    expected.backward(cotangent)
    actual.backward(cotangent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_input.grad, reference.grad, rtol=0, atol=0)
    assert not norm.state_dict()
