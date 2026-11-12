# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Native compressor selection and packed-document arithmetic on CPU."""

import copy

import pytest
import torch
import torch.nn.functional as F
from torchtitan.config import OverrideConfig, apply_overrides
from torchtitan.models.deepseek_v4.compressor import Compressor

from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.override.common.rms_norm import AscRMSNorm
from torchtitan_npu.override.common.rope import AscComplexRoPE
from torchtitan_npu.override.deepseek_v4.compressor import ascendc


def _model_config():
    return _make_v4_config(
        dim=16, n_layers=2, vocab_size=32, n_heads=2, head_dim=8,
        rope_head_dim=4, q_lora_rank=8, o_lora_rank=4, n_groups=1,
        compress_ratios=(4, 128), window_size=128, norm_eps=1e-6,
        index_n_heads=2, index_head_dim=8, index_topk=4,
        moe_inter_dim=8, num_experts=2, num_shared_experts=1, top_k=1,
        n_hash_layers=0, route_norm=False, route_scale=1.0,
        load_balance_coeff=1e-3, hc_mult=2, sinkhorn_iters=2,
        hc_eps=1e-6, max_seq_len=512, compress_rope_theta=10000.0,
        original_seq_len=512,
    )


def test_native_compressor_override_preserves_state_layout_and_composes_children():
    config = _model_config()
    original = copy.deepcopy(config.layers[0].attention.compressor)

    apply_overrides(
        OverrideConfig(imports=["torchtitan_npu.override.deepseek_v4.compressor.asc"]), config
    )

    selected = [node for _, node, _, _ in config.traverse(Compressor.Config)]
    assert len(selected) == 3  # attention C4, its indexer, and HCA
    for node in selected:
        assert type(node) is ascendc.AscNativeCompressor.Config
        assert type(node.norm) is AscRMSNorm.Config
        assert type(node.rope) is AscComplexRoPE.Config
        assert node.rope.dim == node.rope_head_dim
        assert getattr(node.rope, "split", 0) == 0
    assert isinstance(config.layers[0].attention.compressor, Compressor.Config)
    assert type(config.layers[0].attention.indexer).__module__.startswith("torchtitan.models.deepseek_v4")
    with torch.random.fork_rng(devices=[]):
        reference = original.build()
        module = config.layers[0].attention.compressor.build()
    expected = {name: (tuple(value.shape), value.dtype) for name, value in reference.state_dict().items()}
    actual = {name: (tuple(value.shape), value.dtype) for name, value in module.state_dict().items()}
    assert actual == expected
    assert set(dict(module.named_parameters())) == {"ape", "wkv.weight", "wgate.weight", "norm.weight"}


def test_native_compressor_reserved_terminal_slots_select_cann(monkeypatch):
    class KernelSelected(Exception):
        pass

    def selected_kernel():
        raise KernelSelected

    monkeypatch.setattr(ascendc, "load_compressor", selected_kernel)
    config = _model_config()
    apply_overrides(
        OverrideConfig(imports=[(
            "torchtitan_npu.override.deepseek_v4.compressor.asc",
            {"fused_norm": False, "fused_rope": False},
        )]),
        config,
    )
    with torch.random.fork_rng(devices=[]):
        module = config.layers[0].attention.compressor.build()
    boundaries = torch.tensor([0, 4, 8, 8, 8], dtype=torch.int32)

    with pytest.raises(KernelSelected):
        module(torch.ones(8, 16), torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]), boundaries)


def _document_oracle(x, parameters, lengths, ratio, head_dim, rope_dim, theta, eps):
    """Enumerate full document blocks, then upstream's unused fixed-pool slots."""
    values = F.linear(x, parameters["wkv.weight"])
    scores = F.linear(x, parameters["wgate.weight"])
    blocks, positions = [], []
    start = 0
    for length in lengths:
        for offset in range(0, length // ratio * ratio, ratio):
            current = slice(start + offset, start + offset + ratio)
            block_values = values[current, -head_dim:]
            block_scores = scores[current, -head_dim:] + parameters["ape"][:, -head_dim:]
            if ratio == 4 and offset:
                previous = slice(start + offset - ratio, start + offset)
                block_values = torch.cat((values[previous, :head_dim], block_values))
                block_scores = torch.cat((scores[previous, :head_dim] + parameters["ape"][:, :head_dim], block_scores))
            blocks.append((block_values * block_scores.softmax(0)).sum(0))
            positions.append(offset)
        start += length
    unused = x.shape[0] // ratio - len(blocks)
    for index in range(unused):
        block_values = values[:ratio, -head_dim:]
        block_scores = scores[:ratio, -head_dim:] + parameters["ape"][:, -head_dim:]
        if ratio == 4 and index:
            block_values = torch.cat((values[:ratio, :head_dim], block_values))
            block_scores = torch.cat((scores[:ratio, :head_dim] + parameters["ape"][:, :head_dim], block_scores))
        blocks.append((block_values * block_scores.softmax(0)).sum(0))
        positions.append(0)
    pooled = torch.stack(blocks)
    normalized = pooled * (pooled.square().mean(-1, keepdim=True) + eps).rsqrt() * parameters["norm.weight"]
    tail = normalized[:, -rope_dim:].reshape(-1, rope_dim // 2, 2)
    frequencies = theta ** (-torch.arange(0, rope_dim, 2, dtype=x.dtype) / rope_dim)
    angles = torch.tensor(positions, dtype=x.dtype)[:, None] * frequencies
    real = tail[..., 0] * angles.cos() - tail[..., 1] * angles.sin()
    imaginary = tail[..., 1] * angles.cos() + tail[..., 0] * angles.sin()
    rotated = torch.stack((real, imaginary), -1).flatten(-2)
    return torch.cat((normalized[:, :-rope_dim], rotated), -1)


@pytest.mark.parametrize(
    "ratio,lengths",
    [(4, (7, 4)), (4, (3, 3, 5)), (4, (0, 4, 4)), (128, (129, 127))],
    ids=["partial-documents", "unused-pool-slot", "empty-document", "partial-hca"],
)
def test_native_compressor_partial_documents_match_independent_oracle(monkeypatch, ratio, lengths):
    def unexpected_kernel():
        pytest.fail("Partial/empty packed documents must retain native pooling semantics")

    monkeypatch.setattr(ascendc, "load_compressor", unexpected_kernel)
    config = _model_config()
    apply_overrides(
        OverrideConfig(imports=[(
            "torchtitan_npu.override.deepseek_v4.compressor.asc",
            {"fused_norm": False, "fused_rope": False},
        )]),
        config,
    )
    selected = config.layers[0 if ratio == 4 else 1].attention
    cfg = selected.compressor if ratio == 4 else selected.compressor_128
    cfg.rope.scaling = "none"
    with torch.random.fork_rng(devices=[]):
        module = cfg.build()
        module.init_states(buffer_device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(137)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.13)
            if name == "norm.weight":
                parameter.add_(0.9)
    parameters = {}
    for name, value in module.named_parameters():
        parameters[name] = value.detach().double().requires_grad_()
    x = (torch.randn((sum(lengths), 16), generator=generator) * 0.7).requires_grad_()
    reference_x = x.detach().double().requires_grad_()
    boundaries = torch.tensor((0, *torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    positions = torch.cat([torch.arange(length) for length in lengths])

    actual = module(x, positions, boundaries)
    expected = _document_oracle(
        reference_x, parameters, lengths, ratio, module.head_dim, module.rope_head_dim,
        cfg.rope.theta, module.norm.eps,
    )
    upstream_gradient = torch.randn(actual.shape, generator=generator)
    actual.backward(upstream_gradient)
    expected.backward(upstream_gradient.double())

    assert actual.shape == (sum(lengths) // ratio, module.head_dim)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(x.grad.double(), reference_x.grad, rtol=2e-4, atol=2e-5)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter.grad.double(), parameters[name].grad, rtol=2e-4, atol=2e-5)
