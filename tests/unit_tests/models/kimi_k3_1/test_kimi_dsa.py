# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""K3.1 hard selection, selected-key attention gradients and checkpoint state."""

import copy

import pytest
import torch
import torch.nn.functional as F

from torchtitan_npu.models.kimi_k3.config_registry import kimi_k3_debugmodel
from torchtitan_npu.models.kimi_k3_1.config_registry import kimi_k3_1_debugmodel, kimi_k3_1_full
from torchtitan_npu.models.kimi_k3_1.dsa import selected_key_attention
from torchtitan_npu.models.kimi_k3_1.model import KimiK31Model
from torchtitan_npu.models.kimi_k3_1.state_dict_adapter import KimiK31StateDictAdapter


@pytest.mark.parametrize("autocast", [False, True], ids=["fp32", "bf16-autocast"])
def test_indexer_matches_weighted_relu_causal_selection_and_stays_frozen(autocast):
    config = kimi_k3_1_debugmodel().model_spec.model.layers[3].attention.indexer
    config.topk = 2
    module = config.build()
    with torch.random.fork_rng():
        torch.manual_seed(31)
        module.init_states()
        module.k_norm.weight.copy_(torch.linspace(0.7, 1.3, 16))
        module.k_norm.bias.copy_(torch.linspace(-0.2, 0.3, 16))
        hidden = torch.randn(2, 7, 256, requires_grad=True)
        latent = torch.randn(2, 7, 64, requires_grad=True)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        actual = module(hidden, latent)

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        q = F.linear(latent, module.wq_b.weight).reshape(2, 7, 2, 16)
        raw_k = F.linear(hidden, module.wk.weight)
        k = F.layer_norm(raw_k.float(), (16,), module.k_norm.weight, module.k_norm.bias, 1e-5).to(raw_k.dtype)
        weights = F.linear(hidden, module.weights_proj.weight).float() / (32**0.5)
    q, k = q.float(), k.float()
    scores = sum((q[:, :, h] @ k.transpose(-1, -2)).relu() * weights[:, :, h, None] for h in range(2))
    scores = scores.masked_fill(torch.ones(7, 7, dtype=torch.bool).triu(1), -torch.inf)
    values, expected = scores.topk(2)
    expected = expected.masked_fill(~values.isfinite(), -1)
    assert torch.equal(actual, expected)
    assert not actual.requires_grad
    assert all(not parameter.requires_grad for parameter in module.parameters())
    assert all(parameter.grad is None for parameter in module.parameters())


def test_selected_attention_matches_dense_mask_output_and_qkv_gradients_across_chunks():
    generator = torch.Generator().manual_seed(312)
    q = torch.randn(2, 3, 35, 4, generator=generator, requires_grad=True)
    k = torch.randn(2, 3, 35, 4, generator=generator, requires_grad=True)
    v = torch.randn(2, 3, 35, 6, generator=generator, requires_grad=True)
    reference = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    rows = torch.arange(35)
    indices = torch.stack((rows, rows - 1, rows - 2), -1).clamp_min(-1).expand(2, -1, -1)
    dy = torch.randn(2, 3, 35, 6, generator=generator)

    actual = selected_key_attention(q, k, v, indices, 0.5)

    allowed = torch.zeros(2, 35, 35, dtype=torch.bool)
    for batch in range(2):
        for row in range(35):
            allowed[batch, row, indices[batch, row][indices[batch, row] >= 0]] = True
    rq, rk, rv = reference
    scores = (rq @ rk.transpose(-1, -2)) * 0.5
    expected = scores.masked_fill(~allowed[:, None], -torch.inf).softmax(-1) @ rv
    actual.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for tensor, ref in zip((q, k, v), reference, strict=True):
        torch.testing.assert_close(tensor.grad, ref.grad, rtol=1e-5, atol=2e-6)


def test_k3_1_short_sequence_matches_k3_mla_without_indexer_work():
    with torch.random.fork_rng():
        torch.manual_seed(313)
        k3 = kimi_k3_debugmodel().model_spec.model.layers[3].attention.build()
        k3.init_states()
        k31 = kimi_k3_1_debugmodel().model_spec.model.layers[3].attention.build()
        k31.init_states()
        k31.load_state_dict(k3.state_dict(), strict=False)
        hidden = torch.randn(2, 8, 256, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()

    actual = k31(hidden)
    expected = k3(reference_hidden)
    actual.sum().backward()
    expected.sum().backward()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=0, atol=0)
    assert all(p.grad is None for p in k31.indexer.parameters())


def test_k3_1_sparse_mla_matches_independent_dense_mask_oracle_and_parameter_gradients():
    with torch.random.fork_rng():
        torch.manual_seed(314)
        module = kimi_k3_1_debugmodel().model_spec.model.layers[3].attention.build()
        module.init_states()
        reference = copy.deepcopy(module)
        hidden = torch.randn(2, 35, 256, requires_grad=True)
        dy = torch.randn_like(hidden)
    ref_hidden = hidden.detach().clone().requires_grad_()

    actual = module(hidden)

    def norm(value, weight):
        normalized = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + 1e-6)
        return weight * normalized.to(value.dtype)

    latent = norm(F.linear(ref_hidden, reference.q_a_proj.weight), reference.q_a_layernorm.weight)
    q = F.linear(latent, reference.q_b_proj.weight).reshape(2, 35, 8, 32).transpose(1, 2)
    compressed = F.linear(ref_hidden, reference.kv_a_proj_with_mqa.weight)
    kv_latent, k_extra = compressed.split((64, 16), -1)
    kv = F.linear(norm(kv_latent, reference.kv_a_layernorm.weight), reference.kv_b_proj.weight)
    k_base, value = kv.reshape(2, 35, 8, 32).transpose(1, 2).split((16, 16), -1)
    key = torch.cat((k_base, k_extra[:, None].expand(2, 8, 35, 16)), -1)
    indexer = reference.indexer
    with torch.no_grad():
        iq = F.linear(latent, indexer.wq_b.weight).reshape(2, 35, 2, 16)
        ik = F.layer_norm(
            F.linear(ref_hidden, indexer.wk.weight), (16,), indexer.k_norm.weight, indexer.k_norm.bias, 1e-5
        )
        iw = F.linear(ref_hidden, indexer.weights_proj.weight) / (32**0.5)
        scores = sum((iq[:, :, h] @ ik.transpose(-1, -2)).relu() * iw[:, :, h, None] for h in range(2))
        causal = torch.ones(35, 35, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal, -torch.inf)
        chosen = scores.topk(8).indices
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, chosen, True) & causal
    scores = (q @ key.transpose(-1, -2)) / (32**0.5)
    output = scores.masked_fill(~keep[:, None], -torch.inf).softmax(-1) @ value
    output = output.transpose(1, 2).reshape(2, 35, 128)
    output = output * F.linear(ref_hidden, reference.g_proj.weight).sigmoid()
    expected = F.linear(output, reference.o_proj.weight)
    actual.backward(dy)
    expected.backward(dy)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=1e-5, atol=2e-6)
    for name, parameter in module.named_parameters():
        if name.startswith("indexer."):
            assert not parameter.requires_grad and parameter.grad is None
        else:
            assert parameter.grad is not None, name
            torch.testing.assert_close(
                parameter.grad, reference.get_parameter(name).grad, rtol=1e-5, atol=2e-6, msg=name
            )


def test_k3_1_full_geometry_and_indexer_checkpoint_roundtrip():
    spec = kimi_k3_1_full().model_spec
    assert spec.name == "kimi_k3_1" and spec.flavor == "full"
    model = spec.model
    assert isinstance(model, KimiK31Model.Config)
    dsa_layers = [i + 1 for i, layer in enumerate(model.layers) if getattr(layer.attention, "indexer", None)]
    assert len(model.layers) == 93
    assert dsa_layers == [*range(4, 93, 4), 93]
    indexer = model.layers[3].attention.indexer
    assert (indexer.num_heads, indexer.head_dim, indexer.topk) == (32, 128, 2048)
    assert (indexer.wq_b.in_features, indexer.wq_b.out_features) == (1536, 4096)
    assert indexer.wk.in_features == 7168
    original = copy.deepcopy(model)
    indexer.topk = 1
    assert (
        original.layers[3].attention.indexer.topk == kimi_k3_1_full().model_spec.model.layers[3].attention.indexer.topk
    )
    adapter = KimiK31StateDictAdapter(model)
    hf = {
        f"language_model.model.layers.3.self_attn.indexer.{name}": torch.empty(shape, device="meta")
        for name, shape in (
            ("wq_b.weight", (4096, 1536)),
            ("wk.weight", (128, 7168)),
            ("weights_proj.weight", (32, 7168)),
            ("k_norm.weight", (128,)),
            ("k_norm.bias", (128,)),
        )
    }

    converted = adapter.from_hf(hf)
    roundtrip = adapter.to_hf(converted)

    assert len(converted) == len(hf) == len(roundtrip) == 5
    assert all(roundtrip[key] is value for key, value in hf.items())
