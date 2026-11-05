# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Independent behavioral checks for the logical block-diffusion model."""

import pytest
import torch
from raw_reference import (
    BlockDiffusionModel,
    KVCache,
    LayerKV,
    ModelConfig,
    MoE,
    RoutingPlan,
)


def config(**overrides):
    options = dict(
        vocab_size=31,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=24,
        num_experts=0,
        num_shared_experts=0,
        dtype=torch.float32,
        kv_dtype=torch.float32,
    )
    options.update(overrides)
    return ModelConfig(**options)


@pytest.fixture(autouse=True)
def deterministic_torch():
    torch.manual_seed(71)


@torch.no_grad()
def test_cached_causal_execution_matches_full_sequence():
    """A multi-token suffix exposes both bad head layout and RoPE offsets."""
    model = BlockDiffusionModel(config()).eval()
    tokens = torch.tensor([[1, 9, 3, 4, 8, 2], [5, 3, 7, 1, 2, 6]])
    full = model(tokens, phase="prefill", attention_policy="causal")
    prefix = model(tokens[:, :3], phase="prefill", attention_policy="causal")
    suffix = model(tokens[:, 3:], cache=prefix.cache, phase="commit", attention_policy="causal")
    torch.testing.assert_close(suffix.logits, full.logits[:, 3:], atol=2e-6, rtol=2e-5)
    assert suffix.cache.length == tokens.shape[1]
    for actual, expected in zip(suffix.cache.layers, full.cache.layers, strict=False):
        torch.testing.assert_close(actual.key, expected.key, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(actual.value, expected.value, atol=2e-6, rtol=2e-5)


@torch.no_grad()
def test_denoise_is_read_only_commit_is_functional_and_layer_specific():
    model = BlockDiffusionModel(config()).eval()
    prefix = model(torch.tensor([[1, 2, 3]]), phase="prefill", logits_positions=())
    cache = prefix.cache
    saved = [(layer.key.clone(), layer.value.clone()) for layer in cache.layers]
    canvas = torch.tensor([[7, 7]])
    denoised = model(canvas, cache=cache, phase="denoise")
    assert denoised.cache is cache
    committed = model(canvas, cache=cache, phase="commit", logits_positions=())
    assert committed.cache is not cache
    assert cache.length == 3
    assert committed.cache.length == 5
    assert len(committed.cache.layers) == 2
    assert committed.cache.layers[0].key.data_ptr() != committed.cache.layers[1].key.data_ptr()
    for old, (key, value), new in zip(cache.layers, saved, committed.cache.layers, strict=False):
        torch.testing.assert_close(old.key, key)
        torch.testing.assert_close(old.value, value)
        torch.testing.assert_close(new.key[:, :, :3], key)
        torch.testing.assert_close(new.value[:, :, :3], value)


@torch.no_grad()
def test_selected_logits_execute_only_selected_rows_and_none_skips_head():
    model = BlockDiffusionModel(config()).eval()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 2, 3, 1]])
    full = model(tokens).logits
    rows = []
    handle = model.lm_head.register_forward_pre_hook(lambda _module, args: rows.append(args[0].shape))
    selected = model(tokens, logits_positions=(3, 1)).logits
    absent = model(tokens, logits_positions=()).logits
    handle.remove()
    assert absent is None
    assert rows == [torch.Size([2, 2, 16])]
    torch.testing.assert_close(selected, full[:, (3, 1), :])


@pytest.mark.parametrize("tied", [False, True])
def test_parameter_accounting_matches_real_storage_and_ties(tied):
    model = BlockDiffusionModel(config(tie_word_embeddings=tied))
    storages = {}
    for parameter in model.parameters():
        storage = parameter.untyped_storage()
        storages[storage.data_ptr()] = storage.nbytes()
    assert model.parameter_memory_bytes() == sum(storages.values())
    assert (model.lm_head.weight is model.embed_tokens.weight) is tied
    untied = BlockDiffusionModel(config(tie_word_embeddings=False))
    if tied:
        assert untied.parameter_memory_bytes() - model.parameter_memory_bytes() == 31 * 16 * 4


@torch.no_grad()
def test_cache_storage_dtype_and_byte_accounting():
    cfg = config(dtype=torch.float32, kv_dtype=torch.bfloat16)
    model = BlockDiffusionModel(cfg)
    out = model(torch.tensor([[1, 2, 3], [3, 2, 1]]), phase="prefill")
    expected = 2 * 2 * 2 * 3 * 2 * 4 * 2  # K/V, layers, batch, tokens, KV heads, head dim, BF16
    assert out.cache.memory_bytes() == expected
    for layer in out.cache.layers:
        assert layer.key.dtype == torch.bfloat16
        assert layer.value.dtype == torch.bfloat16
    follow = model(torch.tensor([[4], [5]]), cache=out.cache, phase="commit")
    assert follow.logits.dtype == torch.float32
    assert follow.cache.length == 4


def test_routing_plan_conserves_pairs_and_allows_idle_experts():
    plan = RoutingPlan.balanced(num_tokens=2, num_experts=8, top_k=2)
    counts = plan.counts(8)
    assert sum(counts) == 4
    assert sum(count == 0 for count in counts) >= 4
    assert all(len(set(row)) == 2 for row in plan.expert_indices)


@torch.no_grad()
def test_moe_matches_explicit_router_selected_expert_sum():
    cfg = config(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12, num_shared_experts=1)
    moe = MoE(cfg)
    x = torch.randn(2, 3, cfg.hidden_size)
    flat = x.reshape(-1, cfg.hidden_size)
    scores = torch.nn.functional.linear(flat.float(), moe.router.weight.float())
    chosen_logits, choices = scores.topk(2, dim=-1)
    weights = chosen_logits.softmax(-1)
    outputs = []
    for token_id, token in enumerate(flat):
        result = torch.zeros_like(token)
        for slot, expert_id in enumerate(choices[token_id].tolist()):
            result += weights[token_id, slot].to(token.dtype) * moe.experts[expert_id](token.unsqueeze(0)).squeeze(0)
        for expert in moe.shared_experts:
            result += expert(token.unsqueeze(0)).squeeze(0)
        outputs.append(result)
    expected = torch.stack(outputs).reshape_as(x)
    torch.testing.assert_close(moe(x), expected, atol=1e-6, rtol=1e-5)


@torch.no_grad()
def test_moe_replay_uses_specified_experts_and_shared_experts_are_optional():
    cfg = config(num_experts=4, num_experts_per_tok=1, moe_intermediate_size=12)
    moe = MoE(cfg)
    x = torch.randn(1, 3, cfg.hidden_size)
    plan = RoutingPlan(expert_indices=((3,), (0,), (3,)))
    expected = torch.stack([moe.experts[e](x[0, i : i + 1]).squeeze(0) for i, (e,) in enumerate(plan.expert_indices)])
    actual = moe(x, routing_plan=plan)
    torch.testing.assert_close(actual[0], expected, atol=1e-6, rtol=1e-5)
    assert len(moe.shared_experts) == 0


@pytest.mark.parametrize("moe", [False, True])
def test_meta_prefill_denoise_and_commit_need_no_tensor_values(moe):
    cfg = config(num_experts=8 if moe else 0, num_experts_per_tok=2, moe_intermediate_size=12)
    with torch.device("meta"):
        model = BlockDiffusionModel(cfg)
        prefix = torch.empty((2, 3), dtype=torch.long)
        block = torch.empty((2, 2), dtype=torch.long)
        prefill_plan = RoutingPlan.balanced(6, 8, 2) if moe else None
        block_plan = RoutingPlan.balanced(4, 8, 2) if moe else None
        first = model(prefix, phase="prefill", routing_plan=prefill_plan, logits_positions=())
        step = model(block, cache=first.cache, phase="denoise", routing_plan=block_plan, logits_positions=(1,))
        last = model(block, cache=first.cache, phase="commit", routing_plan=block_plan, logits_positions=())
    assert step.logits.shape == (2, 1, 31)
    assert step.logits.device.type == "meta"
    assert first.cache.length == 3
    assert last.cache.length == 5


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_attention_heads": 3, "num_key_value_heads": 2},
        {"head_dim": 3},
        {"num_experts": 2, "num_experts_per_tok": 3},
        {"dtype": torch.int8},
        {"kv_dtype": torch.int8},
        {"hidden_size": 0},
    ],
)
def test_invalid_configs_fail_explicitly(overrides):
    with pytest.raises((ValueError, TypeError)):
        config(**overrides)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"phase": "training"},
        {"attention_policy": "imaginary"},
        {"logits_positions": (3,)},
    ],
)
def test_unsupported_phase_arguments_fail(kwargs):
    model = BlockDiffusionModel(config())
    with pytest.raises((ValueError, TypeError, IndexError)):
        model(torch.tensor([[1, 2, 3]]), **kwargs)


@torch.no_grad()
def test_inconsistent_cache_layers_are_rejected():
    model = BlockDiffusionModel(config())
    cache = model(torch.tensor([[1, 2]]), phase="prefill").cache
    bad_layer = LayerKV(cache.layers[1].key[:, :, :1], cache.layers[1].value[:, :, :1])
    with pytest.raises((ValueError, TypeError)):
        bad = KVCache((cache.layers[0], bad_layer))
        model(torch.tensor([[3]]), cache=bad, phase="commit")


def test_meta_graph_capture_has_no_value_dependent_scalar_extraction():
    from torch.fx.experimental.proxy_tensor import make_fx

    cfg = config(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=12)
    with torch.device("meta"):
        model = BlockDiffusionModel(cfg).eval()
        tokens = torch.empty((1, 3), dtype=torch.long)
    plan = RoutingPlan.balanced(3, 4, 2)
    with torch.no_grad():
        graph = make_fx(lambda ids: model(ids, routing_plan=plan, logits_positions=(2,)).logits)(tokens)
        output = graph(tokens)
    assert output.shape == (1, 1, 31)
    targets = [str(node.target) for node in graph.graph.nodes]
    assert not any("_local_scalar_dense" in name for name in targets)
    assert any("mm" in name for name in targets)


@torch.no_grad()
def test_replay_router_normalization_is_stable_for_low_probability_experts():
    cfg = config(num_experts=2, num_experts_per_tok=1, moe_intermediate_size=12)
    moe = MoE(cfg)
    moe.router.weight[0].fill_(100.0)
    moe.router.weight[1].fill_(-100.0)
    tokens = torch.ones((1, 1, cfg.hidden_size))
    replay = RoutingPlan(expert_indices=((1,),))
    actual = moe(tokens, routing_plan=replay)
    expected = moe.experts[1](tokens)
    assert actual.isfinite().all()
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("policy", ["causal", "bidirectional"])
@torch.no_grad()
def test_attention_matches_independent_gqa_rope_reference_with_prefix(policy):
    from raw_reference import Attention

    cfg = config()
    attention = Attention(cfg)
    batch, length, prefix = 2, 3, 5
    x = torch.randn(batch, length, cfg.hidden_size)
    cache = LayerKV(torch.randn(batch, 2, prefix, 4), torch.randn(batch, 2, prefix, 4))
    actual, current = attention(x, cache, attention_policy=policy, position_offset=prefix)

    # Deliberately materialize GQA heads only in this tiny numerical oracle.
    q = torch.nn.functional.linear(x, attention.q_proj.weight).reshape(batch, length, 4, 4).transpose(1, 2)
    k = torch.nn.functional.linear(x, attention.k_proj.weight).reshape(batch, length, 2, 4).transpose(1, 2)
    v = torch.nn.functional.linear(x, attention.v_proj.weight).reshape(batch, length, 2, 4).transpose(1, 2)
    inverse_frequency = cfg.rope_theta ** (-torch.arange(0, 4, 2).float() / 4)
    angle = torch.arange(prefix, prefix + length).float()[:, None] * inverse_frequency[None, :]
    cos = angle.cos()[None, None]
    sin = angle.sin()[None, None]

    def rotate(tensor):
        first, second = tensor[..., :2], tensor[..., 2:]
        return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)

    q, k = rotate(q), rotate(k)
    torch.testing.assert_close(current.key, k)
    torch.testing.assert_close(current.value, v)
    all_k = torch.cat((cache.key, k), 2).repeat_interleave(2, dim=1)
    all_v = torch.cat((cache.value, v), 2).repeat_interleave(2, dim=1)
    scores = (q @ all_k.transpose(-1, -2)) / 2
    if policy == "causal":
        for query in range(length):
            scores[:, :, query, prefix + query + 1 :] = -torch.inf
    mixed = (scores.softmax(-1) @ all_v).transpose(1, 2).reshape(batch, length, 16)
    expected = torch.nn.functional.linear(mixed, attention.o_proj.weight)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_json_schema_rejects_unknown_fields_and_resolves_dtypes(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"dtype":"bfloat16", "kv_dtype":"float32", "tie_word_embeddings":true}')
    result = ModelConfig.from_json(path)
    assert result.dtype == torch.bfloat16
    assert result.kv_dtype == torch.float32
    assert result.tie_word_embeddings is True
    path.write_text('{"tp_size":8}')
    with pytest.raises((TypeError, ValueError)):
        ModelConfig.from_json(path)


@torch.no_grad()
def test_per_request_positions_match_full_logits_without_unused_head_rows():
    model = BlockDiffusionModel(config()).eval()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    indices = torch.tensor([[3, 1], [0, 2]], dtype=torch.long)
    full = model(tokens).logits
    head_shapes = []
    handle = model.lm_head.register_forward_pre_hook(lambda _module, args: head_shapes.append(args[0].shape))
    selected = model(tokens, logits_positions=indices).logits
    empty = model(tokens, logits_positions=torch.empty((2, 0), dtype=torch.long)).logits
    handle.remove()
    expected = full.gather(1, indices.unsqueeze(-1).expand(2, 2, full.size(-1)))
    torch.testing.assert_close(selected, expected)
    assert head_shapes == [torch.Size([2, 2, 16])]
    assert empty is None


def test_meta_capture_accepts_runtime_per_request_position_map():
    from torch.fx.experimental.proxy_tensor import make_fx

    with torch.device("meta"):
        model = BlockDiffusionModel(config()).eval()
        tokens = torch.empty((2, 4), dtype=torch.long)
        indices = torch.empty((2, 2), dtype=torch.long)
    graph = make_fx(lambda ids, positions: model(ids, logits_positions=positions).logits)(tokens, indices)
    result = graph(tokens, indices)
    assert result.shape == (2, 2, 31)
    targets = [str(node.target) for node in graph.graph.nodes]
    assert any("gather" in target for target in targets)
    assert not any("_local_scalar_dense" in target for target in targets)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2,), torch.long),
        ((1, 2), torch.long),
        ((2, 2, 1), torch.long),
        ((2, 2), torch.float32),
        ((2, 2), torch.int32),
    ],
)
def test_per_request_positions_validate_shape_and_dtype(shape, dtype):
    model = BlockDiffusionModel(config())
    with pytest.raises((ValueError, TypeError)):
        model(torch.ones((2, 4), dtype=torch.long), logits_positions=torch.zeros(shape, dtype=dtype))


def test_per_request_positions_validate_device_and_cpu_bounds():
    model = BlockDiffusionModel(config())
    tokens = torch.ones((2, 4), dtype=torch.long)
    with pytest.raises((ValueError, TypeError)):
        model(tokens, logits_positions=torch.empty((2, 2), dtype=torch.long, device="meta"))
    for bad_index in (-1, 4):
        with pytest.raises((RuntimeError, IndexError)):
            model(tokens, logits_positions=torch.full((2, 1), bad_index, dtype=torch.long))
