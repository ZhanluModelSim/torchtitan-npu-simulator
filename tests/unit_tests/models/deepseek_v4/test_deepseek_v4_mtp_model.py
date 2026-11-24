# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.models.deepseek_v3.mtp import MTPDecoder
from torchtitan.models.deepseek_v4.mtp import MTPBlock

from tests.unit_tests.models.mtp_test_utils import (
    PACKED_POSITIONS,
    SEQ_LEN,
    VOCAB_SIZE,
    assert_packed_mtp_training_v04,
    build_cpu_model,
    preprocess_v04,
)
from torchtitan_npu.models.common.linear import SharedLinear
from torchtitan_npu.models.deepseek_v4.flavors import _make_dsv4_config
from torchtitan_npu.models.deepseek_v4.mtp import MTPChunkedLossWrapper
from torchtitan_npu.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
)


def _build_model_config(
    *,
    compress_ratios=(1, 1, 1),
    rope_head_dim=4,
    original_seq_len=SEQ_LEN,
    num_mtp_layers=1,
):
    return _make_dsv4_config(
        dim=32,
        n_layers=3,
        vocab_size=VOCAB_SIZE,
        n_heads=4,
        head_dim=8,
        rope_head_dim=rope_head_dim,
        q_lora_rank=16,
        o_lora_rank=8,
        n_groups=2,
        compress_ratios=compress_ratios,
        window_size=SEQ_LEN,
        norm_eps=1e-6,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        moe_inter_dim=16,
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
        max_seq_len=SEQ_LEN,
        compress_rope_theta=10000.0,
        original_seq_len=original_seq_len,
        num_mtp_layers=num_mtp_layers,
    )


def test_deepseek_v4_mtp_model_supports_packed_sequences():
    assert_packed_mtp_training_v04(build_cpu_model(_build_model_config()))


def test_deepseek_v4_mtp_metadata_includes_uncompressed_plan():
    model = build_cpu_model(
        _build_model_config(
            compress_ratios=(128, 128, 4),
            rope_head_dim=8,
            original_seq_len=65536,
        )
    )
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    labels = (tokens + 1) % VOCAB_SIZE
    positions = torch.arange(SEQ_LEN)

    inputs, _, extra_kwargs = preprocess_v04(model, tokens, labels, positions)

    # One entry per geometry: the stack's ratios -- the MTP depths' uncompressed one included --
    # are one map key each, and each entry carries only its own ratio's plan.
    entries = extra_kwargs["attention_metadata"]

    # Native upstream cores key the map by backend class; the ratio is the one
    # that backend owns in this debug stack.
    from torchtitan.models.deepseek_v4.attention import (
        CompressedSparseAttention,
        HeavilyCompressedAttention,
        SlidingWindowAttention,
    )

    ratio_by_backend = {
        SlidingWindowAttention: 1,
        CompressedSparseAttention: 4,
        HeavilyCompressedAttention: 128,
    }
    assert len(entries) == 3
    assert {ratio_by_backend[key] for key in entries} == {1, 4, 128}
    output = model(inputs, **extra_kwargs)
    assert isinstance(output, tuple) and len(output) == 2
    assert type(model.mtp_layers[0]) is MTPBlock
    assert type(model).preprocess_inputs is MTPDecoder.preprocess_inputs


@pytest.mark.parametrize("num_mtp_layers", [1, 2], ids=["one-depth", "two-depths"])
def test_deepseek_v4_mtp_model_supports_chunked_loss(num_mtp_layers):
    model = build_cpu_model(_build_model_config(num_mtp_layers=num_mtp_layers))
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    positions = PACKED_POSITIONS.clone()
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, labels, extra_kwargs = preprocess_v04(model, tokens, labels, positions)
    assert "mtp_batch" not in extra_kwargs
    object.__setattr__(model, "_skip_lm_head", True)
    hc_transitions = []

    def record_hc_transition(module, args, result):
        hc_transitions.append((args[1], result[0]))

    handles = [layer.register_forward_hook(record_hc_transition) for layer in model.mtp_layers]
    try:
        output = model(tokens, **extra_kwargs)
    finally:
        for handle in handles:
            handle.remove()
    for depth in range(1, num_mtp_layers):
        assert hc_transitions[depth][0] is hc_transitions[depth - 1][1]

    assert isinstance(output, tuple) and len(output) == num_mtp_layers + 1
    assert all(item.shape[0] == SEQ_LEN for item in output)

    loss_fn = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
        )
    )
    loss_fn.set_lm_head(model.lm_head)
    counts = torch.tensor([SEQ_LEN - 2 * depth for depth in range(num_mtp_layers + 1)])
    expected_loss = sum(
        (1.0 if depth == 0 else 0.3 / num_mtp_layers)
        * torch.nn.functional.cross_entropy(model.lm_head(hidden).float(), target, reduction="sum")
        / count
        for depth, (hidden, target, count) in enumerate(zip(output, labels, counts, strict=True))
    )
    loss, _ = loss_fn(output, labels, counts)
    torch.testing.assert_close(loss, expected_loss)

    loss.backward()
    for layer in model.mtp_layers:
        assert layer.e_proj.weight.grad is not None
        assert layer.e_proj.weight.grad.count_nonzero()


def test_deepseek_v4_mtp_gradient_isolation_detaches_shared_parameters():
    config = _build_model_config()
    config.mtp_isolate_gradients = True
    model = build_cpu_model(config)

    assert isinstance(model.lm_head, SharedLinear)
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, _, extra_kwargs = preprocess_v04(model, tokens, labels, PACKED_POSITIONS.clone())

    output = model(tokens, **extra_kwargs)
    sum(item.square().sum() for item in output[1:]).backward()

    for name, parameter in model.named_parameters():
        if not name.startswith("mtp_layers."):
            assert parameter.grad is None or not parameter.grad.count_nonzero(), name
    assert model.mtp_layers[0].e_proj.weight.grad is not None


def test_deepseek_v4_mtp_state_dict_round_trip():
    model_config = _build_model_config()
    model = build_cpu_model(model_config)
    local_state_dict = {
        key: value.detach().clone() for key, value in model.state_dict().items() if key.startswith("mtp_layers.0.")
    }
    required_local_prefixes = (
        "mtp_layers.0.attention.",
        "mtp_layers.0.hc_attn_pre.",
        "mtp_layers.0.hc_ffn_pre.",
        "mtp_layers.0.hc_head.",
        "mtp_layers.0.feed_forward.router.",
        "mtp_layers.0.feed_forward.shared_experts.",
        "mtp_layers.0.feed_forward.routed_experts.",
    )
    assert all(any(key.startswith(prefix) for key in local_state_dict) for prefix in required_local_prefixes)

    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)
    hf_state_dict = adapter.to_hf(local_state_dict)
    required_hf_keys = {
        "mtp.0.e_proj.weight",
        "mtp.0.h_proj.weight",
        "mtp.0.enorm.weight",
        "mtp.0.hnorm.weight",
        "mtp.0.norm.weight",
        "mtp.0.hc_head_fn",
        "mtp.0.attn.wq_a.weight",
        "mtp.0.ffn.gate.weight",
        "mtp.0.ffn.shared_experts.w1.weight",
        "mtp.0.ffn.experts.0.w1.weight",
    }
    assert required_hf_keys <= hf_state_dict.keys()
    assert all(key.startswith("mtp.0.") for key in hf_state_dict)

    restored_state_dict = adapter.from_hf(hf_state_dict)
    assert restored_state_dict.keys() == local_state_dict.keys()
    for key, expected in local_state_dict.items():
        torch.testing.assert_close(
            restored_state_dict[key],
            expected,
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("method_name", "key"),
    (
        ("from_hf", "mtp.1.e_proj.weight"),
        ("to_hf", "mtp_layers.1.e_proj.weight"),
    ),
)
def test_deepseek_v4_mtp_state_dict_rejects_invalid_depth(method_name, key):
    model_config = _build_model_config()
    adapter = DeepSeekV4StateDictAdapter(model_config, hf_assets_path=None)

    with pytest.raises(ValueError, match="not present"):
        getattr(adapter, method_name)({key: torch.ones(1)})


def test_deepseek_v4_without_mtp_returns_main_logits():
    config = _build_model_config(num_mtp_layers=0)
    config.mtp_layers = None
    model = build_cpu_model(config)
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    inputs, labels, kwargs = preprocess_v04(model, tokens, (tokens + 1) % VOCAB_SIZE, PACKED_POSITIONS.clone())
    output = model(inputs, **kwargs)
    assert model.mtp_layers is None
    assert isinstance(output, torch.Tensor)
    assert output.shape == (SEQ_LEN, VOCAB_SIZE)
    torch.nn.functional.cross_entropy(output.float(), labels).backward()
    assert model.lm_head.weight.grad is not None


def test_deepseek_v4_isolated_chunked_loss_preserves_main_gradients():
    config = _build_model_config(num_mtp_layers=2)
    config.mtp_isolate_gradients = True
    model = build_cpu_model(config)
    model._skip_lm_head = True
    tokens = torch.arange(SEQ_LEN) % VOCAB_SIZE
    inputs, labels, kwargs = preprocess_v04(model, tokens, (tokens + 1) % VOCAB_SIZE, PACKED_POSITIONS.clone())
    output = model(inputs, **kwargs)
    counts = torch.tensor([SEQ_LEN, SEQ_LEN - 2, SEQ_LEN - 4])
    # Save the main-only gradients, then compare after all objectives backpropagate.
    main_loss = (
        torch.nn.functional.cross_entropy(model.lm_head(output[0]).float(), labels[0], reduction="sum") / counts[0]
    )
    main_loss.backward(retain_graph=True)
    expected_grads = {
        name: param.grad.clone()
        for name, param in model.named_parameters()
        if not name.startswith("mtp_layers.") and param.grad is not None
    }
    model.zero_grad(set_to_none=True)
    loss_fn = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            num_chunks=2,
            mtp_isolate_gradients=True,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
        )
    )
    loss_fn.set_lm_head(model.lm_head)
    loss, _ = loss_fn(output, labels, counts)
    loss.backward()
    for name, param in model.named_parameters():
        if name.startswith("mtp_layers."):
            continue
        if name in expected_grads:
            torch.testing.assert_close(param.grad, expected_grads[name], rtol=1e-5, atol=1e-6)
        else:
            assert param.grad is None or not param.grad.count_nonzero(), name
    for layer in model.mtp_layers:
        assert layer.e_proj.weight.grad is not None
        assert layer.e_proj.weight.grad.count_nonzero()
