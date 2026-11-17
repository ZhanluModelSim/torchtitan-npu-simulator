# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss

from tests.unit_tests.models.mtp_test_utils import (
    BATCH_SIZE,
    PACKED_POSITIONS,
    SEQ_LEN,
    VOCAB_SIZE,
    assert_packed_mtp_training,
    build_cpu_model,
)
from torchtitan_npu.models.common.linear import SharedLinear
from torchtitan_npu.models.deepseek_v4 import _make_v4_config
from torchtitan_npu.models.deepseek_v4.config_registry import deepseek_v4_debugmodel
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
    return _make_v4_config(
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
    assert_packed_mtp_training(build_cpu_model(_build_model_config()))


def test_deepseek_v4_mtp_metadata_includes_uncompressed_plan():
    model = build_cpu_model(
        _build_model_config(
            compress_ratios=(128, 128, 4),
            rope_head_dim=8,
            original_seq_len=65536,
        )
    )
    tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
    labels = (tokens + 1) % VOCAB_SIZE
    positions = torch.arange(SEQ_LEN).view(BATCH_SIZE, -1)

    _, _, extra_kwargs = model.build_attention_masks(
        tokens,
        labels,
        {"positions": positions},
    )

    assert set(extra_kwargs["attention_masks"].plans) == {1, 4, 128}
    output = model(tokens, **extra_kwargs)
    assert isinstance(output, list) and len(output) == 2


def test_deepseek_v4_mtp_model_supports_chunked_loss():
    model = build_cpu_model(_build_model_config())
    tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
    positions = PACKED_POSITIONS.clone()
    labels = (tokens + 1) % VOCAB_SIZE
    tokens, labels, extra_kwargs = model.build_attention_masks(
        tokens,
        labels,
        {"positions": positions},
    )
    assert "mtp_batch" not in extra_kwargs
    object.__setattr__(model, "_skip_lm_head", True)
    output = model(tokens, **extra_kwargs)

    assert isinstance(output, list) and len(output) == 2
    assert output[0].shape[:2] == (BATCH_SIZE, SEQ_LEN)

    loss_fn = MTPChunkedLossWrapper(
        MTPChunkedLossWrapper.Config(
            num_chunks=2,
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
        )
    )
    loss_fn.set_lm_head(model.lm_head)
    loss, _ = loss_fn(
        output,
        labels,
        torch.tensor(labels.numel()),
        positions=positions,
    )
    assert torch.isfinite(loss).item()

    loss.backward()
    assert any(
        parameter.grad is not None for parameter in model.mtp_layers.parameters()
    )


@pytest.fixture
def single_threaded_cpu():
    # Keep scatter/reduction scheduling fixed for bitwise gradient comparison.
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def test_deepseek_v4_mtp_loss_flag_reaches_model():
    trainer_config = deepseek_v4_debugmodel(num_mtp_layers=1)
    assert not trainer_config.loss.mtp_isolate_gradients
    trainer_config.loss.mtp_isolate_gradients = True
    model_config = trainer_config.model_spec.model
    model_config.update_from_config(config=trainer_config)
    with torch.device("meta"):
        model = model_config.build()

    assert model.mtp_isolate_gradients
    assert isinstance(model.lm_head, SharedLinear)


@pytest.mark.parametrize("missing_head", [False, True], ids=["ordinary_linear", "missing_head"])
def test_deepseek_v4_mtp_isolation_rejects_incompatible_loss_head(missing_head):
    with torch.random.fork_rng():
        loss_fn = MTPChunkedLossWrapper(
            MTPChunkedLossWrapper.Config(
                num_chunks=2,
                loss_fn=CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE),
                mtp_isolate_gradients=True,
            )
        )
        head = torch.nn.Linear(4, VOCAB_SIZE, bias=False)
        if not missing_head:
            loss_fn.set_lm_head(head)
        hidden = [torch.ones(1, 4, 4, requires_grad=True) for _ in range(2)]
        labels = torch.zeros(1, 4, dtype=torch.long)

        with pytest.raises(ValueError, match="mtp_isolate_gradients requires a SharedLinear LM head"):
            loss_fn(hidden, labels, positions=torch.arange(4).unsqueeze(0))

        assert head.weight.grad is None
        assert all(item.grad is None for item in hidden)


def test_deepseek_v4_mtp_isolation_rejects_replaced_model_head():
    with torch.random.fork_rng():
        config = _build_model_config()
        config.mtp_isolate_gradients = True
        model = build_cpu_model(config)
        model.lm_head = torch.nn.Linear(32, VOCAB_SIZE, bias=False)
        tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
        tokens, _, extra_kwargs = model.build_attention_masks(
            tokens, (tokens + 1) % VOCAB_SIZE, {"positions": PACKED_POSITIONS.clone()}
        )

        with pytest.raises(ValueError, match="mtp_isolate_gradients requires a SharedLinear LM head"):
            model(tokens, **extra_kwargs)


@pytest.mark.parametrize("num_mtp_layers", [1, 2], ids=["depth1", "depth2"])
def test_deepseek_v4_mtp_detaches_all_shared_parameters(num_mtp_layers):
    with torch.random.fork_rng():
        config = _build_model_config(num_mtp_layers=num_mtp_layers)
        config.mtp_isolate_gradients = True
        model = build_cpu_model(config)
        tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
        labels = (tokens + 1) % VOCAB_SIZE
        tokens, _, extra_kwargs = model.build_attention_masks(tokens, labels, {"positions": PACKED_POSITIONS.clone()})

        output = model(tokens, **extra_kwargs)
        sum(item.square().sum() for item in output[1:]).backward()

        for name, parameter in model.named_parameters():
            if not name.startswith("mtp_layers."):
                assert parameter.grad is None or not parameter.grad.count_nonzero(), name
        for layer in model.mtp_layers:
            for projection in (layer.e_proj, layer.h_proj):
                assert projection.weight.grad is not None
                assert projection.weight.grad.count_nonzero()


@pytest.mark.parametrize("chunked", [False, True], ids=["logits", "chunked"])
def test_deepseek_v4_mtp_isolation_preserves_main_gradients(chunked, single_threaded_cpu):
    with torch.random.fork_rng():
        config = _build_model_config()
        reference = build_cpu_model(config)
        isolated_config = deepcopy(config)
        isolated_config.mtp_isolate_gradients = True
        model = build_cpu_model(isolated_config)
        model.load_state_dict(reference.state_dict(), strict=True)
        tokens = (torch.arange(SEQ_LEN) % VOCAB_SIZE).view(BATCH_SIZE, -1)
        labels = (tokens + 1) % VOCAB_SIZE
        positions = PACKED_POSITIONS.clone()
        tokens, labels, extra_kwargs = model.build_attention_masks(tokens, labels, {"positions": positions})
        if chunked:
            object.__setattr__(model, "_skip_lm_head", True)
            object.__setattr__(reference, "_skip_lm_head", True)

        output = model(tokens, **extra_kwargs)
        expected = reference(tokens, **extra_kwargs)
        for actual, original in zip(output, expected, strict=True):
            torch.testing.assert_close(actual, original, rtol=0, atol=0)
        loss_config = CrossEntropyLoss.Config(global_vocab_size=VOCAB_SIZE)
        if chunked:
            main_loss_fn = ChunkedLossWrapper(ChunkedLossWrapper.Config(num_chunks=2, loss_fn=loss_config))
            main_loss_fn.set_lm_head(reference.lm_head)
            loss_fn = MTPChunkedLossWrapper(
                MTPChunkedLossWrapper.Config(num_chunks=2, loss_fn=loss_config, mtp_isolate_gradients=True)
            )
            loss_fn.set_lm_head(model.lm_head)
            main_loss, _ = main_loss_fn(expected[0], labels, torch.tensor(labels.numel()))
            loss, _ = loss_fn(output, labels, torch.tensor(labels.numel()), positions=positions)
        else:
            main_loss, _ = CrossEntropyLoss(loss_config)(expected[0], labels)
            loss = CrossEntropyLoss(loss_config)(output[0], labels)[0] + output[1].square().mean()
        main_loss.backward()
        loss.backward()
        assert model.mtp_layers[0].e_proj.weight.grad.count_nonzero()

        for name, parameter in model.named_parameters():
            if name.startswith("mtp_layers."):
                continue
            expected_grad = dict(reference.named_parameters())[name].grad
            assert (parameter.grad is None) == (expected_grad is None), name
            if expected_grad is not None:
                torch.testing.assert_close(
                    parameter.grad, expected_grad, rtol=0, atol=0, msg=lambda message: f"{name}: {message}"
                )


def test_deepseek_v4_mtp_state_dict_round_trip():
    model_config = _build_model_config()
    model = build_cpu_model(model_config)
    local_state_dict = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith("mtp_layers.0.")
    }
    required_local_prefixes = (
        "mtp_layers.0.attention.",
        "mtp_layers.0.hc_attn_pre.",
        "mtp_layers.0.hc_ffn_pre.",
        "mtp_layers.0.hc_head.",
        "mtp_layers.0.moe.router.",
        "mtp_layers.0.moe.shared_experts.",
        "mtp_layers.0.moe.routed_experts.inner_experts.",
    )
    assert all(
        any(key.startswith(prefix) for key in local_state_dict)
        for prefix in required_local_prefixes
    )

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
